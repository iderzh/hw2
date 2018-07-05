/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>

#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000
#define NREQUESTS 4
#define NSTREAMS 64
#define MAXREGCOUNT 32
#define QUEUENODES 10

typedef unsigned char uchar;
#define OUT

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}

struct stream_node {
	cudaStream_t Stream;
	int stream_id;
	int req_in_processing;
};
typedef stream_node streamNode;

typedef struct _thread_node {
	int req_id;
	float result;
} thread_node;

typedef struct _threads_queue {
	uchar read_index;
	uchar write_index;
	thread_node queue_array[QUEUENODES];
} threads_queue;

/* we'll use these to rate limit the request load */
struct rate_limit_t {
    double last_checked;
    double lambda;
    unsigned seed;
};

void rate_limit_init(struct rate_limit_t *rate_limit, double lambda, int seed) {
    rate_limit->lambda = lambda;
    rate_limit->seed = (seed == -1) ? 0 : seed;
    rate_limit->last_checked = 0;
}

int rate_limit_can_send(struct rate_limit_t *rate_limit) {
    if (rate_limit->lambda == 0) return 1;
    double now = get_time_msec() * 1e-3;
    double dt = now - rate_limit->last_checked;
    double p = dt * rate_limit->lambda;
    rate_limit->last_checked = now;
    if (p > 1) p = 1;
    double r = (double)rand_r(&rate_limit->seed) / RAND_MAX;
    return (p > r);
}

void rate_limit_wait(struct rate_limit_t *rate_limit) {
    while (!rate_limit_can_send(rate_limit)) {
        usleep(1. / (rate_limit->lambda * 1e-6) * 0.01);
    }
}

/* we won't load actual files. just fill the images with random bytes */
void load_image_pairs(uchar *images1, uchar *images2) {
    srand(0);
    for (int i = 0; i < N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION; i++) {
        images1[i] = rand() % 256;
        images2[i] = rand() % 256;
    }
}

__device__ __host__ bool is_in_image_bounds(int i, int j) {
    return (i >= 0) && (i < IMG_DIMENSION) && (j >= 0) && (j < IMG_DIMENSION);
}

__device__ __host__ uchar local_binary_pattern(uchar *image, int i, int j) {
    uchar center = image[i * IMG_DIMENSION + j];
    uchar pattern = 0;
    if (is_in_image_bounds(i - 1, j - 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
    if (is_in_image_bounds(i - 1, j    )) pattern |= (image[(i - 1) * IMG_DIMENSION + (j    )] >= center) << 6;
    if (is_in_image_bounds(i - 1, j + 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
    if (is_in_image_bounds(i    , j + 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j + 1)] >= center) << 4;
    if (is_in_image_bounds(i + 1, j + 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
    if (is_in_image_bounds(i + 1, j    )) pattern |= (image[(i + 1) * IMG_DIMENSION + (j    )] >= center) << 2;
    if (is_in_image_bounds(i + 1, j - 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
    if (is_in_image_bounds(i    , j - 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j - 1)] >= center) << 0;
    return pattern;
}

void image_to_histogram(uchar *image, int *histogram) {
    memset(histogram, 0, sizeof(int) * 256);
    for (int i = 0; i < IMG_DIMENSION; i++) {
        for (int j = 0; j < IMG_DIMENSION; j++) {
            uchar pattern = local_binary_pattern(image, i, j);
            histogram[pattern]++;
        }
    }
}

double histogram_distance(int *h1, int *h2) {
    /* we'll use the chi-square distance */
    float distance = 0;
    for (int i = 0; i < 256; i++) {
        if (h1[i] + h2[i] != 0) {
            distance += ((double)SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
        }
    }
    return distance;
}

__global__ void gpu_image_to_histogram(uchar *image, int *histogram) {
    uchar pattern = local_binary_pattern(image, threadIdx.x / IMG_DIMENSION, threadIdx.x % IMG_DIMENSION);
    atomicAdd(&histogram[pattern], 1);
}

__global__ void gpu_histogram_distance(int *h1, int *h2, double *distance) {
    int length = 256;
    int tid = threadIdx.x;
    distance[tid] = 0;
    if (h1[tid] + h2[tid] != 0) {
        distance[tid] = ((double)SQR(h1[tid] - h2[tid])) / (h1[tid] + h2[tid]);
    }
    __syncthreads();

    while (length > 1) {
        if (threadIdx.x < length / 2) {
            distance[tid] = distance[tid] + distance[tid + length / 2];
        }
        length /= 2;
        __syncthreads();
    }
}

__global__ void process_queues (volatile threads_queue *dev_gpu_cpu_queues, volatile threads_queue *dev_cpu_gpu_queues, const int nrequests) {
	  int req_count = 0;
	  int req_id;
	  while (req_count < nrequests ){
		  // Dequeue request
		  if (dev_cpu_gpu_queues[blockIdx.x].read_index != dev_cpu_gpu_queues[blockIdx.x].write_index) {
			  req_id = dev_cpu_gpu_queues[blockIdx.x].queue_array[dev_cpu_gpu_queues[blockIdx.x].read_index].req_id;
			  dev_cpu_gpu_queues[blockIdx.x].read_index = (dev_cpu_gpu_queues[blockIdx.x].read_index + 1) % QUEUENODES;
			  printf("GPU: Req #%d was dequeued in TB: %d", req_id, blockIdx.x);
		  } else { return; }
		  __threadfence_system();

      	if (dev_gpu_cpu_queues[blockIdx.x].read_index != (dev_gpu_cpu_queues[blockIdx.x].write_index + 1) % QUEUENODES) {
      		// Enqueue
      		dev_gpu_cpu_queues[blockIdx.x].queue_array[dev_gpu_cpu_queues[blockIdx.x].write_index].req_id = req_id;
      		dev_gpu_cpu_queues[blockIdx.x].write_index = (dev_gpu_cpu_queues[blockIdx.x].write_index + 1) % QUEUENODES;
  			printf("GPU: write index #%d was updated by thread %d\n", dev_gpu_cpu_queues[blockIdx.x].write_index, blockIdx.x);
      	}
      	__threadfence_system();
	  }
}

void print_usage_and_die(char *progname) {
    printf("usage:\n");
    printf("%s streams <load (requests/sec)>\n", progname);
    printf("OR\n");
    printf("%s queue <#threads> <load (requests/sec)>\n", progname);
    exit(1);
}


enum {PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE};
int main(int argc, char *argv[]) {

    int mode = -1;
    int threads_queue_mode = -1; /* valid only when mode = queue */
    double load = 0;
    if (argc < 3) print_usage_and_die(argv[0]);

    if        (!strcmp(argv[1], "streams")) {
        if (argc != 3) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_STREAMS;
        load = atof(argv[2]);
    } else if (!strcmp(argv[1], "queue")) {
        if (argc != 4) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_QUEUE;
        threads_queue_mode = atoi(argv[2]);
        load = atof(argv[3]);
    } else {
        print_usage_and_die(argv[0]);
    }

    uchar *images1; /* we concatenate all images in one huge array */
    uchar *images2;
    CUDA_CHECK( cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
    CUDA_CHECK( cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );

    load_image_pairs(images1, images2);
    double t_start, t_finish;
    double total_distance;
#if 1
    /* using CPU */
    printf("\n=== CPU ===\n");
    int histogram1[256];
    int histogram2[256];
    t_start  = get_time_msec();
    for (int i = 0; i < NREQUESTS; i++) {
        int img_idx = i % N_IMG_PAIRS;
        image_to_histogram(&images1[img_idx * IMG_DIMENSION * IMG_DIMENSION], histogram1);
        image_to_histogram(&images2[img_idx * IMG_DIMENSION * IMG_DIMENSION], histogram2);
        total_distance += histogram_distance(histogram1, histogram2);
    }
    t_finish = get_time_msec();
    printf("average distance between images %f\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

    /* using GPU task-serial.. just to verify the GPU code makes sense */
    printf("\n=== GPU Task Serial ===\n");
    do {
        uchar *gpu_image1, *gpu_image2; // TODO: allocate with cudaMalloc
        int *gpu_hist1, *gpu_hist2; // TODO: allocate with cudaMalloc
        double *gpu_hist_distance; //TODO: allocate with cudaMalloc
        double cpu_hist_distance;
        cudaMalloc(&gpu_image1, IMG_DIMENSION * IMG_DIMENSION);
        cudaMalloc(&gpu_image2, IMG_DIMENSION * IMG_DIMENSION);
        cudaMalloc(&gpu_hist1, 256 * sizeof(int));
        cudaMalloc(&gpu_hist2, 256 * sizeof(int));
        cudaMalloc(&gpu_hist_distance, 256 * sizeof(double));

        total_distance = 0;
        t_start = get_time_msec();
        for (int i = 0; i < NREQUESTS; i++) {
            int img_idx = i % N_IMG_PAIRS;
            cudaMemcpy(gpu_image1, &images1[img_idx * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_image2, &images2[img_idx * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice);
            cudaMemset(gpu_hist1, 0, 256 * sizeof(int));
            cudaMemset(gpu_hist2, 0, 256 * sizeof(int));
            gpu_image_to_histogram<<<1, 1024>>>(gpu_image1, gpu_hist1);
            gpu_image_to_histogram<<<1, 1024>>>(gpu_image2, gpu_hist2);
            gpu_histogram_distance<<<1, 256>>>(gpu_hist1, gpu_hist2, gpu_hist_distance);
            cudaMemcpy(&cpu_hist_distance, gpu_hist_distance, sizeof(double), cudaMemcpyDeviceToHost);
            total_distance += cpu_hist_distance;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        t_finish = get_time_msec();
        printf("average distance between images %f\n", total_distance / NREQUESTS);
        printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);
        CUDA_CHECK( cudaFree(gpu_image1) );
        CUDA_CHECK( cudaFree(gpu_image2) );
        CUDA_CHECK( cudaFree(gpu_hist1) );
        CUDA_CHECK( cudaFree(gpu_hist2) );
        CUDA_CHECK( cudaFree(gpu_hist_distance) );
    } while (0);
#endif
    /* now for the client-server part */
    printf("\n=== Client-Server ===\n");
    total_distance = 0;
    double *req_t_start = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_start, 0, NREQUESTS * sizeof(double));

    double *req_t_end = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_end, 0, NREQUESTS * sizeof(double));

    struct rate_limit_t rate_limit;
    rate_limit_init(&rate_limit, load, 0);

    /* TODO allocate / initialize memory, streams, etc... */
    uchar *gpu_image1, *gpu_image2; // TODO: allocate with cudaMalloc
    int *gpu_hist1, *gpu_hist2; // TODO: allocate with cudaMalloc
    double *gpu_hist_distance; //TODO: allocate with cudaMalloc
    double *cpu_hist_distance;

    streamNode streams_array[NSTREAMS] = {0};
    int free_streams = NSTREAMS;

    double ti = get_time_msec();
    if (mode == PROGRAM_MODE_STREAMS) {
    	// Allocate CUDA memory for STREAMS
        CUDA_CHECK(cudaMalloc(&gpu_image1, IMG_DIMENSION * IMG_DIMENSION * NSTREAMS));
        CUDA_CHECK(cudaMalloc(&gpu_image2, IMG_DIMENSION * IMG_DIMENSION * NSTREAMS));
        CUDA_CHECK(cudaMalloc(&gpu_hist1, 256 * sizeof(int) * NSTREAMS));
        CUDA_CHECK(cudaMalloc(&gpu_hist2, 256 * sizeof(int) * NSTREAMS));
        CUDA_CHECK(cudaMalloc(&gpu_hist_distance, 256 * sizeof(double) * NSTREAMS));
        CUDA_CHECK(cudaHostAlloc(&cpu_hist_distance, sizeof(double), 0));

        // Init array of stream nodes
        for (int j = 0; j < NSTREAMS; j++) {
        	streams_array[j].stream_id = j;
        	streams_array[j].req_in_processing = -1;
        	CUDA_CHECK( cudaStreamCreate(&streams_array[j].Stream));
        }

        for (int i = 0; i < NREQUESTS; i++) {

            /* TODO query (don't block) streams for any completed requests.
               update req_t_end of completed requests
               update total_distance */

        	for (int j = 0; j < NSTREAMS; j++) {
        		if ( streams_array[j].req_in_processing != -1) {
        			if ( cudaStreamQuery(streams_array[j].Stream) == cudaSuccess) {
        				req_t_end[streams_array[j].req_in_processing] = get_time_msec();
        				CUDA_CHECK( cudaMemcpyAsync(cpu_hist_distance, &(gpu_hist_distance[streams_array[j].stream_id * 256]), sizeof(double), cudaMemcpyDeviceToHost, streams_array[j].Stream));
        				total_distance += *cpu_hist_distance;
        				streams_array[j].req_in_processing = -1;
        				free_streams++;
        			}
        		}
        	}

            rate_limit_wait(&rate_limit);
            req_t_start[i] = get_time_msec();
            int img_idx = i % N_IMG_PAIRS;

            /* TODO place memcpy's and kernels in a stream */
            if (free_streams > 0) {

            	// Find first free stream
            	streamNode *busy_streams;
            	for (int j = 0; j < NSTREAMS; j++) {
            		if (streams_array[j].req_in_processing == -1) {
            			busy_streams = &streams_array[j];
            			break;
            		}
            	}

            	busy_streams->req_in_processing = i;
            	free_streams--;

                // Enqueue data copy and kernel execution for selected stream
                CUDA_CHECK(cudaMemcpyAsync(&(gpu_image1[busy_streams->stream_id * IMG_DIMENSION * IMG_DIMENSION]), &images1[img_idx * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice, busy_streams->Stream));
                CUDA_CHECK(cudaMemcpyAsync(&(gpu_image2[busy_streams->stream_id * IMG_DIMENSION * IMG_DIMENSION]), &images2[img_idx * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice, busy_streams->Stream));
                CUDA_CHECK(cudaMemsetAsync(&(gpu_hist1[busy_streams->stream_id * 256]), 0, 256 * sizeof(int), busy_streams->Stream));
                CUDA_CHECK(cudaMemsetAsync(&(gpu_hist2[busy_streams->stream_id * 256]), 0, 256 * sizeof(int), busy_streams->Stream));

                gpu_image_to_histogram<<<1, 1024, 0, busy_streams->Stream>>>(&(gpu_image1[busy_streams->stream_id * IMG_DIMENSION * IMG_DIMENSION]), &(gpu_hist1[busy_streams->stream_id * 256]));
                gpu_image_to_histogram<<<1, 1024, 0, busy_streams->Stream>>>(&(gpu_image2[busy_streams->stream_id * IMG_DIMENSION * IMG_DIMENSION]), &(gpu_hist2[busy_streams->stream_id * 256]));
                gpu_histogram_distance<<<1, 256, 0, busy_streams->Stream>>>(&(gpu_hist1[busy_streams->stream_id * 256]), &(gpu_hist2[busy_streams->stream_id * 256]), &(gpu_hist_distance[busy_streams->stream_id * 256]));
            }
        }
        /* TODO now make sure to wait for all streams to finish */
    	for (int j = 0; j < NSTREAMS; j++) {
    		if (streams_array[j].req_in_processing != -1) {
    			CUDA_CHECK( cudaStreamSynchronize(streams_array[j].Stream) );
    			req_t_end[streams_array[j].req_in_processing] = get_time_msec();
    		}
    	}

        for (int j = 0; j < NSTREAMS; j++) {
        	CUDA_CHECK( cudaStreamDestroy(streams_array[j].Stream));
        }

        CUDA_CHECK( cudaFree(gpu_hist_distance) );
        CUDA_CHECK( cudaFreeHost(cpu_hist_distance) );

    } else if (mode == PROGRAM_MODE_QUEUE) {
    	// Check for CUDA device and calculate amount of CPU<->GPU queues accordingly to it's capabilities
    	int deviceCount = 0, cuda_device = 0;
    	CUDA_CHECK( cudaGetDeviceCount(&deviceCount) );
    	if (deviceCount > 0) {
    		printf("CUDA Device(s) found, will use first available device: ");
    	} else {
    		printf("No CUDA Device found, terminating the program!\n");
    		assert(0);
    	}
    	cudaSetDevice(cuda_device);
    	cudaDeviceProp deviceProp;
    	cudaGetDeviceProperties(&deviceProp, cuda_device);
    	printf("%s\n", deviceProp.name);
    	if (	threads_queue_mode > IMG_DIMENSION * IMG_DIMENSION  ||
    			threads_queue_mode <= 0    ||
    			deviceProp.maxThreadsPerBlock < IMG_DIMENSION * IMG_DIMENSION
    		)
    	{
    		printf("Wrong amount of threads requested for 32x32 images or your device incapable to run 1024 threads in one block,\nPlease enter #threads = 1024 or less.\n");
    		assert (0);
    	}

    	if ( deviceProp.canMapHostMemory == 0){
    		printf("Your CUDA Device doesn't support cudaDeviceMapHost, terminating the program!\n");
    		assert(0);
    	}

    	unsigned int max_simult_blocks = deviceProp.multiProcessorCount * (deviceProp.maxThreadsPerMultiProcessor / threads_queue_mode);
    	printf("This device is capable to run %d thread blocks simultaneously.\n", max_simult_blocks);
    	if ( (deviceProp.regsPerBlock / (max_simult_blocks * threads_queue_mode)) < MAXREGCOUNT) {
    		max_simult_blocks = deviceProp.regsPerBlock / ( threads_queue_mode * MAXREGCOUNT );
    		printf("Amount of running blocks (queue pairs) was reduced to %d due to device Registers limitation\n", max_simult_blocks);
    	}
    	if ( (deviceProp.sharedMemPerBlock / (2 * (IMG_DIMENSION * IMG_DIMENSION) + 2 * sizeof (int) * 256 + sizeof (double) * 256)) < 1) {
    		printf("No enough Shared memory per block, terminating the program!\n");
    		assert(0);
    	}

    	// Create CPU<->GPU queues
    	volatile threads_queue *cpu_gpu_queues;
    	volatile threads_queue *gpu_cpu_queues;
    	CUDA_CHECK( cudaHostAlloc(&gpu_cpu_queues, sizeof (threads_queue) * max_simult_blocks, cudaHostAllocMapped) );
		CUDA_CHECK( cudaHostAlloc(&cpu_gpu_queues, sizeof (threads_queue) * max_simult_blocks, cudaHostAllocMapped) );

    	for (int i = 0; i < max_simult_blocks; i++) {
    		cpu_gpu_queues[i].read_index = 0;
    		cpu_gpu_queues[i].write_index = 0;
    		gpu_cpu_queues[i].read_index = 0;
    		gpu_cpu_queues[i].write_index = 0;
    		//CUDA_CHECK( cudaHostGetDevicePointer (&(cpu_gpu_queues[i].read_devp), cpu_gpu_queues[i].queue_array, 0) );
    	}


    	float *gpu_total_distance_f;
    	thread_node *cpu_hist_distance_node;

        CUDA_CHECK( cudaMalloc(&gpu_image1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION) );
        CUDA_CHECK( cudaMalloc(&gpu_image2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION) );
        CUDA_CHECK( cudaMalloc(&gpu_hist1, sizeof (int) * 256 * N_IMG_PAIRS) );
        CUDA_CHECK( cudaMalloc(&gpu_hist2, sizeof (int) * 256 * N_IMG_PAIRS) );
        CUDA_CHECK( cudaMalloc(&gpu_total_distance_f, sizeof (float) * 256 * max_simult_blocks) );
        CUDA_CHECK( cudaHostAlloc(&cpu_hist_distance_node, sizeof(thread_node), 0) );

        CUDA_CHECK( cudaMemcpy(gpu_image1, images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(gpu_image2, images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemset(gpu_total_distance_f,0 , sizeof(float) * 256 * max_simult_blocks));

        volatile threads_queue *dev_cpu_gpu_queues, *dev_gpu_cpu_queues;
        CUDA_CHECK( cudaHostGetDevicePointer ((void **)&dev_gpu_cpu_queues, (void *)gpu_cpu_queues, 0) );
        CUDA_CHECK( cudaHostGetDevicePointer ((void **)&dev_cpu_gpu_queues, (void *)cpu_gpu_queues, 0) );

        //Start CUDA kernel
        process_queues<<<max_simult_blocks, threads_queue_mode>>>(dev_gpu_cpu_queues, dev_cpu_gpu_queues, NREQUESTS);

        for (int i = 0; i < NREQUESTS; ) {

            /* TODO check producer consumer queue for any responses.
               don't block. if no responses are there we'll check again in the next iteration
               update req_t_end of completed requests
               update total_distance */
        	for (int j = 0; j < max_simult_blocks; j++) {
        		// Get current GPU->CPU queue indexes
        		uchar gpu_cpu_read_index = gpu_cpu_queues[j].read_index;
        		// Dequeue completed
        		if (gpu_cpu_read_index != gpu_cpu_queues[j].write_index) {
        			req_t_start[gpu_cpu_queues[j].queue_array[gpu_cpu_read_index].req_id] = get_time_msec();
        			gpu_cpu_queues[j].read_index = (gpu_cpu_read_index + 1) % QUEUENODES;
        			printf("CPU: GPU-CPU read index #%d was updated by thread %d\n", gpu_cpu_read_index, j);
        		}
        	}

            rate_limit_wait(&rate_limit);
            int img_idx = i % N_IMG_PAIRS;
            req_t_start[i] = get_time_msec();

            /* TODO place memcpy's and kernels in a queue */

            // Check for first queue with available node
            for (int j = 0; j < max_simult_blocks; j++) {
            	if (cpu_gpu_queues[j].read_index != (cpu_gpu_queues[j].write_index + 1) % QUEUENODES) {
            		// Enqueue
            		cpu_gpu_queues[j].queue_array[cpu_gpu_queues[j].write_index].req_id = img_idx;
            		cpu_gpu_queues[j].write_index = (cpu_gpu_queues[j].write_index + 1) % QUEUENODES;
        			printf("CPU: CPU-GPU write index #%d was updated by thread %d\n", cpu_gpu_queues[j].write_index, j);
            		// Advance request id
            		i++;
            		break;
            	}
            }
        }
        /* TODO wait until you have responses for all requests */

        // Release memory allocations specific for threads flow
        CUDA_CHECK( cudaFreeHost((void *)gpu_cpu_queues) );
        CUDA_CHECK( cudaFreeHost((void *)cpu_gpu_queues) );
        CUDA_CHECK( cudaFree(gpu_total_distance_f) );
        CUDA_CHECK( cudaFreeHost(cpu_hist_distance_node) );
    } else {
        assert(0);
    }
    double tf = get_time_msec();

    CUDA_CHECK( cudaFree(gpu_image1) );
    CUDA_CHECK( cudaFree(gpu_image2) );
    CUDA_CHECK( cudaFree(gpu_hist1) );
    CUDA_CHECK( cudaFree(gpu_hist2) );

    double avg_latency = 0;
    for (int j = 0; j < NREQUESTS; j++) {
        avg_latency += (req_t_end[j] - req_t_start[j]);
    }
    avg_latency /= NREQUESTS;

    printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", load);
    if (mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", threads_queue_mode);
    printf("average distance between images %f\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);

    return 0;
}
