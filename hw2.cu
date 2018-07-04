/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>

#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000
#define NREQUESTS 100000
#define NSTREAMS 64

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

    CUDA_CHECK(cudaMalloc(&gpu_image1, IMG_DIMENSION * IMG_DIMENSION * NSTREAMS));
    CUDA_CHECK(cudaMalloc(&gpu_image2, IMG_DIMENSION * IMG_DIMENSION * NSTREAMS));
    CUDA_CHECK(cudaMalloc(&gpu_hist1, 256 * sizeof(int) * NSTREAMS));
    CUDA_CHECK(cudaMalloc(&gpu_hist2, 256 * sizeof(int) * NSTREAMS));
    CUDA_CHECK(cudaMalloc(&gpu_hist_distance, 256 * sizeof(double) * NSTREAMS));
    CUDA_CHECK(cudaHostAlloc(&cpu_hist_distance, sizeof(double), 0));

    double ti = get_time_msec();
    if (mode == PROGRAM_MODE_STREAMS) {

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

    } else if (mode == PROGRAM_MODE_QUEUE) {
        for (int i = 0; i < NREQUESTS; i++) {

            /* TODO check producer consumer queue for any responses.
               don't block. if no responses are there we'll check again in the next iteration
               update req_t_end of completed requests
               update total_distance */

            rate_limit_wait(&rate_limit);
            int img_idx = i % N_IMG_PAIRS;
            req_t_start[i] = get_time_msec();

            /* TODO place memcpy's and kernels in a stream */
        }
        /* TODO wait until you have responses for all requests */
    } else {
        assert(0);
    }
    double tf = get_time_msec();

    CUDA_CHECK( cudaFree(gpu_image1) );
    CUDA_CHECK( cudaFree(gpu_image2) );
    CUDA_CHECK( cudaFree(gpu_hist1) );
    CUDA_CHECK( cudaFree(gpu_hist2) );
    CUDA_CHECK( cudaFree(gpu_hist_distance) );
    CUDA_CHECK( cudaFreeHost(cpu_hist_distance) );

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
