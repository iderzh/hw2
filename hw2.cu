#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <windows.h>
#include <intrin.h>
#else
/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#endif

#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000
#define NREQUESTS 10000
#define NSTREAMS 64
#define MAXREGCOUNT 32
#define QUEUENODES 10

typedef unsigned char uchar;
#define OUT

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        return 1;                                                                           \
    }                                                                                       \
} while (0)
#else
#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)
#endif

#define SQR(a) ((a) * (a))

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#define __sync_synchronize() _ReadWriteBarrier()
double static inline get_time_msec(void) {
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer)
    {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1000.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }

    if (hasHighResTimer)
    {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else
    {
        return (double)GetTickCount();
    }
}
void usleep(unsigned int usec)
{
    HANDLE timer;
    LARGE_INTEGER ft;

    ft.QuadPart = -(10 * (__int64)usec);

    timer = CreateWaitableTimer(NULL, TRUE, NULL);
    SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
    WaitForSingleObject(timer, INFINITE);
    CloseHandle(timer);
}
int rand_r(unsigned int *pseed) {
    srand(*pseed);
    return rand();
}
#else
double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}
#endif
struct stream_node {
    cudaStream_t Stream;
    int stream_id;
    int req_in_processing;
};
typedef stream_node streamNode;

typedef struct _thread_node {
    volatile int req_id;
    volatile double result;
} thread_node;

typedef struct _threads_queue {
    volatile uchar read_index;
    volatile uchar write_index;
    volatile thread_node queue_array[QUEUENODES];
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

__global__ void process_queues (volatile threads_queue *dev_gpu_cpu_queues, volatile threads_queue *dev_cpu_gpu_queues,
                                const int max_simult_blocks,
                                volatile uchar *gpu_image1,
                                volatile uchar *gpu_image2) {
    __shared__ int          req_id;
    __shared__ bool         was_deq;
    __shared__ unsigned int req_count;
    __shared__ uchar s_image1[IMG_DIMENSION * IMG_DIMENSION], s_image2[IMG_DIMENSION * IMG_DIMENSION];
    __shared__ int s_hist1[256], s_hist2[256];
    __shared__ float s_distance[256];
    __shared__ uchar pattern1[IMG_DIMENSION * IMG_DIMENSION], pattern2[IMG_DIMENSION * IMG_DIMENSION];
    int nrequests = NREQUESTS / max_simult_blocks + !(blockIdx.x >= NREQUESTS % max_simult_blocks);
    int copy_iter = (IMG_DIMENSION * IMG_DIMENSION) / blockDim.x;
    volatile uchar *dcpu_gpu_read_idx  = &dev_cpu_gpu_queues[blockIdx.x].read_index;
    volatile uchar *dcpu_gpu_write_idx = &dev_cpu_gpu_queues[blockIdx.x].write_index;
    volatile uchar *dgpu_cpu_read_idx  = &dev_gpu_cpu_queues[blockIdx.x].read_index;
    volatile uchar *dgpu_cpu_write_idx = &dev_gpu_cpu_queues[blockIdx.x].write_index;

    if (threadIdx.x == 0) {
        was_deq = false; req_count = 0;
    }
    __threadfence_block();

    while (req_count < nrequests) {
        if (threadIdx.x == 0) {
            volatile uchar cpu_gpu_read_idx = *dcpu_gpu_read_idx;
            volatile uchar cpu_gpu_write_idx = *dcpu_gpu_write_idx;
            __threadfence_system();
            // Dequeue request
            if (!was_deq && cpu_gpu_read_idx != cpu_gpu_write_idx) {
                req_id = dev_cpu_gpu_queues[blockIdx.x].queue_array[cpu_gpu_read_idx].req_id;
                *dcpu_gpu_read_idx = (cpu_gpu_read_idx + 1) % QUEUENODES;
                __threadfence_system();
                was_deq = true;
                __threadfence_block();
            }
        }
        __syncthreads();
        // Init data structures for request processing
        if (was_deq) {
            if (threadIdx.x < 256) {
                s_hist1[threadIdx.x] = 0;
                s_hist2[threadIdx.x] = 0;
                s_distance[threadIdx.x] = 0;
            }
            for (int i = 0; i < copy_iter; i++) {
                s_image1[i * blockDim.x + threadIdx.x] = gpu_image1[req_id * IMG_DIMENSION * IMG_DIMENSION + i * blockDim.x + threadIdx.x];
                s_image2[i * blockDim.x + threadIdx.x] = gpu_image2[req_id * IMG_DIMENSION * IMG_DIMENSION + i * blockDim.x + threadIdx.x];
                pattern1[i * blockDim.x + threadIdx.x] = 0;
                pattern2[i * blockDim.x + threadIdx.x] = 0;
            }
        }
        __threadfence_block();

        // Calculate image patterns
        if (was_deq) {
            for (int j = 0; j < copy_iter; j++) {
            	pattern1[j * blockDim.x + threadIdx.x] = local_binary_pattern((uchar *)s_image1, (j * blockDim.x + threadIdx.x) / IMG_DIMENSION, (j * blockDim.x + threadIdx.x) % IMG_DIMENSION);
            }

            for (int j = 0; j < copy_iter; j++) {
            	pattern2[j * blockDim.x + threadIdx.x] = local_binary_pattern((uchar *)s_image2, (j * blockDim.x + threadIdx.x) / IMG_DIMENSION, (j * blockDim.x + threadIdx.x) % IMG_DIMENSION);
            }
        }
        __threadfence_block();

        // Calculate histograms
        if (was_deq) {
        	for (int j = 0; j < copy_iter; j++) {
        		if (pattern1[j * blockDim.x + threadIdx.x] != 0) {
        			atomicAdd((int *)&(s_hist1[pattern1[j * blockDim.x + threadIdx.x]]), 1);
        		}
        		if (pattern2[j * blockDim.x + threadIdx.x] != 0) {
        			atomicAdd((int *)&(s_hist2[pattern2[j * blockDim.x + threadIdx.x]]), 1);
        		}
        		__threadfence_block();
            }
        }
        __threadfence_block();

        if (was_deq) {
            if (threadIdx.x < 256) {
                if (s_hist1[threadIdx.x] + s_hist1[threadIdx.x] != 0) {
                    s_distance[threadIdx.x] = ((float)SQR(s_hist1[threadIdx.x] - s_hist2[threadIdx.x])) / (s_hist1[threadIdx.x] + s_hist2[threadIdx.x]);
                    __threadfence();
                }
                // Comment IF statement above and uncomment s_distance[threadIdx.x] = 1 below
                // It will result average distance 256, I used it for histogram distance validation
                //s_distance[threadIdx.x] = 1;
            }
        }
        __syncthreads();

        if (was_deq) {
            if (threadIdx.x < 256) {
                int length = 256;
                __threadfence_block();
                while (length > 1) {
                    if (threadIdx.x < length / 2) {
                        s_distance[threadIdx.x] = s_distance[threadIdx.x] + s_distance[threadIdx.x + length / 2];
                        __threadfence();
                    }
                    length /= 2;
                    __syncthreads();
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            volatile uchar gpu_cpu_read_idx = *dgpu_cpu_read_idx;
            volatile uchar gpu_cpu_write_idx = *dgpu_cpu_write_idx;
            if (was_deq && (gpu_cpu_read_idx != (gpu_cpu_write_idx + 1) % QUEUENODES)) {
                // Enqueue
                dev_gpu_cpu_queues[blockIdx.x].queue_array[gpu_cpu_write_idx].result = s_distance[0];
                dev_gpu_cpu_queues[blockIdx.x].queue_array[gpu_cpu_write_idx].req_id = req_id;
                __threadfence_system();
                *dgpu_cpu_write_idx = (gpu_cpu_write_idx + 1) % QUEUENODES;
                __threadfence_system();
                req_count++;
                was_deq = false;
            }
        }
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
            //printf("CPU: Req #%d, Result = %lf\n", i, cpu_hist_distance);
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
    int *gpu_hist1 = NULL, *gpu_hist2 = NULL; // TODO: allocate with cudaMalloc
    double *gpu_hist_distance = NULL; //TODO: allocate with cudaMalloc
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
        printf("This device is capable to run %d thread blocks (TB) simultaneously.\n", max_simult_blocks);
        if ( (deviceProp.regsPerBlock / (max_simult_blocks * threads_queue_mode)) < MAXREGCOUNT) {
            max_simult_blocks = deviceProp.regsPerBlock / ( threads_queue_mode * MAXREGCOUNT );
            printf("Amount of running TBs (queue pairs) was reduced to %d due to GPU Registers limitation per TB\n", max_simult_blocks);
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
        }

        CUDA_CHECK( cudaMalloc(&gpu_image1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION) );
        CUDA_CHECK( cudaMalloc(&gpu_image2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION) );

        CUDA_CHECK( cudaMemcpy(gpu_image1, images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(gpu_image2, images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice) );

        volatile threads_queue *dev_cpu_gpu_queues, *dev_gpu_cpu_queues;
        CUDA_CHECK( cudaHostGetDevicePointer ((void **)&dev_gpu_cpu_queues, (void *)gpu_cpu_queues, 0) );
        CUDA_CHECK( cudaHostGetDevicePointer ((void **)&dev_cpu_gpu_queues, (void *)cpu_gpu_queues, 0) );
        CUDA_CHECK( cudaDeviceSynchronize() );

        //Start CUDA kernel
        process_queues<<<max_simult_blocks, threads_queue_mode>>>(  dev_gpu_cpu_queues,
                                                                    dev_cpu_gpu_queues,
                                                                    max_simult_blocks,
                                                                    gpu_image1,
                                                                    gpu_image2);

        for (unsigned int i = 0, k = 0; k < NREQUESTS ;) {

            /* TODO check producer consumer queue for any responses.
               don't block. if no responses are there we'll check again in the next iteration
               update req_t_end of completed requests
               update total_distance */
            for (int j = 0; j < max_simult_blocks; j++) {
                // Get current GPU->CPU queue indexes
                volatile uchar *pgpu_cpu_read_index = &gpu_cpu_queues[j].read_index;
                volatile uchar *pgpu_cpu_write_index = &gpu_cpu_queues[j].write_index;
                volatile uchar read_idx  = *pgpu_cpu_read_index;
                volatile uchar write_idx = *pgpu_cpu_write_index;

                volatile int req_id = gpu_cpu_queues[j].queue_array[read_idx].req_id;
                volatile double result = gpu_cpu_queues[j].queue_array[read_idx].result;

                // Dequeue completed
                if (read_idx != write_idx) {
                    total_distance += result;
                    req_t_end[req_id] = get_time_msec();
                    *pgpu_cpu_read_index = (read_idx + 1) % QUEUENODES;
                    __sync_synchronize();
                    // Advance completed requests counter
                    k++;
                }
            }


            if (i < NREQUESTS) {
                rate_limit_wait(&rate_limit);
                int queue_idx = i % max_simult_blocks;
                req_t_start[i] = get_time_msec();
                volatile uchar read_idx = cpu_gpu_queues[queue_idx].read_index;
                volatile uchar write_idx = cpu_gpu_queues[queue_idx].write_index;
                volatile uchar *pcpu_gpu_write_index = &cpu_gpu_queues[queue_idx].write_index;
                if (read_idx != (write_idx + 1) % QUEUENODES) {
                    // Enqueue
                    cpu_gpu_queues[queue_idx].queue_array[write_idx].req_id = i;
                    *pcpu_gpu_write_index = (write_idx + 1) % QUEUENODES;
                    __sync_synchronize();
                    //printf("CPU: CPU-GPU write index #%d was increased by thread %d\n", write_idx, queue_idx);
                    // Advance request id
                    i++;
                }
            }
        }
        /* TODO wait until you have responses for all requests */
        CUDA_CHECK( cudaDeviceSynchronize() );

        // Release memory allocations specific for threads flow
        CUDA_CHECK( cudaFreeHost((void *)gpu_cpu_queues) );
        CUDA_CHECK( cudaFreeHost((void *)cpu_gpu_queues) );
    } else {
        assert(0);
    }
    double tf = get_time_msec();

    CUDA_CHECK( cudaFree(gpu_image1) );
    CUDA_CHECK( cudaFree(gpu_image2) );
    if (gpu_hist1 != NULL) CUDA_CHECK( cudaFree(gpu_hist1) );
    if (gpu_hist2 != NULL) CUDA_CHECK( cudaFree(gpu_hist2) );
    if (gpu_hist_distance != NULL) CUDA_CHECK( cudaFree(gpu_hist_distance) );

    double avg_latency = 0;
    for (int j = 0; j < NREQUESTS; j++) {
        avg_latency += (req_t_end[j] - req_t_start[j]);
    }
    avg_latency /= NREQUESTS;

    printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", load);
    if (mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", threads_queue_mode);
    printf("average distance between images %lf\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);

    return 0;
}
