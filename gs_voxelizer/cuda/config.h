#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 8
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y * BLOCK_Z)
#define N_THREADS 128

#define MAX_CHANNELS 4

#define CUDA_CALL(x)                                                           \
    do {                                                                       \
        if ((x) != cudaSuccess) {                                              \
            printf(                                                            \
                "Error at %s:%d - %s\n",                                       \
                __FILE__,                                                      \
                __LINE__,                                                      \
                cudaGetErrorString(cudaGetLastError())                         \
            );                                                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
