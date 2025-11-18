# Exp5 Bubble Sort and Merge sort in CUDA
#### NAME : VINISHRAJ R
#### REG NO : 212223230243
#### DATE : 
**Objective:**
Implement Bubble Sort and Merge Sort on the GPU using CUDA, analyze the efficiency of this sorting algorithm when parallelized, and explore the limitations of Bubble Sort and Merge Sort for large datasets.
## AIM:
Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

Code Overview:
You will work with the provided CUDA implementation of Bubble Sort and Merge Sort. The code initializes an unsorted array, applies the Bubble Sort, Merge Sort algorithm in parallel on the GPU, and returns the sorted array as output.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC, Google Colab with NVCC Compiler, CUDA Toolkit installed, and sample datasets for testing.

## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
```c++
%%writefile sorting.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>

#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(err));                                     \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

__global__ void bubbleSortKernel(int *d_arr, int n, int phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * tid + (phase & 1);
    if (i + 1 < n) {
        int a = d_arr[i];
        int b = d_arr[i + 1];
        if (a > b) {
            d_arr[i] = b;
            d_arr[i + 1] = a;
        }
    }
}

__global__ void mergeKernel(int *d_in, int *d_out, int n, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * (2 * width);
    if (start >= n) return;
    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    int i = start;
    int j = mid;
    int k = start;

    while (i < mid && j < end) {
        if (d_in[i] <= d_in[j]) {
            d_out[k++] = d_in[i++];
        } else {
            d_out[k++] = d_in[j++];
        }
    }
    while (i < mid) d_out[k++] = d_in[i++];
    while (j < end) d_out[k++] = d_in[j++];
}

void bubbleSort(int *arr, int n) {
    int *d_arr;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int threads = 256;
    int pairs = (n + 1) / 2;
    int blocks = (pairs + threads - 1) / threads;
    for (int phase = 0; phase < n; ++phase) {
        bubbleSortKernel<<<blocks, threads>>>(d_arr, n, phase);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_arr));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("Bubble Sort (GPU) took %f milliseconds\n", milliseconds);
}

void mergeSort(int *arr, int n) {
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, arr, n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int threads = 128;
    for (int width = 1; width < n; width *= 2) {
        int numSegments = (n + (2 * width) - 1) / (2 * width);
        int blocks = (numSegments + threads - 1) / threads;
        mergeKernel<<<blocks, threads>>>(d_in, d_out, n, width);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(d_in, d_out);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(arr, d_in, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("Merge Sort (GPU) took %f milliseconds\n", milliseconds);
}

void bubbleSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("Bubble Sort (CPU) took %f milliseconds\n", duration.count());
}

void mergeHost(int *arr, int left, int mid, int right) {
    int n1 = mid - left;
    int n2 = right - mid;
    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));
    for (int i = 0; i < n1; ++i) L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j) R[j] = arr[mid + j];
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    free(L); free(R);
}

void mergeSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int size = 1; size < n; size *= 2) {
        int left = 0;
        while (left + size < n) {
            int mid = left + size;
            int right = std::min(left + 2 * size, n);
            mergeHost(arr, left, mid, right);
            left += 2 * size;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("Merge Sort (CPU) took %f milliseconds\n", duration.count());
}

void printArray(int *arr, int n) {
    int limit = std::min(n, 100);
    for (int i = 0; i < limit; i++) printf("%d ", arr[i]);
    if (n > limit) printf("... (total %d elements)\n", n);
    else printf("\n");
}

int main() {
    int n = 1000;
    int *arr = (int*)malloc(n * sizeof(int));
    srand(123);

    for (int i = 0; i < n; i++) arr[i] = rand() % 1000;

    printf("Original array (first elements):\n");
    printArray(arr, n);

    int *temp = (int*)malloc(n * sizeof(int));
    memcpy(temp, arr, n * sizeof(int));
    bubbleSortCPU(temp, n);
    printf("Sorted array using Bubble Sort (CPU) (first elements):\n");
    printArray(temp, n);
    memcpy(temp, arr, n * sizeof(int));

    bubbleSort(temp, n);
    printf("Sorted array using Bubble Sort (GPU) (first elements):\n");
    printArray(temp, n);
    memcpy(temp, arr, n * sizeof(int));

    printf("Original array for Merge Sort (first elements):\n");
    printArray(arr, n);

    memcpy(temp, arr, n * sizeof(int));
    mergeSortCPU(temp, n);
    printf("Sorted array using Merge Sort (CPU) (first elements):\n");
    printArray(temp, n);
    memcpy(temp, arr, n * sizeof(int));

    mergeSort(temp, n);
    printf("Sorted array using Merge Sort (GPU) (first elements):\n");
    printArray(temp, n);

    free(arr);
    free(temp);
    return 0;
}
```



## OUTPUT:
<img width="1780" height="497" alt="image" src="https://github.com/user-attachments/assets/767f75e6-eb63-4c0e-a633-b0d8af4da125" />


## RESULT:
Thus, the program has been executed using CUDA to implement Bubble Sort and Merge Sort on the GPU using CUDA and analyze the efficiency of this sorting algorithm when parallelized.
