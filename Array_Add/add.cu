#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for adding two arrays
__global__ void addArraysGPU(int* a, int* b, int* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU function for adding two arrays
void addArraysCPU(int* a, int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 20; // Array size (1 million elements)
    const int size = N * sizeof(int);

    // Host arrays
    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_c_cpu = new int[N];
    int *h_c_gpu = new int[N];

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // CPU Timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    addArraysCPU(h_a, h_b, h_c_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // GPU Timing
    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    cudaEventRecord(start_gpu);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addArraysGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaEventRecord(end_gpu);

    cudaEventSynchronize(end_gpu);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, end_gpu);

    // Copy result back to host
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    // Validate GPU results
    bool isValid = true;
    for (int i = 0; i < N; i++) {
        if (h_c_cpu[i] != h_c_gpu[i]) {
            isValid = false;
            break;
        }
    }

    // Print results
    std::cout << "CPU Time: " << cpu_time << " ms\n";
    std::cout << "GPU Time: " << gpu_time << " ms\n";
    std::cout << "Time Difference: " << cpu_time - gpu_time << " ms\n";
    std::cout << "Results Valid: " << (isValid ? "Yes" : "No") << "\n";

    // Print some of the results to verify correctness
    std::cout << "\nFirst 10 results (CPU vs GPU):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "CPU: " << h_c_cpu[i] << " | GPU: " << h_c_gpu[i] << std::endl;
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;

    return 0;
}
