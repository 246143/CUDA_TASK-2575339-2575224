#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel to perform vector addition on GPU
__global__ void vectorAddCUDA(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread index
    if (i < n) {
        C[i] = A[i] + B[i]; // Perform element-wise addition
    }
}

// CPU function for vector addition
void vectorAddCPU(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i]; // Sequential addition
    }
}

// Function to compare CPU vs GPU performance for vector addition
void runTest(int N) {
    cout << "\n Vector Size: " << N << " \n";

    // Allocate memory for host (CPU) vectors
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C_CPU = new float[N]; // Result from CPU
    float *h_C_GPU = new float[N]; // Result from GPU

    // Initialize input vectors with test values
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = (N - i) * 1.0f;
    }

    // CPU Benchmark 
    auto startCPU = chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C_CPU, N); // Run CPU version
    auto endCPU = chrono::high_resolution_clock::now();
    float timeCPU = chrono::duration<float, milli>(endCPU - startCPU).count();

    // Allocate Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float)); // Allocate GPU memory for vector A
    cudaMalloc(&d_B, N * sizeof(float)); // Allocate GPU memory for vector B
    cudaMalloc(&d_C, N * sizeof(float)); // Allocate GPU memory for result vector C

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    //  GPU Benchmark
    int threadsPerBlock = 256; // Standard block size
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks

    auto startGPU = chrono::high_resolution_clock::now();
    vectorAddCUDA<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N); // Launch kernel
    cudaDeviceSynchronize(); // Wait for GPU to finish
    auto endGPU = chrono::high_resolution_clock::now();
    float timeGPU = chrono::duration<float, milli>(endGPU - startGPU).count();

    // Copy the result back to host
    cudaMemcpy(h_C_GPU, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    //  Result Verification 
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C_CPU[i] - h_C_GPU[i]) > 1e-5) { // Compare each element
            correct = false;
            break;
        }
    }

    //Report Performance
    cout << " Correctness: " << (correct ? "PASS" : "FAIL") << endl;
    cout << " CPU Time: " << timeCPU << " ms" << endl;
    cout << " GPU Time: " << timeGPU << " ms" << endl;
    cout << " Speedup (CPU / GPU): " << (timeCPU / timeGPU) << "x" << endl;

    // Cleanup Memory 
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_CPU;
    delete[] h_C_GPU;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Main function to test vector addition with different sizes
int main() {
    cout << "CUDA Vector Addition Performance Comparison\n";

    runTest(1 << 10);  // Test with 1K elements
    runTest(1 << 16);  // Test with 64K elements
    runTest(1 << 20);  // Test with 1M elements
    runTest(1 << 24);  // Test with 16M elements

    return 0;
}
