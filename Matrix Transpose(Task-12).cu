#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 8 // Matrix size: N x N

// CUDA Kernel: Transpose input matrix to output matrix
__global__ void matrixTranspose(float* output, float* input, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Compute global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Compute global column index

    // Only transpose if within matrix bounds
    if (row < width && col < width) {
        output[col * width + row] = input[row * width + col]; // Transpose logic
    }
}

// CPU version of matrix transpose (for verification)
void cpuTranspose(float* output, float* input, int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            output[col * width + row] = input[row * width + col]; // Same transpose logic
        }
    }
}

// Validate: Check if two matrices are equal (element-wise)
bool validate(float* a, float* b, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > 1e-5) // Allowing small floating-point difference
            return false;
    }
    return true;
}

int main() {
    size_t size = N * N * sizeof(float); // Total memory needed (N x N floats)

    // Allocate host memory for input and outputs
    float *h_input = (float*)malloc(size);         // Input matrix
    float *h_output_cpu = (float*)malloc(size);    // Transposed by CPU
    float *h_output_gpu = (float*)malloc(size);    // Transposed by GPU

    // Initialize input matrix with sequential values: 0, 1, 2, ..., N*N-1
    for (int i = 0; i < N * N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate memory on GPU (device)
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input matrix from CPU (host) to GPU (device)
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(4, 4); // 
    dim3 gridSize((N + 1) / 2, (N + 1) / 2); // Enough blocks to cover N x N matrix

    // Launch the CUDA kernel
    matrixTranspose<<<gridSize, blockSize>>>(d_output, d_input, N);
    cudaDeviceSynchronize(); // Wait until kernel execution finishes

    // Copy transposed result from device to host
    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);

    // Do the same transpose on CPU
    cpuTranspose(h_output_cpu, h_input, N);

    // Compare both outputs (CPU and GPU)
    if (validate(h_output_cpu, h_output_gpu, N * N))
        std::cout << " Transpose successful and correct!" << std::endl;
    else
        std::cout << " Transpose mismatch!" << std::endl;

    // Print original matrix
    std::cout << "\nOriginal Matrix:\n";
    for (int i = 0; i < N * N; ++i) {
        std::cout << h_input[i] << " ";
        if ((i + 1) % N == 0) std::cout << "\n";
    }

    // Print GPU-transposed matrix
    std::cout << "\nTransposed Matrix (GPU):\n";
    for (int i = 0; i < N * N; ++i) {
        std::cout << h_output_gpu[i] << " ";
        if ((i + 1) % N == 0) std::cout << "\n";
    }

    // Free all allocated memory (host and device)
    free(h_input); free(h_output_cpu); free(h_output_gpu);
    cudaFree(d_input); cudaFree(d_output);

    return 0;
}
