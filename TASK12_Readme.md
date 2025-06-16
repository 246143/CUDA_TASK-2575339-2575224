# CUDA Matrix Transpose

This project demonstrates how to transpose a square matrix using NVIDIA's CUDA for GPU acceleration, and compares the result with a CPU-based transpose for correctness.

 Description:

- Allocates and fills an `N x N` matrix with float values.
- Performs matrix transpose using:
  - A CUDA kernelfor GPU parallel computation.
  - A CPU function for reference and validation.
- Compares the results of GPU and CPU transpose.
- Prints both the original and transposed matrices.

 Matrix Size

Matrix dimension is defined as:

we can keep any size based our interest for the matrix size
#define N 8


You can change this value in the source code to test with different matrix sizes.



How to Compile:
---------------
Use any modern C++ compiler that supports C++11 or later.



How to Run:
-----------
 
1. Go to(https://leetgpu.com/playground) for running the CUDA C++ program , we can use these LEETGPU playground
2. Type code into the editor
3. Click *Run*

```


## Output:

The program outputs:

- A message indicating whether the GPU transpose matches the CPU result.
- The original matrix.
- The GPU transposed matrix.

Example:

```
Transpose successful and correct!


Original Matrix:
0 1 2 3 4 5 6 7 
8 9 10 11 12 13 14 15 
16 17 18 19 20 21 22 23 
24 25 26 27 28 29 30 31 
32 33 34 35 36 37 38 39 
40 41 42 43 44 45 46 47 
48 49 50 51 52 53 54 55 
56 57 58 59 60 61 62 63 

Transposed Matrix (GPU):
0 8 16 24 32 40 48 56 
1 9 17 25 33 41 49 57 
2 10 18 26 34 42 50 58 
3 11 19 27 35 43 51 59 
4 12 20 28 36 44 52 60 
5 13 21 29 37 45 53 61 
6 14 22 30 38 46 54 62 
7 15 23 31 39 47 55 63 

```

##  Validation:

The `validate()` function checks the CPU and GPU transpose results with a tolerance of `1e-5` to handle floating-point precision.

##  Key Concepts Used:

- CUDA kernel launch with `dim3` grid and block dimensions
- Host ↔ Device memory transfer
- Parallel execution on the GPU
- Matrix indexing in 1D memory layout
- CPU-GPU result comparison

