CUDA Vector Addition Performance Comparison
===========================================


Description:

What This Program Does:
-----------------------
- Adds two large vectors of floats (arrays) element-wise.
- Compares execution time of:
   1. CPU version (runs on your computer's main processor)
   2. GPU version (runs on NVIDIA GPU using CUDA)

Purpose:
--------
- To measure speed improvement (speedup) using GPU vs CPU.
- Helps understand how CUDA parallelism works in practice.

How It Works:
-------------
1. Initializes two input vectors: A and B.
2. Performs element-wise addition (A[i] + B[i] = C[i]) using:
   - `vectorAddCPU()` – standard C++ loop (runs on CPU).
   - `vectorAddCUDA()` – CUDA kernel (runs on GPU).
3. Measures and compares time taken by both.
4. Verifies that both CPU and GPU results match.
5. Prints:
   - CPU and GPU execution times
   - Whether results match (correctness)
   - Speedup achieved



How to Compile:
---------------
Use any modern C++ compiler that supports C++11 or later.


How to Run:(For Task)
-----------
 
1. Go to(https://leetgpu.com/playground) for running the CUDA C++ program , we can use these LEETGPU playground
2. Type code into the editor
3. Click *Run*
    

It runs tests on multiple vector sizes:
    - 1K (1024 elements)
    - 64K
    - 1M
    - 16M

Example Output:
---------------
CUDA Vector Addition Performance Comparison

----- Vector Size: 1024 -----
 Correctness: PASS
 CPU Time: 0.03 ms
 GPU Time: 0.01 ms
 Speedup (CPU / GPU): 3x

You can use different size of vectors

(Your times may vary depending on your GPU and system.)

Key Concepts Demonstrated:
--------------------------
- CUDA kernel functions
- Memory allocation on device and host
- Data transfer between CPU and GPU
- Synchronization using `cudaDeviceSynchronize()`
- Performance benchmarking using `chrono`


