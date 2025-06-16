Parallel Prime Number Finder  CUDA (C++ Program)
===========================================


Description:

What This Program Does:
-----------------------
- Finds all prime numbers from 1 to 1,000,000.
- Uses two methods:
   1. Sequential (normal, one thread)
   2. Multithreaded (runs using multiple CPU cores)

Why Two Methods?
----------------
- To compare the performance of normal vs parallel processing.
- Parallel version is faster by splitting the task into smaller parts.

How It Works:
-------------
1. The `isPrime()` function checks if a number is prime.
2. The program uses two main functions:
   - `findPrimesSequential()` – runs in a single thread.
   - `findPrimesParallel()` – uses multiple threads to check primes.
3. The work is divided evenly among available CPU threads.
4. Both methods are timed, and the total number of primes is shown.

How to Compile:
---------------
Use any modern C++ compiler that supports C++11 or later.



How to Run:
-----------
 
1. Go to(https://leetgpu.com/playground) for running the CUDA C++ program , we can use these LEETGPU playground
2. Type code into the editor
3. Click *Run*
    


Example Output:
---------------
Running sequential prime finder...
Found 78498 primes sequentially in 900 ms.

Running multithreaded prime finder...
Found 78498 primes in parallel using 8 threads in 270 ms.

(Your actual times may vary depending on your CPU.)

Key Concepts Used:
------------------
- Prime number checking logic
- `std::thread` for multithreading
- `std::chrono` for measuring time
- `std::vector` for storing results
- `std::thread::hardware_concurrency()` to use system's CPU cores
