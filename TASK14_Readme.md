Multithreaded Histogram (CUDA)(C++)
==============================

Description:


What This Program Does:
-----------------------
- It creates 1 million random numbers between 0 and 255.
- Then, it counts how many times each number appears (this is called a histogram).
- It does this in two ways:
   1. One by one (normal way)
   2. **Using 8 threads at the same time** (faster way)

Why Two Ways?
-------------
- The first way (sequential) is simple but slow.
- The second way (multithreaded) is faster because it splits the work into 8 parts.

How It Works:
-------------
1. Data is generated randomly.
2. Histogram is calculated in both ways.
3. Time taken by each method is shown.
4. It checks if both results are the same.


How to Compile:
---------------
Use any modern C++ compiler that supports C++11 or later.



How to Run:(For Task)
-----------
 
1. Go to(https://leetgpu.com/playground) for running the CUDA C++ program , we can use these LEETGPU playground
2. Type code into the editor
3. Click *Run*
    



Sample Output:
--------------
Running Sequential Histogram...
Running Multithreaded Histogram (8 threads)...

Summary:
Sequential Time   : 11 ms
Multithreaded Time: 1 ms
Histograms Match  : YES

