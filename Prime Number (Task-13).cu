#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

const int MAX = 1000000;  // Check primes from 2 to MAX (1 million)

// Utility function to check if a number is prime
bool isPrime(int n) {
    if (n <= 1) return false;             // 0 and 1 are not prime
    if (n == 2) return true;              // 2 is the only even prime
    if (n % 2 == 0) return false;         // eliminate even numbers
    int sqrtN = static_cast<int>(std::sqrt(n));
    for (int i = 3; i <= sqrtN; i += 2) { // check odd divisors only
        if (n % i == 0) return false;
    }
    return true;
}

// Sequential version: find all primes from 2 to MAX
void findPrimesSequential(std::vector<int>& primes) {
    for (int i = 2; i <= MAX; ++i) {
        if (isPrime(i)) {
            primes.push_back(i); // Add prime to list
        }
    }
}

// Threaded function: find primes in a sub-range 
void findPrimesInRange(int start, int end, std::vector<int>& localPrimes) {
    for (int i = start; i <= end; ++i) {
        if (isPrime(i)) {
            localPrimes.push_back(i);
        }
    }
}

// Multithreaded version: divide work among threads
void findPrimesParallel(std::vector<int>& primes, int numThreads) {
    std::vector<std::thread> threads;
    std::vector<std::vector<int>> threadPrimes(numThreads); // one prime list per thread

    int chunkSize = MAX / numThreads; // divide data into chunks

    // Launch threads
    for (int t = 0; t < numThreads; ++t) {
        int start = t * chunkSize + 1;
        int end = (t == numThreads - 1) ? MAX : (t + 1) * chunkSize;
        threads.emplace_back(findPrimesInRange, start, end, std::ref(threadPrimes[t]));
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Merge results from each thread into final prime list
    for (auto& tp : threadPrimes) {
        primes.insert(primes.end(), tp.begin(), tp.end());
    }
}

// Utility to time function execution
template <typename Func>
long long benchmark(Func f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();  // Run the function
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

// Main function
int main() {
    std::vector<int> primesSequential, primesParallel;

    // Run sequential version
    std::cout << "Running sequential prime finder..." << std::endl;
    long long timeSeq = benchmark([&]() {
        findPrimesSequential(primesSequential);
    });

    std::cout << "Found " << primesSequential.size()
              << " primes sequentially in " << timeSeq << " ms.\n";

    // Run parallel version using number of CPU cores
    std::cout << "\nRunning multithreaded prime finder..." << std::endl;
    int numThreads = std::thread::hardware_concurrency(); // e.g., 8 on a typical CPU
    long long timePar = benchmark([&]() {
        findPrimesParallel(primesParallel, numThreads);
    });

    std::cout << "Found " << primesParallel.size()
              << " primes in parallel using " << numThreads
              << " threads in " << timePar << " ms.\n";

    return 0;
}
