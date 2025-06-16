#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>

const int NUM_BINS = 256;         // Number of bins in the histogram (0 to 255)
const int DATA_SIZE = 1'000'000;  // Total number of random values to generate
const int NUM_THREADS = 8;        // Number of threads to use in multithreading

// Function to generate random data values between 0 and 255
void generateData(std::vector<int>& data) {
    std::mt19937 gen(42);  // Random number generator with fixed seed for reproducibility
    std::uniform_int_distribution<> dis(0, 255);  // Distribution range

    for (int& val : data) {
        val = dis(gen);  // Fill the data array with random numbers
    }
}

// Compute histogram sequentially (single-threaded)
void computeSequential(const std::vector<int>& data, std::vector<int>& histogram) {
    for (int val : data) {
        histogram[val]++;  // Increment the count for each data value
    }
}

// Function to compute part of histogram in a thread
void computeChunk(const std::vector<int>& data, std::vector<int>& localHist, int start, int end) {
    for (int i = start; i < end; ++i) {
        localHist[data[i]]++;  // Increment local histogram bin for data[i]
    }
}

// Compute histogram using multithreading
void computeMultithreaded(const std::vector<int>& data, std::vector<int>& histogram) {
    std::vector<std::thread> threads;  // To store threads
    std::vector<std::vector<int>> threadHist(NUM_THREADS, std::vector<int>(NUM_BINS, 0));  // One local histogram per thread

    int chunkSize = data.size() / NUM_THREADS;  // Divide data among threads

    // Launch threads to compute their chunks
    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * chunkSize;  // Start index for this thread
        int end = (i == NUM_THREADS - 1) ? data.size() : start + chunkSize;  // End index (last thread takes remainder)
        threads.emplace_back(computeChunk, std::ref(data), std::ref(threadHist[i]), start, end);
    }

    // Wait for all threads to finish
    for (auto& t : threads) t.join();

    // Merge all thread-local histograms into final histogram
    for (int bin = 0; bin < NUM_BINS; ++bin) {
        for (int t = 0; t < NUM_THREADS; ++t) {
            histogram[bin] += threadHist[t][bin];
        }
    }
}

// Measure execution time of any function passed to it
template <typename Func>
long long measureTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();  // Start timer
    func();  // Run the function
    auto end = std::chrono::high_resolution_clock::now();    // End timer
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();  // Return elapsed time in ms
}

int main() {
    std::vector<int> data(DATA_SIZE);         // Data array of 1 million integers
    generateData(data);                       // Fill with random values

    std::vector<int> histSeq(NUM_BINS, 0);    // Histogram for sequential computation
    std::vector<int> histPar(NUM_BINS, 0);    // Histogram for parallel computation

    std::cout << "Running Sequential Histogram...\n";
    auto timeSeq = measureTime([&]() {
        computeSequential(data, histSeq);     // Measure time taken by sequential method
    });

    std::cout << "Running Multithreaded Histogram (8 threads)...\n";
    auto timePar = measureTime([&]() {
        computeMultithreaded(data, histPar);  // Measure time taken by parallel method
    });

    // Compare both histograms to check if they are the same
    bool same = (histSeq == histPar);

    // Display summary
    std::cout << "\nSummary:\n";
    std::cout << "Sequential Time   : " << timeSeq << " ms\n";
    std::cout << "Multithreaded Time: " << timePar << " ms\n";
    std::cout << "Histograms Match  : " << (same ? "YES " : "NO ") << "\n";

    return 0;
}
