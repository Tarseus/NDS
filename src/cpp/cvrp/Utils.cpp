/*
 * CVRP Utilities Implementation
 * 
 * Implementation of utility functions for random number generation and array operations.
 */

#include "Utils.h"

namespace {
std::mt19937 randomGenerator(std::random_device{}());
std::vector<float> fastRandomValues;
size_t fastRandomIndex = 0;
const size_t fastRandomPoolSize = 10000;
}

std::mt19937& getRandomGenerator() {
    return randomGenerator;
}

void setRandomSeed(unsigned int seed) {
    randomGenerator.seed(seed);
    fastRandomValues.clear();
    fastRandomIndex = 0;
}

// Generate a random integer in the range [min, max] (inclusive)
int getRandomNumber(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(randomGenerator);
}

// Generate a random float in the range [min, max]
float getRandomFraction(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(randomGenerator);
}

// Generate a random float between 0.0 and 1.0 using pre-generated values for speed
float getRandomFractionFast() {
    // Initialize the random value pool if empty
    if (fastRandomValues.empty()) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        // Pre-generate random values
        fastRandomValues.reserve(fastRandomPoolSize);
        for (size_t i = 0; i < fastRandomPoolSize; ++i) {
            fastRandomValues.push_back(dist(randomGenerator));
        }
    }

    // Get the next random value from the pool
    float randomValue = fastRandomValues[fastRandomIndex];
    fastRandomIndex = (fastRandomIndex + 1) % fastRandomPoolSize;

    return randomValue;
}

// Perform argsort on a vector of float values - returns indices sorted by values
std::vector<int> argsort(const std::vector<float>& values) {
    // Create an index vector [0, 1, 2, ..., n-1]
    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort the indices based on the corresponding values
    std::sort(indices.begin(), indices.end(), [&values](size_t i1, size_t i2) {
        return values[i1] < values[i2];
    });

    return indices;
}
