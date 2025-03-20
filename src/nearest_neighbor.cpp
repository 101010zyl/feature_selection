#include "feature_selection/nearest_neighbor.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <omp.h>  // Include OpenMP header

namespace feature_selection {

double NearestNeighbor::calculateDistance(
    const DataPoint& a, 
    const DataPoint& b, 
    const FeatureSet& featureSubset
) {
    double sum = 0.0;
    
    // If using a specific feature subset
    if (!featureSubset.empty()) {
        for (FeatureIndex idx : featureSubset) {
            if (idx < a.size() && idx < b.size()) {
                double diff = a[idx] - b[idx];
                sum += diff * diff;
            }
        }
    } 
    // If using all features
    else {
        for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
    }
    
    return std::sqrt(sum);
}

std::size_t NearestNeighbor::findNearestNeighbor(
    const DataMatrix& data,
    const DataPoint& point,
    std::size_t excludeIndex,
    const FeatureSet& featureSubset
) {
    double minDistance = std::numeric_limits<double>::max();
    std::size_t nearestIndex = 0;
    
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (i == excludeIndex) {
            continue;
        }
        
        double distance = calculateDistance(point, data[i], featureSubset);
        
        if (distance < minDistance) {
            minDistance = distance;
            nearestIndex = i;
        }
    }
    
    return nearestIndex;
}

double NearestNeighbor::leaveOneOutCrossValidation(
    const DataMatrix& data,
    const LabelVector& labels,
    const FeatureSet& featureSubset,
    bool verbose
) {
    if (data.empty() || labels.empty() || data.size() != labels.size()) {
        return 0.0;
    }
    
    std::size_t totalInstances = data.size();
    std::size_t correctPredictions = 0;
    
    // Using OpenMP to parallelize the leave-one-out cross-validation
    #pragma omp parallel reduction(+:correctPredictions)
    {
        // Get thread info for debugging/verbose mode
        int threadId = 0;
        int numThreads = 1;
        
        #ifdef _OPENMP
        threadId = omp_get_thread_num();
        numThreads = omp_get_num_threads();
        
        // Print thread info only from the master thread and only once
        if (threadId == 0 && verbose) {
            #pragma omp single
            {
                std::cout << "Running with " << numThreads << " threads" << std::endl;
            }
        }
        #endif
        
        // Parallelize the loop over all instances
        #pragma omp for schedule(dynamic)
        for (std::size_t i = 0; i < totalInstances; ++i) {
            std::size_t nearestIndex = findNearestNeighbor(data, data[i], i, featureSubset);
            
            // Thread-safe verbose output
            if (verbose) {
                #pragma omp critical
                {
                    std::cout << "Object " << i + 1 << " is class " << labels[i] << std::endl;
                    std::cout << "Its nearest neighbor is " << nearestIndex + 1 
                              << " which is in class " << labels[nearestIndex] << std::endl;
                }
            }
            
            if (labels[i] == labels[nearestIndex]) {
                correctPredictions++;
            }
        }
    }
    
    double accuracy = static_cast<double>(correctPredictions) / static_cast<double>(totalInstances);
    return accuracy;
}

} // namespace feature_selection