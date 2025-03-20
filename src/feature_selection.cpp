#include "feature_selection/feature_selection.h"
#include "feature_selection/nearest_neighbor.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <mutex>
#include <omp.h>  // Include OpenMP header

namespace feature_selection {

SearchResult FeatureSelection::forwardSelection(
    const DataMatrix& data,
    const LabelVector& labels,
    bool verbose
) {
    Timer timer("Forward Selection");
    
    SearchResult result;
    result.bestAccuracy = 0.0;
    
    std::size_t numFeatures = 0;
    if (!data.empty()) {
        numFeatures = data[0].size();
    }
    
    if (verbose) {
        std::cout << "Beginning Forward Selection search." << std::endl;
        
        // Print OpenMP information if available
        #ifdef _OPENMP
        std::cout << "Using OpenMP version " << _OPENMP 
                  << " with a maximum of " << omp_get_max_threads() 
                  << " threads." << std::endl;
        #else
        std::cout << "OpenMP is not enabled." << std::endl;
        #endif
    }
    
    // Start with empty feature set
    FeatureSet currentSet;
    
    // First evaluate with no features (default rate)
    double baselineAccuracy = NearestNeighbor::leaveOneOutCrossValidation(
        data, labels, currentSet, false
    );
    
    if (verbose) {
        std::cout << "Using feature(s) " << featureSetToString(currentSet) 
                  << " accuracy is " << std::fixed << std::setprecision(1) 
                  << (baselineAccuracy * 100.0) << "%" << std::endl;
    }
    
    result.allResults.push_back({currentSet, baselineAccuracy});
    result.bestFeatureSet = currentSet;
    result.bestAccuracy = baselineAccuracy;
    
    // At each level, add the feature that gives the best accuracy
    for (std::size_t i = 0; i < numFeatures; ++i) {
        FeatureIndex bestFeatureToAdd = 0;
        double bestNewAccuracy = 0.0;
        bool foundBetter = false;
        
        // Store results from each candidate evaluation
        struct CandidateResult {
            FeatureIndex feature;
            double accuracy;
            FeatureSet featureSet;
        };
        
        std::vector<CandidateResult> candidateResults;
        candidateResults.reserve(numFeatures - currentSet.size());
        
        // For thread safety during results collection
        std::mutex resultsMutex;
        
        // Process features in parallel
        #pragma omp parallel for schedule(dynamic) if(numFeatures > 8)
        for (FeatureIndex featureToAdd = 0; featureToAdd < numFeatures; ++featureToAdd) {
            // Skip if this feature is already in the set
            if (currentSet.find(featureToAdd) != currentSet.end()) {
                continue;
            }
            
            // Create a candidate set with the new feature
            FeatureSet candidateSet = currentSet;
            candidateSet.insert(featureToAdd);
            
            // Evaluate the candidate set
            double accuracy = NearestNeighbor::leaveOneOutCrossValidation(
                data, labels, candidateSet, false
            );
            
            // Store result (thread-safe)
            {
                std::lock_guard<std::mutex> lock(resultsMutex);
                candidateResults.push_back({featureToAdd, accuracy, candidateSet});
            }
            
            // Verbose output (thread-safe)
            if (verbose) {
                #pragma omp critical
                {
                    std::cout << "Using feature(s) " << featureSetToString(candidateSet) 
                              << " accuracy is " << std::fixed << std::setprecision(1) 
                              << (accuracy * 100.0) << "%" << std::endl;
                }
            }
        }
        
        // Find the best candidate
        for (const auto& candidate : candidateResults) {
            if (!foundBetter || candidate.accuracy > bestNewAccuracy) {
                bestNewAccuracy = candidate.accuracy;
                bestFeatureToAdd = candidate.feature;
                foundBetter = true;
            }
        }
        
        // If we couldn't find a better feature, break
        if (!foundBetter) {
            break;
        }
        
        // Add the best feature to our current set
        currentSet.insert(bestFeatureToAdd);
        
        if (verbose) {
            std::cout << "Feature set " << featureSetToString(currentSet) 
                      << " was best, accuracy is " << std::fixed << std::setprecision(1) 
                      << (bestNewAccuracy * 100.0) << "%" << std::endl;
        }
        
        // Record result
        result.allResults.push_back({currentSet, bestNewAccuracy});
        
        // Update overall best result if applicable
        if (bestNewAccuracy > result.bestAccuracy) {
            result.bestAccuracy = bestNewAccuracy;
            result.bestFeatureSet = currentSet;
        }
    }
    
    if (verbose) {
        std::cout << "Finished search!! The best feature subset is " 
                  << featureSetToString(result.bestFeatureSet) 
                  << ", which has an accuracy of " << std::fixed << std::setprecision(1) 
                  << (result.bestAccuracy * 100.0) << "%" << std::endl;
    }
    
    return result;
}

SearchResult FeatureSelection::backwardElimination(
    const DataMatrix& data,
    const LabelVector& labels,
    bool verbose
) {
    Timer timer("Backward Elimination");
    
    SearchResult result;
    result.bestAccuracy = 0.0;
    
    std::size_t numFeatures = 0;
    if (!data.empty()) {
        numFeatures = data[0].size();
    }
    
    if (verbose) {
        std::cout << "Beginning Backward Elimination search." << std::endl;
        
        // Print OpenMP information if available
        #ifdef _OPENMP
        std::cout << "Using OpenMP version " << _OPENMP 
                  << " with a maximum of " << omp_get_max_threads() 
                  << " threads." << std::endl;
        #else
        std::cout << "OpenMP is not enabled." << std::endl;
        #endif
    }
    
    // Start with all features
    FeatureSet currentSet;
    for (FeatureIndex i = 0; i < numFeatures; ++i) {
        currentSet.insert(i);
    }
    
    // First evaluate with all features
    double baselineAccuracy = NearestNeighbor::leaveOneOutCrossValidation(
        data, labels, currentSet, false
    );
    
    if (verbose) {
        std::cout << "Using feature(s) " << featureSetToString(currentSet) 
                  << " accuracy is " << std::fixed << std::setprecision(1) 
                  << (baselineAccuracy * 100.0) << "%" << std::endl;
    }
    
    result.allResults.push_back({currentSet, baselineAccuracy});
    result.bestFeatureSet = currentSet;
    result.bestAccuracy = baselineAccuracy;
    
    // Store all features in a vector for parallel processing
    std::vector<FeatureIndex> allFeatures(currentSet.begin(), currentSet.end());
    
    // At each level, remove the feature that gives the least reduction in accuracy
    for (std::size_t i = 0; i < numFeatures && allFeatures.size() > 1; ++i) {
        // Store results from each candidate evaluation
        struct CandidateResult {
            FeatureIndex feature;
            double accuracy;
            FeatureSet featureSet;
        };
        
        std::vector<CandidateResult> candidateResults;
        candidateResults.reserve(allFeatures.size());
        
        // For thread safety during results collection
        std::mutex resultsMutex;
        
        // Process features in parallel
        #pragma omp parallel for schedule(dynamic) if(allFeatures.size() > 8)
        for (std::size_t j = 0; j < allFeatures.size(); ++j) {
            FeatureIndex featureToRemove = allFeatures[j];
            
            // Create a candidate set without the feature
            FeatureSet candidateSet = currentSet;
            candidateSet.erase(featureToRemove);
            
            // Skip if we'd be removing the last feature
            if (candidateSet.empty()) {
                continue;
            }
            
            // Evaluate the candidate set
            double accuracy = NearestNeighbor::leaveOneOutCrossValidation(
                data, labels, candidateSet, false
            );
            
            // Store result (thread-safe)
            {
                std::lock_guard<std::mutex> lock(resultsMutex);
                candidateResults.push_back({featureToRemove, accuracy, candidateSet});
            }
            
            // Verbose output (thread-safe)
            if (verbose) {
                #pragma omp critical
                {
                    std::cout << "Using feature(s) " << featureSetToString(candidateSet) 
                              << " accuracy is " << std::fixed << std::setprecision(1) 
                              << (accuracy * 100.0) << "%" << std::endl;
                }
            }
        }
        
        // If no candidates were evaluated, break
        if (candidateResults.empty()) {
            break;
        }
        
        // Find the best candidate
        FeatureIndex bestFeatureToRemove = candidateResults[0].feature;
        double bestNewAccuracy = candidateResults[0].accuracy;
        
        for (std::size_t j = 1; j < candidateResults.size(); ++j) {
            if (candidateResults[j].accuracy > bestNewAccuracy) {
                bestNewAccuracy = candidateResults[j].accuracy;
                bestFeatureToRemove = candidateResults[j].feature;
            }
        }
        
        // Remove the best feature from our current set
        currentSet.erase(bestFeatureToRemove);
        
        // Update the allFeatures vector
        allFeatures.erase(
            std::remove(allFeatures.begin(), allFeatures.end(), bestFeatureToRemove),
            allFeatures.end()
        );
        
        if (verbose) {
            std::cout << "Feature set " << featureSetToString(currentSet) 
                      << " was best, accuracy is " << std::fixed << std::setprecision(1) 
                      << (bestNewAccuracy * 100.0) << "%" << std::endl;
        }
        
        // Record result
        result.allResults.push_back({currentSet, bestNewAccuracy});
        
        // Update overall best result if applicable
        if (bestNewAccuracy > result.bestAccuracy) {
            result.bestAccuracy = bestNewAccuracy;
            result.bestFeatureSet = currentSet;
        }
    }
    
    // Also consider the empty set
    if (!allFeatures.empty()) {
        FeatureSet emptySet;
        double emptySetAccuracy = NearestNeighbor::leaveOneOutCrossValidation(
            data, labels, emptySet, false
        );
        
        if (verbose) {
            std::cout << "Using feature(s) " << featureSetToString(emptySet) 
                      << " accuracy is " << std::fixed << std::setprecision(1) 
                      << (emptySetAccuracy * 100.0) << "%" << std::endl;
        }
        
        result.allResults.push_back({emptySet, emptySetAccuracy});
        
        if (emptySetAccuracy > result.bestAccuracy) {
            result.bestAccuracy = emptySetAccuracy;
            result.bestFeatureSet = emptySet;
        }
    }
    
    if (verbose) {
        std::cout << "Finished search!! The best feature subset is " 
                  << featureSetToString(result.bestFeatureSet) 
                  << ", which has an accuracy of " << std::fixed << std::setprecision(1) 
                  << (result.bestAccuracy * 100.0) << "%" << std::endl;
    }
    
    return result;
}

void FeatureSelection::printSearchResults(
    const SearchResult& result, 
    const std::string& algorithmName
) {
    std::cout << "\n===== " << algorithmName << " Results =====" << std::endl;
    std::cout << "Best feature subset: " << featureSetToString(result.bestFeatureSet) << std::endl;
    std::cout << "Best accuracy: " << std::fixed << std::setprecision(1) 
              << (result.bestAccuracy * 100.0) << "%" << std::endl;
    
    std::cout << "\nFeature Sets Evaluated:" << std::endl;
    for (const auto& [featureSet, accuracy] : result.allResults) {
        std::cout << "  " << featureSetToString(featureSet) << ": " 
                  << std::fixed << std::setprecision(1) << (accuracy * 100.0) << "%" << std::endl;
    }
    std::cout << std::endl;
}

} // namespace feature_selection