#pragma once

#include "feature_selection/utils.h"
#include <string>
#include <tuple>
#include <vector>
#include <fstream>
#include <sstream>
#include <future>
#include <thread>
#include <memory>
#include <stdexcept>

namespace feature_selection {

/**
 * @brief Class for loading and manipulating datasets
 */
class DataLoader {
public:
    /**
     * @brief Loads data from a file and returns feature matrix and label vector
     * @param filename Path to the dataset file
     * @return Tuple containing (data matrix, label vector)
     * 
     * Format: First column is class label (1 or 2), remaining columns are features
     */
    static std::tuple<DataMatrix, LabelVector> loadDataset(const std::string& filename);
    
    /**
     * @brief Get the number of features in the dataset
     * @param data The dataset to analyze
     * @return Number of features per instance
     */
    static std::size_t getFeatureCount(const DataMatrix& data);
    
    /**
     * @brief Get the number of instances in the dataset
     * @param data The dataset to analyze
     * @return Number of instances (rows) in the dataset
     */
    static std::size_t getInstanceCount(const DataMatrix& data);
    
    /**
     * @brief Print basic dataset statistics
     * @param data The dataset to analyze
     * @param labels The corresponding labels
     */
    static void printDatasetInfo(const DataMatrix& data, const LabelVector& labels);
    
    /**
     * @brief Extract a subset of features from the data
     * @param data The full dataset
     * @param features The set of features to extract
     * @return Data matrix containing only the selected features
     */
    static DataMatrix extractFeatures(const DataMatrix& data, const FeatureSet& features);

private:
    /**
     * @brief Read a chunk of a file using the specified bounds
     * @param filename File to read
     * @param startPos Starting position in the file
     * @param chunkSize Size of the chunk to read in bytes
     * @return Vector of rows where each row contains the raw values (including label)
     */
    static std::vector<std::vector<double>> readFileChunk(
        const std::string& filename, 
        size_t startPos, 
        size_t chunkSize
    );
    
    /**
     * @brief Read a file concurrently using multiple threads
     * @param filename File to read
     * @return Vector of rows where each row contains the raw values (including label)
     */
    static std::vector<std::vector<double>> readFileConcurrent(const std::string& filename);
    
    /**
     * @brief Process raw data into features and labels
     * @param rawData The raw data from the file (includes labels)
     * @return Tuple containing (data matrix, label vector)
     */
    static std::tuple<DataMatrix, LabelVector> processRawData(
        const std::vector<std::vector<double>>& rawData
    );
    
    /**
     * @brief Verify dataset consistency (all rows have same number of features)
     * @param data The dataset to check
     * @return True if the dataset is consistent, false otherwise
     */
    static bool verifyDatasetConsistency(const DataMatrix& data);
};

} // namespace feature_selection