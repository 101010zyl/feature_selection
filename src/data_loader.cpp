#include "feature_selection/data_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <iomanip>

namespace feature_selection {

// std::tuple<DataMatrix, LabelVector> DataLoader::loadDataset(const std::string& filename) {
//     std::ifstream file(filename);
//     if (!file.is_open()) {
//         throw std::runtime_error("Could not open file: " + filename);
//     }
    
//     DataMatrix data;
//     LabelVector labels;
    
//     std::string line;
//     while (std::getline(file, line)) {
//         std::istringstream iss(line);
//         double value;
        
//         // First value is the class label
//         if (!(iss >> value)) {
//             continue; // Skip empty or invalid lines
//         }
        
//         labels.push_back(static_cast<Label>(value));
        
//         // Remaining values are features
//         DataPoint point;
//         while (iss >> value) {
//             point.push_back(value);
//         }
        
//         if (!point.empty()) {
//             data.push_back(point);
//         }
//     }
    
//     // Check if data was loaded correctly
//     if (data.empty()) {
//         throw std::runtime_error("No data loaded from file: " + filename);
//     }
    
//     // Verify all data points have the same number of features
//     std::size_t featureCount = data[0].size();
//     for (const auto& point : data) {
//         if (point.size() != featureCount) {
//             throw std::runtime_error("Inconsistent feature count in dataset");
//         }
//     }
    
//     // Verify labels and data points match
//     if (labels.size() != data.size()) {
//         throw std::runtime_error("Mismatch between number of labels and data points");
//     }
    
//     return {data, labels};
// }

std::size_t DataLoader::getFeatureCount(const DataMatrix& data) {
    if (data.empty()) {
        return 0;
    }
    return data[0].size();
}

std::size_t DataLoader::getInstanceCount(const DataMatrix& data) {
    return data.size();
}

void DataLoader::printDatasetInfo(const DataMatrix& data, const LabelVector& labels) {
    if (data.empty()) {
        std::cout << "Dataset is empty." << std::endl;
        return;
    }
    
    std::size_t instanceCount = getInstanceCount(data);
    std::size_t featureCount = getFeatureCount(data);
    
    std::cout << "This dataset has " << featureCount 
              << " features (not including the class attribute), with "
              << instanceCount << " instances." << std::endl;
    
    // Count classes
    std::unordered_map<Label, int> classCounts;
    for (const auto& label : labels) {
        classCounts[label]++;
    }
    
    std::cout << "Class distribution:" << std::endl;
    for (const auto& [label, count] : classCounts) {
        double percentage = 100.0 * count / static_cast<double>(instanceCount);
        std::cout << "  Class " << label << ": " << count << " instances (" 
                  << std::fixed << std::setprecision(1) << percentage << "%)" << std::endl;
    }
    
    // Calculate default accuracy (if we predict the majority class)
    Label majorityClass = 0;
    int maxCount = 0;
    for (const auto& [label, count] : classCounts) {
        if (count > maxCount) {
            maxCount = count;
            majorityClass = label;
        }
    }
    
    double defaultAccuracy = 100.0 * maxCount / static_cast<double>(instanceCount);
    std::cout << "Default accuracy (always predict class " << majorityClass << "): " 
              << std::fixed << std::setprecision(1) << defaultAccuracy << "%" << std::endl;
}

DataMatrix DataLoader::extractFeatures(const DataMatrix& data, const FeatureSet& features) {
    if (data.empty() || features.empty()) {
        return DataMatrix();
    }
    
    DataMatrix result;
    result.reserve(data.size());
    
    for (const auto& dataPoint : data) {
        DataPoint newPoint;
        newPoint.reserve(features.size());
        
        for (FeatureIndex idx : features) {
            if (idx < dataPoint.size()) {
                newPoint.push_back(dataPoint[idx]);
            }
        }
        
        result.push_back(newPoint);
    }
    
    return result;
}

// These are the new concurrent file reading methods

std::vector<std::vector<double>> DataLoader::readFileChunk(
    const std::string& filename,
    size_t startPos, 
    size_t chunkSize
) {
    std::vector<std::vector<double>> chunkData;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    file.seekg(startPos);
    std::string line;
    size_t bytesRead = 0;
    
    // If not at the beginning of the file, discard the first (partial) line
    if (startPos > 0) {
        std::getline(file, line);
        bytesRead += line.length() + 1; // +1 for newline character
    }
    
    // Read and process complete lines
    while (bytesRead < chunkSize && std::getline(file, line)) {
        bytesRead += line.length() + 1; // +1 for newline character
        
        std::vector<double> row;
        std::istringstream iss(line);
        double value;
        
        while (iss >> value) {
            row.push_back(value);
        }
        
        if (!row.empty()) {
            chunkData.push_back(row);
        }
    }
    
    return chunkData;
}

std::vector<std::vector<double>> DataLoader::readFileConcurrent(const std::string& filename) {
    // Get file size
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    const size_t fileSize = file.tellg();
    file.close();
    
    // Determine optimal number of threads (not more than 8)
    const unsigned int numThreads = std::min(8u, std::thread::hardware_concurrency());
    const size_t chunkSize = fileSize / numThreads;
    
    // Prepare threads and futures
    std::vector<std::future<std::vector<std::vector<double>>>> futures;
    std::vector<std::vector<double>> result;
    
    // Reserve space for the results based on an estimate
    result.reserve(fileSize / 100); // Rough estimate, adjust based on avg line length
    
    // Create a thread for each chunk
    for (unsigned int i = 0; i < numThreads; ++i) {
        size_t startPos = i * chunkSize;
        size_t endPos = (i == numThreads - 1) ? fileSize : startPos + chunkSize;
        size_t actualChunkSize = endPos - startPos;
        
        // Use async to launch threads
        futures.push_back(std::async(std::launch::async, 
            [filename, startPos, actualChunkSize]() {
                return readFileChunk(filename, startPos, actualChunkSize);
            }
        ));
    }
    
    // Collect results from all threads
    for (auto& future : futures) {
        auto chunkData = future.get();
        result.insert(result.end(), chunkData.begin(), chunkData.end());
    }
    
    return result;
}

std::tuple<DataMatrix, LabelVector> DataLoader::processRawData(
    const std::vector<std::vector<double>>& rawData
) {
    DataMatrix data;
    LabelVector labels;
    
    // Reserve space for efficiency
    data.reserve(rawData.size());
    labels.reserve(rawData.size());
    
    // Process each row of the raw data
    for (const auto& row : rawData) {
        if (row.empty()) {
            continue;
        }
        
        // First column is the class label
        labels.push_back(static_cast<Label>(row[0]));
        
        // Remaining columns are features
        DataPoint point;
        point.reserve(row.size() - 1);
        
        for (size_t i = 1; i < row.size(); ++i) {
            point.push_back(row[i]);
        }
        
        data.push_back(point);
    }
    
    return {data, labels};
}

bool DataLoader::verifyDatasetConsistency(const DataMatrix& data) {
    if (data.empty()) {
        return true;
    }
    
    std::size_t featureCount = data[0].size();
    return std::all_of(data.begin(), data.end(), 
        [featureCount](const DataPoint& point) {
            return point.size() == featureCount;
        }
    );
}

// This is the updated loadDataset method that uses concurrent file reading
std::tuple<DataMatrix, LabelVector> DataLoader::loadDataset(const std::string& filename) {
    try {
        // Read the data concurrently
        auto rawData = readFileConcurrent(filename);
        
        // Process raw data into features and labels
        auto [data, labels] = processRawData(rawData);
        
        // Verify data consistency
        if (!verifyDatasetConsistency(data)) {
            throw std::runtime_error("Inconsistent feature count in dataset");
        }
        
        // Verify labels and data points match
        if (labels.size() != data.size()) {
            throw std::runtime_error("Mismatch between number of labels and data points");
        }
        
        return {data, labels};
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Failed to load dataset '" + filename + "': " + e.what());
    }
}

} // namespace feature_selection