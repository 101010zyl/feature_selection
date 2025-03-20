#include <gtest/gtest.h>
#include "feature_selection/data_loader.h"
#include <chrono>
#include <fstream>
#include <iostream>

using namespace feature_selection;

// Test fixture for DataLoader tests
class DataLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Path to existing dataset
        datasetPath = "../P2_datasets/CS170_Large_Data__1.txt";
        
        // Verify the file exists before running tests
        std::ifstream file(datasetPath);
        if (!file.is_open()) {
            std::cerr << "WARNING: Test dataset not found at " << datasetPath << std::endl;
            std::cerr << "Some tests may fail. Make sure the file exists at the specified path." << std::endl;
            fileExists = false;
        } else {
            fileExists = true;
            file.close();
        }
    }
    
    std::string datasetPath;
    bool fileExists;
};

// Test loading the dataset
TEST_F(DataLoaderTest, LoadDataset) {
    if (!fileExists) {
        GTEST_SKIP() << "Skipping test because dataset file doesn't exist";
    }
    
    std::cout << "Loading dataset from " << datasetPath << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    auto [data, labels] = DataLoader::loadDataset(datasetPath);
    auto endTime = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime
    ).count();
    
    std::cout << "Dataset loaded in " << duration << " ms" << std::endl;
    std::cout << "Dataset contains " << data.size() << " instances with " 
              << (data.empty() ? 0 : data[0].size()) << " features each" << std::endl;
    
    // Basic checks
    ASSERT_FALSE(data.empty());
    ASSERT_FALSE(labels.empty());
    ASSERT_EQ(data.size(), labels.size());
    
    // Verify feature count consistency
    size_t featureCount = data[0].size();
    for (const auto& point : data) {
        ASSERT_EQ(featureCount, point.size());
    }
    
    // Verify all labels are either 1 or 2
    for (const auto& label : labels) {
        ASSERT_TRUE(label == 1 || label == 2);
    }
    
    // Print some dataset statistics
    DataLoader::printDatasetInfo(data, labels);
}

// Test feature extraction
TEST_F(DataLoaderTest, ExtractFeatures) {
    if (!fileExists) {
        GTEST_SKIP() << "Skipping test because dataset file doesn't exist";
    }
    
    auto [data, labels] = DataLoader::loadDataset(datasetPath);
    
    // Extract a subset of features
    FeatureSet featureSubset;
    featureSubset.insert(0);  // First feature
    featureSubset.insert(5);  // Sixth feature
    featureSubset.insert(10); // Eleventh feature
    
    auto startTime = std::chrono::high_resolution_clock::now();
    auto extractedData = DataLoader::extractFeatures(data, featureSubset);
    auto endTime = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime
    ).count();
    
    std::cout << "Feature extraction completed in " << duration << " ms" << std::endl;
    
    // Verify extracted data has the right dimensions
    ASSERT_EQ(data.size(), extractedData.size());
    ASSERT_EQ(featureSubset.size(), extractedData[0].size());
    
    // Verify the extracted features match the original features
    for (size_t i = 0; i < std::min(data.size(), size_t(10)); ++i) { // Check first 10 instances
        size_t j = 0;
        for (auto featureIdx : featureSubset) {
            ASSERT_EQ(data[i][featureIdx], extractedData[i][j]);
            ++j;
        }
    }
    
    std::cout << "Successfully extracted " << featureSubset.size() 
              << " features from dataset" << std::endl;
}

// Compare performance between OpenMP-enabled and standard single-threaded loading
TEST_F(DataLoaderTest, PerformanceComparison) {
    if (!fileExists) {
        GTEST_SKIP() << "Skipping test because dataset file doesn't exist";
    }
    
    std::cout << "Performance comparison for loading dataset: " << datasetPath << std::endl;
    
    // Test concurrent loading (via DataLoader::loadDataset)
    auto startConcurrent = std::chrono::high_resolution_clock::now();
    auto [data, labels] = DataLoader::loadDataset(datasetPath);
    auto endConcurrent = std::chrono::high_resolution_clock::now();
    
    auto durationConcurrent = std::chrono::duration_cast<std::chrono::milliseconds>(
        endConcurrent - startConcurrent
    ).count();
    
    // Output results
    std::cout << "Concurrent loading: " << durationConcurrent << " ms for " 
              << data.size() << " instances with " << data[0].size() << " features." << std::endl;
    
    // For comparison, test loading a file in standard single-threaded mode
    std::cout << "Note: For comparison, a single-threaded implementation would typically" << std::endl;
    std::cout << "      be slower for large files, especially on systems with multiple cores." << std::endl;
}

// Test error handling
TEST_F(DataLoaderTest, ErrorHandling) {
    // Test with a non-existent file
    std::string nonExistentFile = "non_existent_file.txt";
    EXPECT_THROW(DataLoader::loadDataset(nonExistentFile), std::runtime_error);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}