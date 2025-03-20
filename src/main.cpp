#include "feature_selection/data_loader.h"
#include <iostream>
#include <string>

using namespace feature_selection;

int main(int argc, char** argv) {
    std::cout << "Feature Selection Data Loader Test" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Default dataset path
    std::string datasetPath = "../P2_datasets/CS170_Large_Data__1.txt";
    
    // If command line argument is provided, use it as the dataset path
    if (argc > 1) {
        datasetPath = argv[1];
    }
    
    std::cout << "Loading dataset: " << datasetPath << std::endl;
    
    try {
        // Load dataset
        auto [data, labels] = DataLoader::loadDataset(datasetPath);
        
        // Print dataset information
        std::cout << "\nDataset Information:" << std::endl;
        DataLoader::printDatasetInfo(data, labels);
        
        // Print first few data points for verification
        std::cout << "\nFirst 5 data points (showing first 3 features):" << std::endl;
        for (size_t i = 0; i < std::min(data.size(), size_t(5)); ++i) {
            std::cout << "Instance " << i << ": Class " << labels[i] << " - Features: [";
            for (size_t j = 0; j < std::min(data[i].size(), size_t(3)); ++j) {
                std::cout << data[i][j];
                if (j < std::min(data[i].size(), size_t(3)) - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << ", ...]" << std::endl;
        }
        
        std::cout << "\nData loading successful!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}