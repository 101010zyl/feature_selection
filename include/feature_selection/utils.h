#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <set>
#include <chrono>
#include <iomanip>

namespace feature_selection {

// Type definitions for clarity
using FeatureIndex = std::size_t;
using FeatureSet = std::set<FeatureIndex>;
using DataPoint = std::vector<double>;
using DataMatrix = std::vector<DataPoint>;
using Label = int;
using LabelVector = std::vector<Label>;

// Helper to print feature sets in a readable format
inline std::string featureSetToString(const FeatureSet& features) {
    if (features.empty()) {
        return "{}";
    }
    
    std::string result = "{";
    for (auto it = features.begin(); it != features.end(); ++it) {
        if (it != features.begin()) {
            result += ",";
        }
        result += std::to_string(*it);
    }
    result += "}";
    return result;
}

// Timer class for measuring algorithm performance
class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::string name;

public:
    Timer(const std::string& timer_name) : name(timer_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double seconds = duration.count() / 1000.0;
        
        std::cout << name << " took ";
        if (seconds < 60.0) {
            std::cout << std::fixed << std::setprecision(2) << seconds << " seconds" << std::endl;
        } else {
            int minutes = static_cast<int>(seconds) / 60;
            double remaining_seconds = seconds - (minutes * 60);
            std::cout << minutes << " minutes, " << std::fixed << std::setprecision(2) 
                      << remaining_seconds << " seconds" << std::endl;
        }
    }
};

} // namespace feature_selection