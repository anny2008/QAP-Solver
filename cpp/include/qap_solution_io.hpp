/**
 * Unified QAP Solution I/O Module (C++)
 * 
 * Provides consistent utilities for reading and writing QAP solutions
 * in QAPLIB format across all solvers.
 * 
 * Format (QAPLIB):
 *   Line 1: n optimal_value
 *   Line 2: location_1 location_2 ... location_n (1-indexed)
 * 
 * Example:
 *   12  578
 *   12  7  9  3  4  8  11  1  5  6  10  2
 */

#pragma once

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <iomanip>

namespace qap {

/**
 * Read a QAP solution from QAPLIB format file.
 * 
 * @param filepath Path to .sln file
 * @param n Output: problem size
 * @param assignment Output: assignment array (0-indexed)
 * @param objective Output: objective value
 * @throws std::runtime_error if file format is invalid
 */
inline void read_solution(const std::string& filepath, int& n, std::vector<int>& assignment, double& objective)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    // Read first line: n optimal_value
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty file: " + filepath);
    }
    
    std::istringstream iss(line);
    if (!(iss >> n >> objective)) {
        throw std::runtime_error("Invalid format in " + filepath + ": first line should have 'n objective'");
    }
    
    // Read second line: assignment (1-indexed)
    if (!std::getline(file, line)) {
        throw std::runtime_error("Missing assignment line in " + filepath);
    }
    
    std::istringstream ass_stream(line);
    assignment.clear();
    assignment.reserve(n);
    
    int location_1indexed;
    while (ass_stream >> location_1indexed) {
        // Convert from 1-indexed to 0-indexed
        assignment.push_back(location_1indexed - 1);
    }
    
    if ((int)assignment.size() != n) {
        throw std::runtime_error("Expected " + std::to_string(n) + " assignments, got " + std::to_string(assignment.size()));
    }
}

/**
 * Write a QAP solution to QAPLIB format file.
 * 
 * @param filepath Path to output .sln file
 * @param n Problem size
 * @param assignment Assignment array (0-indexed)
 * @param objective Objective value
 * @throws std::runtime_error if assignment size doesn't match n
 */
inline void write_solution(const std::string& filepath, int n, const std::vector<int>& assignment, double objective)
{
    if ((int)assignment.size() != n) {
        throw std::runtime_error("Assignment size " + std::to_string(assignment.size()) + 
                                " doesn't match n=" + std::to_string(n));
    }
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    
    // Write first line: n objective
    file << std::setw(6) << n << " " << std::setw(12) << std::fixed << std::setprecision(0) << objective << "\n";
    
    // Write second line: assignment (1-indexed, space-separated)
    for (int i = 0; i < n; ++i) {
        if (i > 0) file << " ";
        file << (assignment[i] + 1);  // Convert 0-indexed to 1-indexed
    }
    file << "\n";
}

/**
 * Read a warm-start solution into a map format.
 * 
 * @param filepath Path to .sln file
 * @return Map {(i, u): 1.0} for warm-starting (0-indexed keys)
 */
inline std::map<std::pair<int, int>, double> read_warmstart(const std::string& filepath)
{
    int n;
    std::vector<int> assignment;
    double obj;
    read_solution(filepath, n, assignment, obj);
    
    std::map<std::pair<int, int>, double> warm_start;
    for (int i = 0; i < n; ++i) {
        warm_start[{i, assignment[i]}] = 1.0;
    }
    
    return warm_start;
}

}  // namespace qap

#endif  // QAP_SOLUTION_IO_HPP
