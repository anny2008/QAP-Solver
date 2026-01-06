/**
 * Unified QAP Solution I/O Module (C++)
 * 
 * Provides consistent utilities for reading and writing QAP solutions
 * in QAPLIB format across all solvers.
 * 
 * Format (framework matching current .sln files):
 *   Line 1: n objective_value
 *   Line 2: location_1 location_2 ... location_n  (values are facilities, 1-indexed)
 * 
 * Example:
 *   578  12  7  9  3  4  8  11  1  5  6  10  2
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

    // First line: n objective
    if (!(file >> n >> objective)) {
        throw std::runtime_error("Invalid solution header in: " + filepath);
    }

    assignment.clear();
    assignment.reserve(n);
    int fac1;
    while (file >> fac1) {
        assignment.push_back(fac1 - 1); // loc -> facility, 0-indexed
    }

    if ((int)assignment.size() != n) {
        throw std::runtime_error("Expected " + std::to_string(n) + " facilities, got " + std::to_string(assignment.size()));
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
    
    // Write legacy/current format: line1 n objective, line2 loc->facility (1-indexed)
    file << std::setw(6) << n << " " << std::setw(12) << std::fixed << std::setprecision(0) << objective << "\n";
    for (int i = 0; i < n; ++i) {
        if (i > 0) file << " ";
        file << (assignment[i] + 1);
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
