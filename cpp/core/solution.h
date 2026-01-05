#ifndef QAP_CORE_SOLUTION_H
#define QAP_CORE_SOLUTION_H

#include <vector>
#include <string>
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * Represents a QAP solver result.
 */
class Solution {
public:
    std::string instance;                    // Instance name
    std::string solver;                      // Solver identifier
    std::vector<int> assignment;             // Permutation
    double objective = -1.0;                 // Objective value (-1 = not set)
    double lower_bound = -1.0;               // Lower bound (-1 = not set)
    double time = 0.0;                       // Execution time

    /**
     * Constructor.
     *
     * @param instance Instance name
     * @param solver Solver identifier
     */
    Solution(const std::string& instance, const std::string& solver)
        : instance(instance), solver(solver) {}

    /**
     * Calculate optimality gap (%).
     *
     * @return Gap percentage, or -1.0 if gap cannot be computed
     */
    double getGap() const {
        if (objective < 0 || lower_bound < 0 || lower_bound == 0) {
            return -1.0;
        }
        return 100.0 * (objective - lower_bound) / lower_bound;
    }

    /**
     * Convert solution to JSON.
     *
     * @return JSON representation
     */
    json toJSON() const {
        json result;
        result["instance"] = instance;
        result["solver"] = solver;
        result["objective"] = (objective < 0) ? nullptr : json(objective);
        result["lower_bound"] = (lower_bound < 0) ? nullptr : json(lower_bound);

        double gap = getGap();
        result["gap"] = (gap < 0) ? nullptr : json(gap);
        result["time"] = time;
        result["assignment"] =
            assignment.empty() ? nullptr : json(assignment);

        return result;
    }

    /**
     * Write solution to JSON file.
     *
     * @param filepath Output file path
     */
    void write(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        file << toJSON().dump(2) << std::endl;
        file.close();
    }
};

#endif  // QAP_CORE_SOLUTION_H
