#ifndef QAP_CORE_PROBLEM_H
#define QAP_CORE_PROBLEM_H

#include <vector>
#include <string>
#include <fstream>

/**
 * Represents a Quadratic Assignment Problem instance.
 */
class Problem {
public:
    int n;                           // Problem size
    std::vector<std::vector<double>> F;  // Flow matrix (n x n)
    std::vector<std::vector<double>> D;  // Distance matrix (n x n)

    /**
     * Constructor.
     *
     * @param n Problem size
     * @param F Flow matrix
     * @param D Distance matrix
     */
    Problem(int n, const std::vector<std::vector<double>>& F,
            const std::vector<std::vector<double>>& D)
        : n(n), F(F), D(D) {}

    /**
     * Load a QAP instance from QAPLIB format file.
     *
     * @param filepath Path to the QAPLIB file
     * @return Problem instance
     */
    static Problem fromQAPLIB(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        int n;
        file >> n;

        // Read flow matrix
        std::vector<std::vector<double>> F(n, std::vector<double>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                file >> F[i][j];
            }
        }

        // Read distance matrix
        std::vector<std::vector<double>> D(n, std::vector<double>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                file >> D[i][j];
            }
        }

        file.close();
        return Problem(n, F, D);
    }

    /**
     * Evaluate the objective value for a given assignment.
     *
     * @param assignment Permutation π where π[i] = j means facility i → location j
     * @return Objective value
     */
    double evaluate(const std::vector<int>& assignment) const {
        if (assignment.size() != n) {
            throw std::runtime_error("Assignment size does not match problem size");
        }

        double obj = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                obj += F[i][j] * D[assignment[i]][assignment[j]];
            }
        }
        return obj;
    }
};

#endif  // QAP_CORE_PROBLEM_H
