#pragma once
#include "IncrementalRMP.h"
#include <optional>
#include <map>
#include "Config.h"

class SFDCGSolver {
public:
    SFDCGSolver(const SolverConfig& cfg);

    // Build and solve for a Problem
    Solution solve(Problem& problem, const std::string& instance_path);

private:
    SolverConfig cfg;
    /**
     * get all positive variable e from assignment
     * - assignment: Permutation π where π[u] = i means facility u → location i
     * - subgraphs: list of all subgraphs
     */
    std::vector<TripleKey> extractColumnFromAssignment(std::vector<int>& assignment, std::vector<Subgraph>& subgraphs);


};