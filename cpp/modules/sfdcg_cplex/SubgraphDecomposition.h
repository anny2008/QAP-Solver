#pragma once
#include "../../core/problem.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>


/**
 * Subgraph struct
 * id k: index of the subgraph
 * arcs: set of arcs (u,v) in this subgraph
 * nodes: set of nodes in this subgraph
 * degree_out[u]: degree of node u in this subgraph (number of arcs (u,v) in subgraph)
 * degree_in[v]: degree of node v in this subgraph (number of arcs (u,v) in subgraph)
 */
struct Subgraph {
    int k;
    double F_k;
    std::unordered_set<int> nodes; // nodes in this subgraph
    std::vector<std::pair<int,int>> arcs; // arcs (u,v) in this subgraph
    std::unordered_map<int,int> degree_out; // degree_out[u] for u in M
    std::unordered_map<int,int> degree_in;  // degree_in[v] for v in M
    // constructor to build from list of arcs
    Subgraph(int k_, double _F_k, const std::vector<std::pair<int,int>>& arc_list) : k(k_), F_k(_F_k) {
        for (const auto& arc : arc_list) {
            int u = arc.first, v = arc.second;
            arcs.push_back({u,v});
            nodes.insert(u);
            nodes.insert(v);
            degree_out[u]++;
            degree_in[v]++;
        }
    }
};

std::vector<Subgraph> decomposeIntoSubgraphs(const Problem& problem, std::string strategy);
// Decomposition of the flows graph of the problem into subgraphs, using layer strategy
std::vector<Subgraph> decomposeIntoSubgraphsLayer(const Problem& problem);
// Decomposition of the flows graph of the problem into subgraphs, using value grouping strategy
std::vector<Subgraph> decomposeIntoSubgraphsValueGrouping(const Problem& problem);
// save decomposition result to file for loading later
void saveDecomposition(const std::vector<Subgraph>& subgraphs, const std::string& filepath);
// load decomposition result from file
std::vector<Subgraph> loadDecomposition(const std::string& filepath);
