#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <iterator>

using Clique = std::set<int>;

std::vector<Clique> groupCliquesSparseQAP(const Problem& qap, int C) {
    int n = qap.n;
    if (n == 0 || C < 2) return {};
    
    // 1. Calculate interaction density for sorting
    std::vector<std::pair<double, int>> node_density(n);
    for (int i = 0; i < n; ++i) {
        double weight = 0;
        for (int j = 0; j < n; ++j) {
            if (i != j) weight += std::abs(qap.D[i][j]);
        }
        node_density[i] = {weight, i};
    }
    std::sort(node_density.rbegin(), node_density.rend());

    // 2. Direct fast block extraction
    std::vector<Clique> rawBlocks;
    for (const auto& [weight, seed_node] : node_density) {
        std::vector<int> neighbors;
        for (int j = 0; j < n; ++j) {
            if (seed_node != j && std::abs(qap.D[seed_node][j]) > 1e-9) {
                neighbors.push_back(j);
            }
        }
        
        if (neighbors.empty()) {
            rawBlocks.push_back({seed_node});
            continue;
        }

        size_t neighbor_idx = 0;
        while (neighbor_idx < neighbors.size()) {
            Clique subClique;
            subClique.insert(seed_node);
            while (subClique.size() < static_cast<size_t>(C) && neighbor_idx < neighbors.size()) {
                subClique.insert(neighbors[neighbor_idx]);
                neighbor_idx++;
            }
            rawBlocks.push_back(subClique);
        }
    }

    // Sort blocks descending by size to prepare for high-quality greedy merging
    std::sort(rawBlocks.begin(), rawBlocks.end(), [](const Clique& a, const Clique& b) {
        return a.size() > b.size();
    });

    // 3. Post-Processing: Consolidate smaller overlapping blocks up to limit C
    std::vector<Clique> finalGroupedCliques;

    for (const auto& block : rawBlocks) {
        int best_target_idx = -1;
        size_t max_intersection = 0;
        Clique best_union_set;

        for (size_t g_idx = 0; g_idx < finalGroupedCliques.size(); ++g_idx) {
            const auto& grouped = finalGroupedCliques[g_idx];
            
            std::vector<int> intersection;
            std::set_intersection(block.begin(), block.end(),
                                  grouped.begin(), grouped.end(),
                                  std::back_inserter(intersection));

            if (!intersection.empty()) {
                Clique testUnion = grouped;
                testUnion.insert(block.begin(), block.end());

                if (testUnion.size() <= static_cast<size_t>(C)) {
                    if (intersection.size() > max_intersection) {
                        max_intersection = intersection.size();
                        best_target_idx = static_cast<int>(g_idx);
                        best_union_set = std::move(testUnion);
                    }
                }
            }
        }

        if (best_target_idx != -1) {
            finalGroupedCliques[best_target_idx] = std::move(best_union_set);
        } else {
            finalGroupedCliques.push_back(block);
        }
    }

    // Erase duplicate blocks
    std::sort(finalGroupedCliques.begin(), finalGroupedCliques.end());
    finalGroupedCliques.erase(std::unique(finalGroupedCliques.begin(), finalGroupedCliques.end()), finalGroupedCliques.end());

    return finalGroupedCliques;
}

std::vector<Clique> groupCliquesStrictOverlap(const Problem& qap, int C) {
    int n = qap.n;
    if (n == 0 || C < 2) return {};

    // 1. Calculate interaction density to pick the best "hub" / pivot nodes
    std::vector<std::pair<double, int>> node_density(n);
    for (int i = 0; i < n; ++i) {
        double weight = 0;
        for (int j = 0; j < n; ++j) {
            if (i != j) weight += std::abs(qap.D[i][j]);
        }
        node_density[i] = {weight, i};
    }
    // Sort nodes descending by density to prioritize packing heavy nodes first
    std::sort(node_density.rbegin(), node_density.rend());

    std::vector<bool> visited(n, false);
    std::vector<Clique> finalCliques;

    // 2. Build the first primary clique from the absolute heaviest nodes
    Clique firstClique;
    for (int i = 0; i < n && firstClique.size() < static_cast<size_t>(C); ++i) {
        int node = node_density[i].second;
        firstClique.insert(node);
        visited[node] = true;
    }
    if (!firstClique.empty()) {
        finalCliques.push_back(firstClique);
    }

    // 3. Iteratively sprout new cliques using exactly ONE pivot node from existing cliques
    // to guarantee that no two cliques ever share more than 1 node.
    size_t density_idx = 0;
    
    while (true) {
        // Find the next unvisited node with the highest interaction density
        int next_unvisited = -1;
        while (density_idx < node_density.size()) {
            int node = node_density[density_idx].second;
            if (!visited[node]) {
                next_unvisited = node;
                break;
            }
            density_idx++;
        }

        // If all nodes in the graph are visited, we are done
        if (next_unvisited == -1) break;

        Clique newClique;
        newClique.insert(next_unvisited);
        visited[next_unvisited] = true;

        // Find the best existing node to act as the SINGLE pivot/hub for this new clique.
        // We pick the visited node that has the highest edge weight to our new unvisited node.
        int best_pivot = -1;
        double max_pivot_weight = -1.0;
        for (int v_node = 0; v_node < n; ++v_node) {
            if (visited[v_node] && v_node != next_unvisited) {
                double edge_w = std::abs(qap.D[next_unvisited][v_node]);
                if (edge_w > max_pivot_weight) {
                    max_pivot_weight = edge_w;
                    best_pivot = v_node;
                }
            }
        }

        // Attach the single pivot node (this forms the 1-node overlap link)
        if (best_pivot != -1) {
            newClique.insert(best_pivot);
        }

        // Fill up the rest of this clique's capacity (C) using entirely fresh, unvisited nodes
        while (newClique.size() < static_cast<size_t>(C)) {
            int best_fresh_cand = -1;
            double max_cand_weight = -1.0;

            // Greedily find an unvisited node strongly connected to the current clique elements
            for (int cand = 0; cand < n; ++cand) {
                if (!visited[cand]) {
                    double total_connection = 0.0;
                    for (int member : newClique) {
                        total_connection += std::abs(qap.D[cand][member]);
                    }
                    if (total_connection > max_cand_weight) {
                        max_cand_weight = total_connection;
                        best_fresh_cand = cand;
                    }
                }
            }

            if (best_fresh_cand == -1) break; // No more fresh nodes available
            newClique.insert(best_fresh_cand);
            visited[best_fresh_cand] = true;
        }

        finalCliques.push_back(newClique);
    }

    return finalCliques;
}

std::vector<Clique> groupNodesNoOverlap(const Problem& qap, int C) {
    int n = qap.n;
    if (n == 0 || C < 2) return {};

    // 1. Calculate interaction density for high-quality greedy sorting
    std::vector<std::pair<double, int>> node_density(n);
    for (int i = 0; i < n; ++i) {
        double weight = 0;
        for (int j = 0; j < n; ++j) {
            if (i != j) weight += std::abs(qap.D[i][j]);
        }
        node_density[i] = {weight, i};
    }
    // Sort nodes descending by total cost interaction
    std::sort(node_density.rbegin(), node_density.rend());

    std::vector<Clique> partitioned_sets;
    std::vector<bool> assigned(n, false);

    // 2. Linear sweep to pack nodes into isolated bins of maximum size C
    for (int i = 0; i < n; ++i) {
        int seed_node = node_density[i].second;
        if (assigned[seed_node]) continue;

        Clique current_set;
        current_set.insert(seed_node);
        assigned[seed_node] = true;

        // Fill up this block's remaining slots up to capacity C
        // It scans down the sorted density list to grab the next best available node
        for (int j = i + 1; j < n && current_set.size() < static_cast<size_t>(C); ++j) {
            int candidate = node_density[j].second;
            if (!assigned[candidate]) {
                current_set.insert(candidate);
                assigned[candidate] = true;
            }
        }

        partitioned_sets.push_back(current_set);
    }

    return partitioned_sets;
}