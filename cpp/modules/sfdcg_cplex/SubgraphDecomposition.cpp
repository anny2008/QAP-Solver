#include "SubgraphDecomposition.h"
#include <limits>

std::vector<Subgraph> decomposeIntoSubgraphs(const Problem& problem, std::string strategy) {
    if (strategy == "layer" || strategy == "value_layer") {
        return decomposeIntoSubgraphsLayer(problem);
    } else if (strategy == "value_grouping" || strategy == "value_only") {
        return decomposeIntoSubgraphsValueGrouping(problem);
    } else {
        throw std::invalid_argument("Invalid decomposition strategy: " + strategy);
    }
}


/**
 * Decomposition of the flows graph of the problem into subgraphs, using layer strategy
 * First take the minimum positive flow value, and create a subgraph with all arcs with that flow value;
 *  then remove those arcs and repeat until no arcs remain
*/ 

std::vector<Subgraph> decomposeIntoSubgraphsLayer(const Problem& problem){
    std::vector<Subgraph> subgraphs;
    // copy problem F to F
    auto F = std::vector<std::vector<double>>(problem.F);
    int k = 0;
    // Build flow graph as list of arcs (u,v) with flow value F[u][v]
    std::vector<std::tuple<int,int,double>> arcs;
    while (true) {
        arcs.clear();
        for (int u=0;u<problem.m;++u) {
            for (int v=0;v<problem.m;++v) {
                if (F[u][v] > 0.0) {
                    arcs.push_back({u,v,F[u][v]});
                }
            }
        }
        if (arcs.empty()) break;
        // Find minimum positive flow value
        double min_flow = std::numeric_limits<double>::max();
        for (const auto& arc : arcs) {
            min_flow = std::min(min_flow, std::get<2>(arc));
        }
        if (min_flow == 0.)
            break;
        // Create subgraph with all arcs with that flow value
        std::vector<std::pair<int,int>> subgraph_arcs;
        for (const auto& arc : arcs) {
            if (std::get<2>(arc) == min_flow) {
                subgraph_arcs.push_back({std::get<0>(arc), std::get<1>(arc)});
            }
        }
        subgraphs.emplace_back(k++, min_flow, subgraph_arcs);
        // Substract min_flow from those arc
        std::vector<std::vector<double>> F = problem.F;

        for (int u = 0; u < F.size(); ++u) {
            for (int v = 0; v < F[u].size(); ++v) {
                if (F[u][v] > 0.0)
                    F[u][v] -= min_flow;
            }
        }
    }
    return subgraphs;
}
/**
 * Decomposition of the flows graph of the problem into subgraphs, using value grouping strategy
 * Group all arcs by their flow value, and create a subgraph for each distinct flow value;
*/ 
std::vector<Subgraph> decomposeIntoSubgraphsValueGrouping(const Problem& problem) {
    std::vector<Subgraph> subgraphs;
    auto F = problem.F;
    int k = 0;
    // Build flow graph as list of arcs (u,v) with flow value F[u][v]
    std::unordered_map<double, std::vector<std::pair<int,int>>> flow_groups;
    for (int u=0;u<problem.m;++u) {
        for (int v=0;v<problem.m;++v) {
            if (F[u][v] > 0.0) {
                flow_groups[F[u][v]].push_back({u,v});
            }
        }
    }
    // Create subgraph for each distinct flow value
    for (const auto& kv : flow_groups) {
        subgraphs.emplace_back(k++, kv.first, kv.second);
    }
    return subgraphs;
}

void saveDecomposition(const std::vector<Subgraph>& subgraphs, const std::string& filepath)
{
    std::ofstream out(filepath);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    // Write number of subgraphs
    out << subgraphs.size() << std::endl;
    // For each subgraph, write k, F_k, number of arcs, then list of arcs
    for (const auto& subgraph : subgraphs) {
        out << subgraph.k << " " << subgraph.F_k << " " << subgraph.arcs.size() << std::endl;
        for (const auto& arc : subgraph.arcs) {
            out << arc.first << " " << arc.second << std::endl;
        }
    }
}

// load decomposition result from file
std::vector<Subgraph> loadDecomposition(const std::string& filepath)
{
    std::ifstream in(filepath);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    std::vector<Subgraph> subgraphs;
    int num_subgraphs;
    in >> num_subgraphs;
    for (int i = 0; i < num_subgraphs; ++i) {
        int k, num_arcs;
        double F_k;
        in >> k >> F_k >> num_arcs;
        std::vector<std::pair<int,int>> arcs(num_arcs);
        for (int j = 0; j < num_arcs; ++j) {
            int u, v;
            in >> u >> v;
            arcs[j] = {u,v};
        }
        subgraphs.emplace_back(k, F_k, arcs);
    }
    return subgraphs;
}