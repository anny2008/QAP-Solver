#include "SFDGurobi.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <sstream>
#include "../local_search/local_search.h"
#include "../../include/qap_solution_io.hpp"
#include <omp.h>
#include <sfdvolume.hpp>
#include <set>


class SFDBendersCallback : public GRBCallback {
public:
    SFDBendersCallback(
        std::map<std::pair<int,int>, GRBVar>& x_vars,
        std::map<std::tuple<int,int,int>, GRBVar>& e_vars,
        std::map<int, SubgraphData>& subgraphs,
        Problem problem,
        int n,
        bool use_lazy_constraints,
        bool use_cuts
    ) : x(x_vars), e(e_vars),
        subgraphs(subgraphs), problem(problem), n(n),
        use_lazy_constraints(use_lazy_constraints),
        use_cuts(use_cuts),
        subproblem_env(),
        subproblem_model(nullptr), 
        y(), 
        subproblem_constraints_iuv(), 
        subproblem_constraints_iuj(), 
        subproblem_constraints_kij(),
        x_sol(), e_sol(), alpha(), beta(), gamma(), kij_indices()
    {
        // initialize kij_indices
        for (const auto &kv : subgraphs) {
            int k = kv.first;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i == j) continue;
                    kij_indices.push_back(std::make_tuple(k,i,j));
                }
            }
        }
    }

    ~SFDBendersCallback() {
        if (subproblem_model != nullptr) {
            delete subproblem_model;
            subproblem_model = nullptr;
        }
        // if (volume_hooks != nullptr) {
        //     delete volume_hooks;
        //     volume_hooks = nullptr;
        // }
    }

    void callback() override {
        try {
            if (where == GRB_CB_MIPSOL && use_lazy_constraints) {
                // update current objective value
                curObjVal = getDoubleInfo(GRB_CB_MIPSOL_OBJ);
                z_sol = getSolution(z_var);
                x_sol.clear();
                e_sol.clear();
                // obtain the MIP solution values for x and e variables
                for (const auto& kv : x) {
                    x_sol[kv.first] = getSolution(kv.second);
                }
                for (const auto& kv : e) {
                    e_sol[kv.first] = getSolution(kv.second);
                }
                // add lazy constraints based on the current solution
                addLazyCuts();
            } else if (where == GRB_CB_MIPNODE && use_cuts) {
                // update current objective value
                curObjVal = getDoubleInfo(GRB_CB_MIPNODE_OBJBND);
                // if node== 1, print the number of cuts added so far
                if (getDoubleInfo(GRB_CB_MIPNODE_NODCNT) == 1) {
                    std::cout << "At node 1, added lazy cuts: " << lazy_cuts_added << ", added node cuts: " << node_cuts_added << std::endl;
                }
                // if solving status is optimal, obtain the current node's LP relaxation solution values for x and e variables
                if (getIntInfo(GRB_CB_MIPNODE_STATUS) == GRB_OPTIMAL) {
                    x_sol.clear();
                    e_sol.clear();
                    for (const auto& kv : x) {
                        x_sol[kv.first] = getNodeRel(kv.second);
                    }
                    for (const auto& kv : e) {
                        e_sol[kv.first] = getNodeRel(kv.second);
                    }
                    z_sol = getNodeRel(z_var);
                    // add node cuts based on the current LP solution
                    addNodeCuts();
                }
                
            }
        } catch (GRBException &e) {
            std::cerr << "Error during callback: " << e.getMessage() << " (code=" << e.getErrorCode() << ")" << std::endl;
        } catch (std::exception &ex) {
            std::cerr << "Standard exception during callback: " << ex.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception during callback." << std::endl;
        }
    }

    int addedLazyCuts() const { return lazy_cuts_added; }
    int addedNodeCuts() const { return node_cuts_added; }


    GRBVar z_var;
    int n_callbacks = 0;
    double callback_time = 0.0;


private:
    std::map<std::pair<int,int>, GRBVar>& x;
    std::map<std::tuple<int,int,int>, GRBVar>& e;
    std::map<int, SubgraphData>& subgraphs;
    Problem problem;
    int n;
    bool use_lazy_constraints;
    bool use_cuts;
    int lazy_cuts_added = 0;
    int node_cuts_added = 0;
    double curObjVal = 0.0;
    GRBConstr obj_constr;
    double z_sol = 0.0;

    // store an env and a model for subproblem solving if needed for cut generation
    GRBEnv subproblem_env;
    GRBModel* subproblem_model = nullptr;
    std::map<std::tuple<int,int,int,int>, GRBVar> y;
    // store constraints
    std::vector<std::pair<GRBConstr,std::tuple<int,int,int>>> subproblem_constraints_iuv;
    std::vector<std::pair<GRBConstr,std::tuple<int,int,int>>> subproblem_constraints_iuj;
    std::vector<std::pair<GRBConstr,std::tuple<int,int,int>>> subproblem_constraints_kij;
    std::map<std::pair<int,int>, double> x_sol;
    std::map<std::tuple<int,int,int>, double> e_sol;


    std::map<std::tuple<int,int,int>, GRBVar> alpha;
    std::map<std::tuple<int,int,int>, GRBVar> beta;
    std::map<std::tuple<int,int,int>, GRBVar> gamma;

    // all kij index tuples
    std::vector<std::tuple<int,int,int>> kij_indices;


    std::map<std::pair<int,int>, int> uv_to_k;


    void addLazyCuts( )
    {
        std::vector<std::pair<GRBVar*,double>> cutlhs;
        if (findBenderCut(cutlhs)) {
            GRBLinExpr lhs = 0;
            for (const auto& pair : cutlhs) {
                lhs += pair.second * (*pair.first);
            }
            addLazy(lhs >= 0);
            lazy_cuts_added++;
        }
    }
    void addNodeCuts()
    {
        std::vector<std::pair<GRBVar*,double>> cutlhs;
        if (findBenderCut(cutlhs)) {
            GRBLinExpr lhs = 0;
            for (const auto& pair : cutlhs) {
                lhs += pair.second * (*pair.first);
            }
            addCut(lhs >= 0);
            node_cuts_added++;
        }
    }

    void setup_uv_to_k() {
        for (const auto &kv : subgraphs) {
            int k = kv.first;
            auto G_k = kv.second;
            for (const auto &uv : G_k.arcs) {
                int u = uv.first;
                int v = uv.second;
                if (uv_to_k.find(std::make_pair(u,v)) != uv_to_k.end()) {
                    std::cerr << "Error: arc (" << u << "," << v << ") appears in multiple subgraphs, which is not supported by the current implementation." << std::endl;
                }
                uv_to_k[std::make_pair(u,v)] = k;
            }
        }
        // check that every (u,v) is in exactly one subgraph
        for (int u =0, v = 0; u < n; ++u) {
            for (v = 0; v < n; ++v) {
                if (u == v) continue;
                if (uv_to_k.find(std::make_pair(u,v)) == uv_to_k.end()) {
                    std::cerr << "Error: arc (" << u << "," << v << ") does not appear in any subgraph, which is not supported by the current implementation." << std::endl;
                }
            }
        }
    }

    int verifyModel(std::vector<std::pair<std::string,std::tuple<int,int,int>>>& givenIIS, std::vector<std::pair<std::string,std::tuple<int,int,int>>>& returnIIS) {
        // create a new model with the same env as the subproblem model, add the constraints in the IIS of the subproblem model, and check if it is infeasible
        // reconstruct a model with only the constraints in the IIS to verify the cut
        try{

        GRBModel verify_model(subproblem_env);
        std::map<std::tuple<int,int,int,int>, GRBVar> y_verify;
        for (auto i = 0; i < n; ++i) {
            for (auto j = i+1; j < n; ++j) {
                for (auto u = 0; u < n; ++u) {
                    for (auto v = 0; v < n; ++v) {
                        if (u == v) continue;
                        std::string name = "y_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(u) + "_" + std::to_string(v);
                        auto y_iujv = verify_model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
                        y_verify[std::make_tuple(i,u,j,v)] = y_iujv;
                        y_verify[std::make_tuple(j,v,i,u)] = y_iujv; // symmetry
                    }
                }
            }
        }
        verify_model.update();
        for(auto pair : givenIIS) {
            auto constr_type = pair.first;
            auto indices = pair.second;
            auto a = std::get<0>(indices);
            auto b = std::get<1>(indices);
            auto c = std::get<2>(indices);
            // std::cout << "Adding constraint to verify model: " << constraint_name << ",index " <<a << " " << b << " " << c << std::endl;
            if (constr_type == "iuv") {
                // this is a iuv constraint
                auto i = a;
                auto u = b;
                auto v = c;
                GRBLinExpr lhs = 0;
                for (int j = 0; j < n; ++j) {
                    if (j == i) continue;
                    lhs += y_verify[std::make_tuple(i,u,j,v)];
                }
                verify_model.addConstr(lhs == x_sol[std::make_pair(i,u)], "iuv_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v));
            } else if (constr_type == "iuj") {
                // this is a iuj constraint
                auto i = a;
                auto u = b;
                auto j = c;
                GRBLinExpr lhs = 0;
                for (int v = 0; v < n; ++v) {
                    if (v == u) continue;
                    // if the corresponding iuv constraint is not in the IIS, we also skip it since its dual is zero and does not contribute to the cut verification
                    lhs += y_verify[std::make_tuple(i,u,j,v)];
                }
                verify_model.addConstr(lhs == x_sol[std::make_pair(i,u)], "iuj_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j));
            } else if (constr_type == "kij") {
                // this is a kij constraint
                auto k = a;
                auto i = b;
                auto j = c;
                GRBLinExpr lhs = 0;
                for (const auto &uv : subgraphs[k].arcs) {
                    int u = uv.first;
                    int v = uv.second;
                    lhs += y_verify[std::make_tuple(i,u,j,v)];
                }
                verify_model.addConstr(lhs == e_sol[std::make_tuple(k,i,j)], "kij_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j));
            } else {
                std::cout << "Unknown constraint type in IIS: " << constr_type << std::endl;
            }

        }
        // set objective 0 since we only care about feasibility for cut verification
        GRBLinExpr obj = 0;
        verify_model.setObjective(obj, GRB_MINIMIZE);
        verify_model.set(GRB_IntParam_Method, 1);
        verify_model.set(GRB_IntParam_Presolve, 0); // disable presolve to get correct duals for infeasibility
        verify_model.set(GRB_IntParam_InfUnbdInfo, 1); // alway see unbounded info
        verify_model.set(GRB_IntParam_DualReductions, 0); // alway see unbounded info
        verify_model.optimize();
        if (verify_model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
            verify_model.computeIIS();
            //  verify if any dual Farkas of verify_model's constraints in the IIS is zero, if so, the cut is not valid
            GRBConstr* constrs = verify_model.getConstrs();
            int numConstrs = verify_model.get(GRB_IntAttr_NumConstrs);
            int numIISConstrs = 0;
            returnIIS.clear();
            for (int i = 0; i < numConstrs; i++) {
                if (constrs[i].get(GRB_IntAttr_IISConstr)) {
                    numIISConstrs++;
                    auto constraint_name = constrs[i].get(GRB_StringAttr_ConstrName);
                    // find the corresponding indices in the original subproblem model based on the constraint name
                    int a,b,c;
                    std::string constr_type;
                    if (constraint_name.find("iuv") != std::string::npos) {
                        sscanf(constraint_name.c_str(), "iuv_%d_%d_%d", &a, &b, &c);
                        constr_type = "iuv";
                    } else if (constraint_name.find("iuj") != std::string::npos) {
                        sscanf(constraint_name.c_str(), "iuj_%d_%d_%d", &a, &b, &c);
                        constr_type = "iuj";
                    } else if (constraint_name.find("kij") != std::string::npos) {
                        sscanf(constraint_name.c_str(), "kij_%d_%d_%d", &a, &b, &c);
                        constr_type = "kij";
                    } else {
                        std::cout << "Unknown constraint type in verify model IIS: " << constraint_name << std::endl;
                    }
                    returnIIS.push_back({constr_type, std::make_tuple(a,b,c)});
                    double dual = constrs[i].get(GRB_DoubleAttr_FarkasDual);
                    if (std::fabs(dual) < 1e-9) {
                        std::cout << "Cut verification failed, a constraint in the IIS has zero dual." << std::endl;
                        // return;
                    }
                }
            }

            return numIISConstrs;

        } else {
            std::cout << "Cut verification failed, the cut is not valid." << std::endl;
        }
        } catch (GRBException &e) {
            std::cerr << "Error during verifyModel: " << e.getMessage() << " (code=" << e.getErrorCode() << ")" << std::endl;
        } catch (std::exception &ex) {
            std::cerr << "Standard exception during verifyModel: " << ex.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception during verifyModel." << std::endl;
        }
        return -1;
    }

    bool verifyFarkasRay(std::map<std::string, double>& pi) {
        // verify that the cutlhs corresponds to a valid Farkas ray of the subproblem model
        // pi = alpha, beta, gamma
        //  ATpi <= 0, where A is the constraint matrix of the subproblem model, and pi is the dual vector corresponding to the constraints in the subproblem model
        //  bTpi > 0, where b is the right-hand side vector of the subproblem model
        //  if both conditions are satisfied, then the cut is valid
        //  if not, then the cut is not valid
        // check bTpi > 0
        double bTpi = 0.0;
        
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < n; ++u) {
                for (int j = 0; j < n; ++j) {
                    if (i == j) continue;
                    auto beta_iuj = pi["beta_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j)];
                    bTpi += beta_iuj * x_sol[std::make_pair(i,u)];
                }
                for (int v = 0; v < n; ++v) {
                    if (u == v) continue;
                    auto alpha_iuv = pi["alpha_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v)];
                    bTpi += alpha_iuv * x_sol[std::make_pair(i,u)];
                }
            }
        }
        for (const auto &kv : subgraphs) {
            int k = kv.first;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i == j) continue;
                    auto gamma_kij = pi["gamma_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j)];
                    bTpi += gamma_kij * e_sol[std::make_tuple(k,i,j)];
                }
            }
        }
        // check ATpi >= 0
        int count_failed = 0;
        int count_passed = 0;
        // alpha_iuv + alpha_jvu + beta_iuj + beta_jvi + gamma_kij + gamma_kji >= 0
        for (auto i = 0; i < n; ++i) {
            for (auto j = 0; j < n; ++j) {
                for (auto u = 0; u < n; ++u) {
                    for (auto v = u + 1; v < n; ++v) {
                        if (i == j) continue;
                        // make sure uv is in some subgraph, otherwise skip
                        if (uv_to_k.find(std::make_pair(u,v)) == uv_to_k.end()) {
                            std::cout << "Skipping constraint verification for i=" << i << ", j=" << j << ", u=" << u << ", v=" << v << " since arc (" << u << "," << v << ") is not in any subgraph." << std::endl;
                            continue;
                        }
                        auto k1 = uv_to_k[std::make_pair(u,v)];
                        auto k2 = uv_to_k[std::make_pair(v,u)];
                        auto alpha_iuv = pi["alpha_" + std::to_string(i)  + "_" + std::to_string(u) + "_" + std::to_string(v)];
                        auto alpha_jvu = pi["alpha_" + std::to_string(j)  + "_" + std::to_string(v) + "_" + std::to_string(u)];
                        auto beta_iuj  = pi["beta_"  + std::to_string(i)  + "_" + std::to_string(u) + "_" + std::to_string(j)];
                        auto beta_jvi  = pi["beta_"  + std::to_string(j)  + "_" + std::to_string(v) + "_" + std::to_string(i)];
                        auto gamma_kij = pi["gamma_" + std::to_string(k1) + "_" + std::to_string(i) + "_" + std::to_string(j)];
                        auto gamma_kji = pi["gamma_" + std::to_string(k2) + "_" + std::to_string(j) + "_" + std::to_string(i)];
                        double lhs = alpha_iuv + alpha_jvu + beta_iuj + beta_jvi + gamma_kij + gamma_kji;
                        // auto c_iujv = problem.D[i][j] * problem.F[u][v] + problem.D[j][i] * problem.F[v][u];
                        if (lhs < - 1e-6) {
                            count_failed++;
                        } else {
                            count_passed++;
                        }
                    }
                }
            }
        }
        if (count_failed == 0 && bTpi < 1e-6) {
            std::cout << "Passed" << std::endl;
            return true;
        } else {
            std::cout << "Failed, bTpi " << bTpi << ", failed constraints: " << count_failed << ", passed constraints: " << count_passed << std::endl;
            return false;
        }
    }

    bool findBenderCut(std::vector<std::pair<GRBVar*,double>>& cutlhs)
    {
        // measure the time taken by cut generation
        auto start_time = std::chrono::high_resolution_clock::now();
        // if subproblem_model is nullptr, create it with the same env
        if (subproblem_model == nullptr) 
        {
            subproblem_env.set(GRB_IntParam_OutputFlag, 0); // suppress subproblem output
            subproblem_env.start();
            subproblem_model = new GRBModel(subproblem_env);
            subproblem_constraints_iuv.clear();
            subproblem_constraints_iuj.clear();
            subproblem_constraints_kij.clear();
            y.clear();

            // create y_iujv >= 0, i < j, u != v, y_jviu = y_iujv
            for (auto i = 0; i < n; ++i) {
                for (auto j = i+1; j < n; ++j) {
                    for (auto u = 0; u < n; ++u) {
                        for (auto v = 0; v < n; ++v) {
                            if (u == v) continue;
                            std::string name = "y_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(u) + "_" + std::to_string(v);
                            auto y_iujv = subproblem_model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
                            y[std::make_tuple(i,u,j,v)] = y_iujv;
                            y[std::make_tuple(j,v,i,u)] = y_iujv; // symmetry
                        }
                    }
                }
            }

            // add all constraints
            // sum_{(u,v) in G_k.arcs} y_iujv = e_sol[k,i,j] for all k, i, j: i!= j
            for (const auto &kv : subgraphs) {
                int k = kv.first;
                auto G_k = kv.second;
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        if (i == j) continue;
                        GRBLinExpr lhs = 0;
                        for (const auto &uv : G_k.arcs) {
                            int u = uv.first;
                            int v = uv.second;
                            
                            lhs += y[std::make_tuple(i,u,j,v)];
                        }
                        auto constr = subproblem_model->addConstr(lhs == e_sol[std::make_tuple(k,i,j)], "kij_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j));
                        subproblem_constraints_kij.push_back({constr, std::make_tuple(k,i,j)});
                    }
                }
            }
            // sum_j y_iujv = x_sol[i,u] for all i,u,v: u!=v
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    // if (x_sol[std::make_pair(i,u)] < 1e-6) {
                    //     continue; // if x_i_u is zero in the current solution, we can skip creating these constraints since they will not contribute to the cut
                    // }
                    for (int v = 0; v < n; ++v) {
                        if (u == v) continue;
                        GRBLinExpr lhs = 0;
                        for (int j = 0; j < n; ++j) {
                            if (j == i) continue;
                            lhs += y[std::make_tuple(i,u,j,v)];
                        }
                        auto constr = subproblem_model->addConstr(lhs == x_sol[std::make_pair(i,u)], "iuv_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v));
                        subproblem_constraints_iuv.push_back({constr, std::make_tuple(i,u,v)});
                    }
                }
            }

            // sum_v y_iujv = x_sol[i,u] for all i,u,j: i!=j
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    // if (x_sol[std::make_pair(i,u)] < 1e-6) {
                    //     continue; // if x_i_u is zero in the current solution, we can skip creating these constraints since they will not contribute to the cut
                    // }
                    for (int j = 0; j < n; ++j) {
                        if (j == i) continue;
                        GRBLinExpr lhs = 0;
                        for (int v = 0; v < n; ++v) {
                            if (v == u) continue;
                            lhs += y[std::make_tuple(i,u,j,v)];
                        }
                        auto constr = subproblem_model->addConstr(lhs == x_sol[std::make_pair(i,u)], "iuj_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j));
                        subproblem_constraints_iuj.push_back({constr, std::make_tuple(i,u,j)});
                    }
                }
            }
            // set objective 0 since we only care about feasibility for cut generation
            GRBLinExpr obj = 0;
            // for (const auto &kv : y) {
            //     // get i,j,u,v from the key
            //     auto key = kv.first;
            //     int i = std::get<0>(key);
            //     int u = std::get<1>(key);
            //     int j = std::get<2>(key);
            //     int v = std::get<3>(key);
            //     obj += kv.second* (problem.D[i][j]*problem.F[u][v] + problem.D[j][i]*problem.F[v][u]);
            // }
            subproblem_model->setObjective(obj, GRB_MINIMIZE);

            // subproblem_model->update();
        } else {
            // update the rhs of constraints based on current x_sol and e_sol
            for (auto pair : subproblem_constraints_iuv) {
                auto constr = pair.first;
                auto indices = pair.second;
                // update the rhs of the constraint
                constr.set(GRB_DoubleAttr_RHS, x_sol[std::make_pair(std::get<0>(indices), std::get<1>(indices))]);
            }
            for (auto pair : subproblem_constraints_iuj) {
                auto constr = pair.first;
                auto indices = pair.second;
                // update the rhs of the constraint
                constr.set(GRB_DoubleAttr_RHS, x_sol[std::make_pair(std::get<0>(indices), std::get<1>(indices))]);
            }
            for (auto pair : subproblem_constraints_kij) {
                auto constr = pair.first;
                auto indices = pair.second;
                // update the rhs of the constraint
                constr.set(GRB_DoubleAttr_RHS, e_sol[std::make_tuple(std::get<0>(indices), std::get<1>(indices), std::get<2>(indices))]);
            }
            subproblem_model->update();
            subproblem_model->reset();
        }
        // solve the subproblem
        subproblem_model->set(GRB_IntParam_Method, 1);
        subproblem_model->set(GRB_IntParam_Presolve, 0); // disable presolve to get correct duals for infeasibility
        subproblem_model->set(GRB_IntParam_InfUnbdInfo, 1); // alway see unbounded info
        subproblem_model->set(GRB_IntParam_DualReductions, 0); // alway see unbounded info

        subproblem_model->optimize();
        std::map<std::string, double> farkas_ray;
        // if status is infeasible, we found a Benders cut
        if (subproblem_model->get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
            // auto n_coeffs_in_cut = 0;
            // auto cut_verify = 0.0;
            // obtain dualfarkas
            for (auto pair : subproblem_constraints_iuv) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto iu = std::make_pair(std::get<0>(indices), std::get<1>(indices));
                // cut_verify += coeff*x_sol[iu];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&x[iu], coeff});
                    // std::cout << "alpha coeff for iuv " << iu.first << "," << iu.second << "," << std::get<2>(indices) << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
                farkas_ray["alpha_" + std::to_string(std::get<0>(indices)) + "_" + std::to_string(std::get<1>(indices)) + "_" + std::to_string(std::get<2>(indices))] = coeff;
            }
            for (auto pair : subproblem_constraints_iuj) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto iu = std::make_pair(std::get<0>(indices), std::get<1>(indices));
                // cut_verify += coeff*x_sol[iu];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&x[iu], coeff});
                    // std::cout << "beta coeff for iuj " << iu.first << "," << iu.second << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                    
                }
                farkas_ray["beta_" + std::to_string(std::get<0>(indices)) + "_" + std::to_string(std::get<1>(indices)) + "_" + std::to_string(std::get<2>(indices))] = coeff;
            }
            for (auto pair : subproblem_constraints_kij) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto kij = std::make_tuple(std::get<0>(indices), std::get<1>(indices), std::get<2>(indices));
                // cut_verify += coeff*e_sol[kij];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&e[kij], coeff});
                    // std::cout << "gamma coeff for kij " << std::get<0>(indices) << "," << std::get<1>(indices) << "," << std::get<2>(indices) << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
                farkas_ray["gamma_" + std::to_string(std::get<0>(kij)) + "_" + std::to_string(std::get<1>(kij)) + "_" + std::to_string(std::get<2>(kij))] = coeff;
            }
            // if (uv_to_k.empty()) {
            //     setup_uv_to_k();
            //     std::cout << "uv_to_k mapping set up." << std::endl;
            // } else {
            //     // check that every (u,v) is in exactly one subgraph
            //     for (int u =0, v = 0; u < n; ++u) {
            //         for (v = 0; v < n; ++v) {
            //             if (u == v) continue;
            //             if (uv_to_k.find(std::make_pair(u,v)) == uv_to_k.end()) {
            //                 std::cerr << "Error: arc (" << u << "," << v << ") does not appear in any subgraph, which is not supported by the current implementation." << std::endl;
            //             }
            //         }
            //     }
            // }
            // verifyFarkasRay(farkas_ray);
            // measure the time taken by cut generation
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            if (duration > 1000) {
                std::cout << "Cut generation time: " << duration*0.001 << " s." <<std::endl;
            }
            n_callbacks++;
            callback_time += duration;
            // if (duration > 1000 || cut_verify > 0) {
            //     std::cout << "verification value: " << cut_verify << ", time: " << duration*0.001 << " s." <<std::endl;
            // }
            // compute IIS and print the number of constraints in the IIS as an estimate of cut complexity

            return true;
        } else {
            std::cout << "1 Subproblem status: " << subproblem_model->get(GRB_IntAttr_Status) << std::endl;
        }
        return false;
    }

    void set_up_volume_problem(VOL_problem& vol_problem) {
        int K = subgraphs.size();
        vol_problem.psize = n*n*n*n;
        vol_problem.dsize = 2*n*n*n + K*n*n;
        
        // Set dual bounds (all free variables)
        vol_problem.dual_lb.allocate(vol_problem.dsize);
        vol_problem.dual_ub.allocate(vol_problem.dsize);
        vol_problem.dual_lb = -DBL_MAX;
        vol_problem.dual_ub = DBL_MAX;
        
        // Initialize dual solution to zero
        vol_problem.dsol.allocate(vol_problem.dsize);
        // vol_problem.dsol = 0.0;

        // // create hooks
        // if (volume_hooks == nullptr) {
        //     volume_hooks = new SFDVolumeHooks(problem, subgraphs);
        // }

        // Set Volume algorithm parameters (from qap.par defaults)
        vol_problem.parm.lambdainit = 0.01;
        vol_problem.parm.alphainit = 0.01;
        vol_problem.parm.alphamin = 0.001;
        vol_problem.parm.alphafactor = 0.66;
        vol_problem.parm.alphaint = 50;
        
        vol_problem.parm.maxsgriters = 10000000000; // effectively no limit
        vol_problem.parm.primal_abs_precision = 0.001;
        vol_problem.parm.gap_abs_precision = 0.0;
        vol_problem.parm.gap_rel_precision = 0.001;
        vol_problem.parm.granularity = 0.0;
        
        vol_problem.parm.ascent_first_check = 500;
        vol_problem.parm.ascent_check_invl = 500;
        vol_problem.parm.minimum_rel_ascent = 0.0001;
        
        vol_problem.parm.greentestinvl = 0;
        vol_problem.parm.yellowtestinvl = 4;
        vol_problem.parm.redtestinvl = 20;
        
        // Printing control
        vol_problem.parm.printflag = 0;  // 1=iteration info, 3=add lambda info
        vol_problem.parm.printinvl = 100;
        vol_problem.parm.heurinvl = 100;
    }

    bool findBenderCut6(std::vector<std::pair<GRBVar*,double>>& cutlhs)
    {
        if(uv_to_k.empty()) {
            setup_uv_to_k();
        }
        std::cout << "findBenderCut called, current solution value: " << curObjVal << std::endl;
        VOL_problem vol_problem;
        set_up_volume_problem(vol_problem);

        SFDVolumeHooks* volume_hooks = nullptr;
        volume_hooks = new SFDVolumeHooks(problem, subgraphs);
        volume_hooks->UB = curObjVal*4;
        // vol_problem.dsol = 0.0; // reset dual solution
        auto start_time = std::chrono::high_resolution_clock::now();
        // set x_sol and e_sol in volume_hooks
        for (const auto& kv : x_sol) {
            auto iu = kv.first;
            auto val = kv.second;
            auto i = iu.first;
            auto u = iu.second;
            volume_hooks->x_sol[i * n + u] = val;
        }
        for (const auto& kv : e_sol) {
            auto kij = kv.first;
            auto val = kv.second;
            auto k = std::get<0>(kij);
            auto i = std::get<1>(kij);
            auto j = std::get<2>(kij);
            volume_hooks->e_sol[k * n * n + i * n + j] = val;
        }
        
        // solve the volume problem to get dual solution

        int retval =vol_problem.solve(*volume_hooks, false);
        if(retval != 0 || vol_problem.value < curObjVal*2) {
            std::cout << "Volume problem failed to solve, return value: " << retval << ", objective value: " << vol_problem.value << std::endl;
            return false;
        }
        
        auto farkas_ray = volume_hooks->farkas_ray;
        verifyFarkasRay(farkas_ray);
        auto verified_val = 0.0;
        for (const auto& kv : x_sol) {
            auto iu = kv.first;
            auto val = 0.0;
            auto i = iu.first;
            auto u = iu.second;
            for (int v = 0; v < n; ++v) {
                if (v != u) {
                    val += farkas_ray["alpha_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v)];
                }
                if (v != i) {
                    val += farkas_ray["beta_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v)];
                }
            }
            verified_val += val * x_sol[iu];
            cutlhs.push_back({&x[iu], -val});
        }
        for (const auto& kv : e_sol) {
            auto kij = kv.first;
            auto val = 0.0;
            auto k = std::get<0>(kij);
            auto i = std::get<1>(kij);
            auto j = std::get<2>(kij);
            if (i == j) continue;
            val = farkas_ray["gamma_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j)];
            cutlhs.push_back({&e[kij], -val});
            verified_val += val * e_sol[kij];
        }
        std::cout << "Verification value from dual solution: " << verified_val << std::endl;
        delete volume_hooks;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // if (duration > 1000) 
        {
            std::cout << "Cut generation time: " << duration*0.001 << " s." <<std::endl;
        }
        
        return true;
    }

    bool findBenderCut5(std::vector<std::pair<GRBVar*,double>>& cutlhs)
    {
        if(uv_to_k.empty()) {
            subproblem_env.set(GRB_IntParam_OutputFlag, 0); // suppress subproblem output
            subproblem_env.start();
            setup_uv_to_k();
        }
        // measure the time taken by cut generation
        auto start_time = std::chrono::high_resolution_clock::now();
        // get all iu that x_sol[iu] > 1e-6, and only create constraints related to those iu to reduce the size of the subproblem model for faster cut generation, since other constraints with zero x_sol[iu] will not contribute to the cut
        std::set<std::pair<int,int>> active_iu;
        for (const auto& kv : x_sol) {
            if (kv.second > 1e-6) {
                active_iu.insert(kv.first);
            }
        }
        int ii, uu, jj, vv, kk;
        bool found_tuple = false;
        // find one tuple i,u,j,v such that 
        // x_sol[i,u], x_sol[j,v] > 1e-6 and e_sol[uv_to_k[std::make_pair(u,v)],i,j] = 0 or e_sol[uv_to_k[std::make_pair(v,u)],j,i] = 0, and only create constraints related to those tuples to reduce the size of the subproblem model for faster cut generation, since other constraints with zero x_sol[iu] or e_sol[kij] will not contribute to the cut
        for (const auto& pair1 : active_iu) {
            int i = pair1.first;
            int u = pair1.second;
            for (const auto& pair2 : active_iu) {
                int j = pair2.first;
                int v = pair2.second;
                if (i == j || u == v) continue;
                if (e_sol[std::make_tuple(uv_to_k[std::make_pair(u,v)],i,j)] < 1e-6 || e_sol[std::make_tuple(uv_to_k[std::make_pair(v,u)],j,i)] < 1e-6) {
                    // found such tuple
                    ii = i; uu = u; jj = j; vv = v;
                    auto k1 = uv_to_k[std::make_pair(u,v)];
                    auto k2 = uv_to_k[std::make_pair(v,u)];
                    kk = (e_sol[std::make_tuple(k1,i,j)] < 1e-6) ? k1 : k2;
                    found_tuple = true;
                    break;
                }
            }
            if (found_tuple) break;
        }

        GRBModel* subproblem_model = new GRBModel(subproblem_env);
        std::vector<std::pair<GRBConstr, std::tuple<int,int,int>>> subproblem_constraints_iuv;
        std::vector<std::pair<GRBConstr, std::tuple<int,int,int>>> subproblem_constraints_iuj;
        std::vector<std::pair<GRBConstr, std::tuple<int,int,int>>> subproblem_constraints_kij;
        std::map<std::tuple<int,int,int,int>, GRBVar> y;
        
        std::vector<std::pair<int,int>> iu_in_constraints;
        iu_in_constraints.push_back(std::make_pair(ii,uu));
        iu_in_constraints.push_back(std::make_pair(jj,vv));

        // create y_iujv >= 0, y_jviu = y_iujv, i = ii, j =jj, (u,v) \in A_kk
        for (auto uv : subgraphs[kk].arcs) {
            int u = uv.first;
            int v = uv.second;
            if (v == u) continue;
            std::string name = "y_" + std::to_string(ii) + "_" + std::to_string(jj) + "_" + std::to_string(u) + "_" + std::to_string(v);
            auto y_iujv = subproblem_model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
            y[std::make_tuple(ii,u,jj,v)] = y_iujv;
            y[std::make_tuple(jj,v,ii,u)] = y_iujv; // symmetry
        }
        for (auto j = 0; j < n; ++j) {
            if (j == ii) continue;
            for (auto v = 0; v < n; ++v) {
                if (v == uu) continue;
                std::string name = "y_" + std::to_string(ii) + "_" + std::to_string(j) + "_" + std::to_string(uu) + "_" + std::to_string(v);
                auto y_iujv = subproblem_model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
                y[std::make_tuple(ii,uu,j,v)] = y_iujv;
                y[std::make_tuple(j,v,ii,uu)] = y_iujv; // symmetry
                std::string name2 = "y_" + std::to_string(j) + "_" + std::to_string(ii) + "_" + std::to_string(v) + "_" + std::to_string(uu);
                auto y_jiuv2 = subproblem_model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name2);
                y[std::make_tuple(j,v,ii,uu)] = y_jiuv2;
                y[std::make_tuple(ii,uu,j,v)] = y_jiuv2; // symmetry
            }
        }

        // add all constraints
        // sum_{(u,v) in G_k.arcs} y_iujv = e_sol[k,i,j] for all k, i, j: i!= j
        
        GRBLinExpr lhs = 0;
        auto G_k = subgraphs[kk];
        for (const auto &uv : G_k.arcs) {
            int u = uv.first;
            int v = uv.second;
            lhs += y[std::make_tuple(ii,u,jj,v)];
        }
        auto constr = subproblem_model->addConstr(lhs == e_sol[std::make_tuple(kk,ii,jj)], "kij_" + std::to_string(kk) + "_" + std::to_string(ii) + "_" + std::to_string(jj));
        subproblem_constraints_kij.push_back({constr, std::make_tuple(kk,ii,jj)});
        // sum_j y_iujv = x_sol[i,u] for all i,u,v: u!=v

        for (auto pair : iu_in_constraints) {
            auto i = pair.first;
            auto u = pair.second;
            for (int v = 0; v < n; ++v) {
                if (u == v) continue;
                GRBLinExpr lhs = 0;
                for (int j = 0; j < n; ++j) {
                    if (j == i) continue;
                    lhs += y[std::make_tuple(i,u,j,v)];
                }
                auto constr = subproblem_model->addConstr(lhs == x_sol[std::make_pair(i,u)], "iuv_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v));
                subproblem_constraints_iuv.push_back({constr, std::make_tuple(i,u,v)});
            }
        }

        // sum_v y_iujv = x_sol[i,u] for all i,u,j: i!=j
        for (auto pair : iu_in_constraints) {
            auto i = pair.first;
            auto u = pair.second;
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                GRBLinExpr lhs = 0;
                for (int v = 0; v < n; ++v) {
                    if (v == u) continue;
                    // if (x_sol[std::make_pair(j,v)] < 1e-6) {
                    //     continue;
                    // }
                    lhs += y[std::make_tuple(i,u,j,v)];
                }
                auto constr = subproblem_model->addConstr(lhs == x_sol[std::make_pair(i,u)], "iuj_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j));
                subproblem_constraints_iuj.push_back({constr, std::make_tuple(i,u,j)});
            }
        }
        // set objective 0 since we only care about feasibility for cut generation
        GRBLinExpr obj = 0;
        for (const auto &kv : y) {
            // get i,j,u,v from the key
            auto key = kv.first;
            int i = std::get<0>(key);
            int u = std::get<1>(key);
            int j = std::get<2>(key);
            int v = std::get<3>(key);
            obj += kv.second* (problem.D[i][j]*problem.F[u][v] + problem.D[j][i]*problem.F[v][u]);
        }
        subproblem_model->setObjective(obj, GRB_MINIMIZE);

        // solve the subproblem
        subproblem_model->set(GRB_IntParam_Method, 1);
        subproblem_model->set(GRB_IntParam_Presolve, 0); // disable presolve to get correct duals for infeasibility
        subproblem_model->set(GRB_IntParam_InfUnbdInfo, 1); // alway see unbounded info
        subproblem_model->set(GRB_IntParam_DualReductions, 0); // alway see unbounded info

        subproblem_model->optimize();
        // if status is infeasible, we found a Benders cut
        if (subproblem_model->get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
            // auto n_coeffs_in_cut = 0;
            auto cut_verify = 0.0;
            // obtain dualfarkas
            for (auto pair : subproblem_constraints_iuv) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto iu = std::make_pair(std::get<0>(indices), std::get<1>(indices));
                cut_verify += coeff*x_sol[iu];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&x[iu], coeff});
                    // std::cout << "alpha coeff for iuv " << iu.first << "," << iu.second << "," << std::get<2>(indices) << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }
            for (auto pair : subproblem_constraints_iuj) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto ij = std::make_pair(std::get<0>(indices), std::get<1>(indices));
                cut_verify += coeff*x_sol[ij];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&x[ij], coeff});
                    // std::cout << "beta coeff for iuj " << ij.first << "," << ij.second << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }
            for (auto pair : subproblem_constraints_kij) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto kij = std::make_tuple(std::get<0>(indices), std::get<1>(indices), std::get<2>(indices));
                cut_verify += coeff*e_sol[kij];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&e[kij], coeff});
                    // std::cout << "gamma coeff for kij " << std::get<0>(indices) << "," << std::get<1>(indices) << "," << std::get<2>(indices) << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }

            // measure the time taken by cut generation
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            // if (duration > 1000) {
            //     std::cout << "Cut generation time: " << duration*0.001 << " s." <<std::endl;
            // }
            n_callbacks++;
            callback_time += duration;
            if (duration > 1000 || cut_verify > 0) {
                std::cout << "verification value: " << cut_verify << ", time: " << duration*0.001 << " s." <<std::endl;
            }
            return true;
        } else {
            std::cout << "Cut generation failed, no violated cut found." << subproblem_model->get(GRB_IntAttr_Status) << std::endl;
        }
        return false;
    }

    bool findBenderCut3(std::vector<std::pair<GRBVar*,double>>& cutlhs)
    {
        if(uv_to_k.empty()) {
            subproblem_env.set(GRB_IntParam_OutputFlag, 0); // suppress subproblem output
            subproblem_env.start();
            setup_uv_to_k();
        }
        // measure the time taken by cut generation
        auto start_time = std::chrono::high_resolution_clock::now();
        // if subproblem_model is nullptr, create it with the same env

        GRBModel* subproblem_model = new GRBModel(subproblem_env);
        std::vector<std::pair<GRBConstr, std::tuple<int,int,int>>> subproblem_constraints_iuv;
        std::vector<std::pair<GRBConstr, std::tuple<int,int,int>>> subproblem_constraints_iuj;
        std::vector<std::pair<GRBConstr, std::tuple<int,int,int>>> subproblem_constraints_kij;
        std::map<std::tuple<int,int,int,int>, GRBVar> y;
        // get all iu that x_sol[iu] > 1e-6, and only create constraints related to those iu to reduce the size of the subproblem model for faster cut generation, since other constraints with zero x_sol[iu] will not contribute to the cut
        std::set<std::pair<int,int>> active_iu;
        for (const auto& kv : x_sol) {
            if (kv.second > 1e-6) {
                active_iu.insert(kv.first);
            }
        }
        // get all k,i,j that e_sol[k,i,j] == 0 to skip creating constraints related to those kij since they also will not contribute to the cut
        std::set<std::tuple<int,int,int>> zero_kij;
        for (const auto& kv : e_sol) {
            if (kv.second < 1e-6) {
                zero_kij.insert(kv.first);
            }
        }
        // create y_iujv >= 0, i < j, u != v, y_jviu = y_iujv
        for(auto i = 0; i < n; ++i) {
            for (auto j = i + 1; j < n; ++j) {
                for (auto u = 0; u < n; ++u) {
                    for (auto v = 0; v < n; ++v) {
                        if (v == u) continue;
                        // if (x_sol[std::make_pair(i,u)] < 1e-6
                        //     || x_sol[std::make_pair(j,v)] < 1e-6
                        //  && e_sol[uv_to_k[std::make_pair(u,v)],i,j] > 1e-6 
                        //  && e_sol[uv_to_k[std::make_pair(v,u)],j,i] > 1e-6
                        // ) {
                        //     continue; // if either x_i_u or x_j_v is zero in the current solution, we can skip creating this variable since it will not contribute to the cut
                        // }
                        std::string name = "y_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(u) + "_" + std::to_string(v);
                        auto y_iujv = subproblem_model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
                        y[std::make_tuple(i,u,j,v)] = y_iujv;
                        y[std::make_tuple(j,v,i,u)] = y_iujv; // symmetry
                    }
                }
            }
        }

        // add all constraints
        // sum_{(u,v) in G_k.arcs} y_iujv = e_sol[k,i,j] for all k, i, j: i!= j
        
        for (auto key : zero_kij) {
            int k = std::get<0>(key);
            int i = std::get<1>(key);
            int j = std::get<2>(key);
            if (i == j) continue;
            GRBLinExpr lhs = 0;
            auto G_k = subgraphs[k];
            for (const auto &uv : G_k.arcs) {
                int u = uv.first;
                int v = uv.second;
                // if (x_sol[std::make_pair(i,u)] < 1e-6 || x_sol[std::make_pair(j,v)] < 1e-6) {
                //     continue; // if either x_i_u or x_j_v is zero in the current solution, we can skip creating this variable and constraint since it will not contribute to the cut
                // }
                lhs += y[std::make_tuple(i,u,j,v)];
            }
            auto constr = subproblem_model->addConstr(lhs == e_sol[std::make_tuple(k,i,j)], "kij_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j));
            subproblem_constraints_kij.push_back({constr, std::make_tuple(k,i,j)});
        }
        // sum_j y_iujv = x_sol[i,u] for all i,u,v: u!=v
        for (auto pair : active_iu) {
            auto i = pair.first;
            auto u = pair.second;
            for (int v = 0; v < n; ++v) {
                if (u == v) continue;
                GRBLinExpr lhs = 0;
                for (int j = 0; j < n; ++j) {
                    if (j == i) continue;
                    // if (x_sol[std::make_pair(j,v)] < 1e-6) {
                    //     continue;
                    // }
                    lhs += y[std::make_tuple(i,u,j,v)];
                }
                auto constr = subproblem_model->addConstr(lhs == x_sol[std::make_pair(i,u)], "iuv_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v));
                subproblem_constraints_iuv.push_back({constr, std::make_tuple(i,u,v)});
            }
        }

        // sum_v y_iujv = x_sol[i,u] for all i,u,j: i!=j
        for (auto pair : active_iu) {
            auto i = pair.first;
            auto u = pair.second;
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                GRBLinExpr lhs = 0;
                for (int v = 0; v < n; ++v) {
                    if (v == u) continue;
                    // if (x_sol[std::make_pair(j,v)] < 1e-6) {
                    //     continue;
                    // }
                    lhs += y[std::make_tuple(i,u,j,v)];
                }
                auto constr = subproblem_model->addConstr(lhs == x_sol[std::make_pair(i,u)], "iuj_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j));
                subproblem_constraints_iuj.push_back({constr, std::make_tuple(i,u,j)});
            }
        }
        // set objective 0 since we only care about feasibility for cut generation
        GRBLinExpr obj = 0;
        for (const auto &kv : y) {
            // get i,j,u,v from the key
            auto key = kv.first;
            int i = std::get<0>(key);
            int u = std::get<1>(key);
            int j = std::get<2>(key);
            int v = std::get<3>(key);
            obj += kv.second* (problem.D[i][j]*problem.F[u][v] + problem.D[j][i]*problem.F[v][u]);
        }
        subproblem_model->setObjective(obj, GRB_MINIMIZE);

        // solve the subproblem
        subproblem_model->set(GRB_IntParam_Method, 1);
        subproblem_model->set(GRB_IntParam_Presolve, 0); // disable presolve to get correct duals for infeasibility
        subproblem_model->set(GRB_IntParam_InfUnbdInfo, 1); // alway see unbounded info
        subproblem_model->set(GRB_IntParam_DualReductions, 0); // alway see unbounded info

        subproblem_model->optimize();
        // if status is infeasible, we found a Benders cut
        if (subproblem_model->get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
            // auto n_coeffs_in_cut = 0;
            auto cut_verify = 0.0;
            // obtain dualfarkas
            for (auto pair : subproblem_constraints_iuv) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto iu = std::make_pair(std::get<0>(indices), std::get<1>(indices));
                cut_verify += coeff*x_sol[iu];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&x[iu], coeff});
                    // std::cout << "alpha coeff for iuv " << iu.first << "," << iu.second << "," << std::get<2>(indices) << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }
            for (auto pair : subproblem_constraints_iuj) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto ij = std::make_pair(std::get<0>(indices), std::get<1>(indices));
                cut_verify += coeff*x_sol[ij];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&x[ij], coeff});
                    // std::cout << "beta coeff for iuj " << ij.first << "," << ij.second << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }
            for (auto pair : subproblem_constraints_kij) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto kij = std::make_tuple(std::get<0>(indices), std::get<1>(indices), std::get<2>(indices));
                cut_verify += coeff*e_sol[kij];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&e[kij], coeff});
                    // std::cout << "gamma coeff for kij " << std::get<0>(indices) << "," << std::get<1>(indices) << "," << std::get<2>(indices) << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }

            // measure the time taken by cut generation
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            // if (duration > 1000) {
            //     std::cout << "Cut generation time: " << duration*0.001 << " s." <<std::endl;
            // }
            n_callbacks++;
            callback_time += duration;
            if (duration > 1000 || cut_verify > 0) {
                std::cout << "verification value: " << cut_verify << ", time: " << duration*0.001 << " s." <<std::endl;
            }
            return true;
        } else {
            std::cout << "Cut generation failed, no violated cut found." << subproblem_model->get(GRB_IntAttr_Status) << std::endl;
        }
        return false;
    }

    bool findBenderCut2(std::vector<std::pair<GRBVar*,double>>& cutlhs)
    {
        // measure the time taken by cut generation
        auto start_time = std::chrono::high_resolution_clock::now();
        // if subproblem_model is nullptr, create it with the same env
        if (subproblem_model == nullptr) 
        {
            subproblem_env.set(GRB_IntParam_OutputFlag, 0); // suppress subproblem output
            subproblem_env.start();
            subproblem_model = new GRBModel(subproblem_env);
            subproblem_constraints_iuv.clear();
            subproblem_constraints_iuj.clear();
            subproblem_constraints_kij.clear();
            y.clear();

            // create y_iujv >= 0, i < j, u != v, y_jviu = y_iujv
            for (auto i = 0; i < n; ++i) {
                for (auto j = i; j < n; ++j) {
                    for (auto u = 0; u < n; ++u) {
                        for (auto v = 0; v < n; ++v) {
                            if (u == v) continue;
                            std::string name = "y_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(u) + "_" + std::to_string(v);
                            auto y_iujv = subproblem_model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
                            y[std::make_tuple(i,u,j,v)] = y_iujv;
                            y[std::make_tuple(j,v,i,u)] = y_iujv; // symmetry
                        }
                    }
                }
            }

            GRBLinExpr obj = 0;
            // add all constraints
            // sum_{(u,v) in G_k.arcs} y_iujv = e_sol[k,i,j] for all k, i, j: i!= j
            for (const auto &kv : subgraphs) {
                int k = kv.first;
                auto G_k = kv.second;
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        if (i == j) continue;
                        GRBLinExpr lhs = 0;
                        for (const auto &uv : G_k.arcs) {
                            int u = uv.first;
                            int v = uv.second;
                            
                            lhs += y[std::make_tuple(i,u,j,v)];
                            obj += y[std::make_tuple(i,u,j,v)];
                        }
                        auto constr = subproblem_model->addConstr(lhs == e_sol[std::make_tuple(k,i,j)], "kij_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j));
                        subproblem_constraints_kij.push_back({constr, std::make_tuple(k,i,j)});
                    }
                }
            }
            // sum_j y_iujv = x_sol[i,u] for all i,u,v: u!=v
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    // if (x_sol[std::make_pair(i,u)] < 1e-6) {
                    //     continue; // if x_i_u is zero in the current solution, we can skip creating these constraints since they will not contribute to the cut
                    // }
                    for (int v = 0; v < n; ++v) {
                        if (u == v) continue;
                        GRBLinExpr lhs = 0;
                        for (int j = 0; j < n; ++j) {
                            if (j == i) continue;
                            lhs += y[std::make_tuple(i,u,j,v)];
                        }
                        auto constr = subproblem_model->addConstr(lhs == x_sol[std::make_pair(i,u)], "iuv_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v));
                        subproblem_constraints_iuv.push_back({constr, std::make_tuple(i,u,v)});
                    }
                }
            }

            // sum_v y_iujv = x_sol[i,u] for all i,u,j: i!=j
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    // if (x_sol[std::make_pair(i,u)] < 1e-6) {
                    //     continue; // if x_i_u is zero in the current solution, we can skip creating these constraints since they will not contribute to the cut
                    // }
                    for (int j = 0; j < n; ++j) {
                        if (j == i) continue;
                        GRBLinExpr lhs = 0;
                        for (int v = 0; v < n; ++v) {
                            if (v == u) continue;
                            lhs += y[std::make_tuple(i,u,j,v)];
                        }
                        auto constr = subproblem_model->addConstr(lhs == x_sol[std::make_pair(i,u)], "iuj_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j));
                        subproblem_constraints_iuj.push_back({constr, std::make_tuple(i,u,j)});
                    }
                }
            }
            GRBLinExpr lhs = 0;
            for (const auto &kv : y) {
                // get i,j,u,v from the key
                auto key = kv.first;
                int i = std::get<0>(key);
                int u = std::get<1>(key);
                int j = std::get<2>(key);
                int v = std::get<3>(key);
                lhs += kv.second* problem.D[i][j]*problem.F[u][v];
            }
            // sum_{i,j,u,v} (D[i][j]*F[u][v] + D[j][i]*F[v][u]) * y_iujv <= Obj
            obj_constr = subproblem_model->addConstr(lhs == z_sol, "obj_constr");
            // set objective 0 since we only care about feasibility for cut generation
            subproblem_model->setObjective(obj, GRB_MINIMIZE);

            // subproblem_model->update();
        } else {
            // update the rhs of constraints based on current x_sol and e_sol
            for (auto pair : subproblem_constraints_iuv) {
                auto constr = pair.first;
                auto indices = pair.second;
                // update the rhs of the constraint
                constr.set(GRB_DoubleAttr_RHS, x_sol[std::make_pair(std::get<0>(indices), std::get<1>(indices))]);
            }
            for (auto pair : subproblem_constraints_iuj) {
                auto constr = pair.first;
                auto indices = pair.second;
                // update the rhs of the constraint
                constr.set(GRB_DoubleAttr_RHS, x_sol[std::make_pair(std::get<0>(indices), std::get<1>(indices))]);
            }
            for (auto pair : subproblem_constraints_kij) {
                auto constr = pair.first;
                auto indices = pair.second;
                // update the rhs of the constraint
                constr.set(GRB_DoubleAttr_RHS, e_sol[std::make_tuple(std::get<0>(indices), std::get<1>(indices), std::get<2>(indices))]);
            }
            obj_constr.set(GRB_DoubleAttr_RHS, z_sol);


            subproblem_model->update();
            subproblem_model->reset();
        }
        // solve the subproblem
        subproblem_model->set(GRB_IntParam_Method, 1);
        subproblem_model->set(GRB_IntParam_Presolve, 0); // disable presolve to get correct duals for infeasibility
        subproblem_model->set(GRB_IntParam_InfUnbdInfo, 1); // alway see unbounded info
        subproblem_model->set(GRB_IntParam_DualReductions, 0); // alway see unbounded info

        subproblem_model->optimize();
        // if status is infeasible, we found a Benders cut
        if (subproblem_model->get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
            // auto n_coeffs_in_cut = 0;
            // auto cut_verify = 0.0;
            // obtain dualfarkas
            for (auto pair : subproblem_constraints_iuv) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto iu = std::make_pair(std::get<0>(indices), std::get<1>(indices));
                // cut_verify += coeff*x_sol[iu];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&x[iu], coeff});
                    // std::cout << "alpha coeff for iuv " << iu.first << "," << iu.second << "," << std::get<2>(indices) << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }
            for (auto pair : subproblem_constraints_iuj) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto ij = std::make_pair(std::get<0>(indices), std::get<1>(indices));
                // cut_verify += coeff*x_sol[ij];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&x[ij], coeff});
                    // std::cout << "beta coeff for iuj " << ij.first << "," << ij.second << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }
            for (auto pair : subproblem_constraints_kij) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_FarkasDual);
                auto kij = std::make_tuple(std::get<0>(indices), std::get<1>(indices), std::get<2>(indices));
                // cut_verify += coeff*e_sol[kij];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&e[kij], coeff});
                    // std::cout << "gamma coeff for kij " << std::get<0>(indices) << "," << std::get<1>(indices) << "," << std::get<2>(indices) << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }
            // std::cout << "z_solution: " << z_sol << std::endl;
            // std::cout << "Farkas dual for obj_constr: " << obj_constr.get(GRB_DoubleAttr_FarkasDual) << std::endl;
            cutlhs.push_back({&z_var, obj_constr.get(GRB_DoubleAttr_FarkasDual)});

            // measure the time taken by cut generation
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            if (duration > 1000) {
                std::cout << "Cut generation time: " << duration*0.001 << " s." <<std::endl;
            }
            n_callbacks++;
            callback_time += duration;
            // if (duration > 1000 || cut_verify > 0) {
            //     std::cout << "verification value: " << cut_verify << ", time: " << duration*0.001 << " s." <<std::endl;
            // }
            // compute IIS and print the number of constraints in the IIS as an estimate of cut complexity
            return true;
        } else if (subproblem_model->get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            for (auto pair : subproblem_constraints_iuv) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_Pi);
                auto iu = std::make_pair(std::get<0>(indices), std::get<1>(indices));
                // cut_verify += coeff*x_sol[iu];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&x[iu], coeff});
                    // std::cout << "alpha coeff for iuv " << iu.first << "," << iu.second << "," << std::get<2>(indices) << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }
            for (auto pair : subproblem_constraints_iuj) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_Pi);
                auto ij = std::make_pair(std::get<0>(indices), std::get<1>(indices));
                // cut_verify += coeff*x_sol[ij];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&x[ij], coeff});
                    // std::cout << "beta coeff for iuj " << ij.first << "," << ij.second << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }
            for (auto pair : subproblem_constraints_kij) {
                auto constr = pair.first;
                auto indices = pair.second;
                double coeff = constr.get(GRB_DoubleAttr_Pi);
                auto kij = std::make_tuple(std::get<0>(indices), std::get<1>(indices), std::get<2>(indices));
                // cut_verify += coeff*e_sol[kij];
                if (std::fabs(coeff) > 1e-9) {
                    cutlhs.push_back({&e[kij], coeff});
                    // std::cout << "gamma coeff for kij " << std::get<0>(indices) << "," << std::get<1>(indices) << "," << std::get<2>(indices) << ": " << coeff << std::endl;
                    // n_coeffs_in_cut++;
                }
            }
            // std::cout << "z_solution: " << z_sol << std::endl;
            // std::cout << "Farkas dual for obj_constr: " << obj_constr.get(GRB_DoubleAttr_FarkasDual) << std::endl;
            cutlhs.push_back({&z_var, obj_constr.get(GRB_DoubleAttr_Pi)});

            // measure the time taken by cut generation
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            if (duration > 1000) {
                std::cout << "Cut generation time optimal: " << duration*0.001 << " s." <<std::endl;
            }
            n_callbacks++;
            callback_time += duration;
            // if (duration > 1000 || cut_verify > 0) {
            //     std::cout << "verification value: " << cut_verify << ", time: " << duration*0.001 << " s." <<std::endl;
            // }
            // compute IIS and print the number of constraints in the IIS as an estimate of cut complexity
            return true;
            
        } else {
            std::cout << "Cut generation failed, no violated cut found." << subproblem_model->get(GRB_IntAttr_Status) << std::endl;
        }
        return false;
    }

    bool findBenderCut4(std::vector<std::pair<GRBVar*,double>>& cutlhs)
    {
        // measure the time taken by cut generation
        auto start_time = std::chrono::high_resolution_clock::now();
        // if subproblem_model is nullptr, create it with the same env
        if (subproblem_model == nullptr) 
        // if(0)
        {
            setup_uv_to_k();
            subproblem_env.set(GRB_IntParam_OutputFlag, 0); // suppress subproblem output
            subproblem_env.start();
            subproblem_model = new GRBModel(subproblem_env);
            subproblem_constraints_iuv.clear();
            subproblem_constraints_iuj.clear();
            subproblem_constraints_kij.clear();
            alpha.clear();
            beta.clear();
            gamma.clear();
            
            // create y_iujv >= 0, i < j, u != v, y_jviu = y_iujv
            for (auto i = 0; i < n; ++i) {
                for (auto u = 0; u < n; ++u) {
                    // if (x_sol[std::make_pair(i,u)] < 1e-6) continue; // only create variables for positive x_sol to save time
                    for (auto v = 0; v < n; ++v) {
                        if (u == v) continue;
                        std::string name = "alpha_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v);
                        auto alpha_iuv = subproblem_model->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
                        alpha[std::make_tuple(i,u,v)] = alpha_iuv;
                    }
                    for (auto j = 0; j < n; ++j) {
                        if (j == i) continue;
                        std::string name = "beta_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j);
                        auto beta_iuj = subproblem_model->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
                        beta[std::make_tuple(i,u,j)] = beta_iuj;
                    }
                }
            }
            for (auto kv: subgraphs) {
                int k = kv.first;
                for (auto i = 0; i < n; ++i) {
                    for (auto j = 0; j < n; ++j) {
                        if (j == i) continue;
                        std::string name = "gamma_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j);
                        auto gamma_kij = subproblem_model->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
                        gamma[std::make_tuple(k,i,j)] = gamma_kij;
                    }
                }
            }

            // add all constraints
            // alpha_iuv + alpha_jvu + beta_iuj + beta_jvi + sum_{k: (u,v) in G_k} gamma_kij = problem.D[i][j]*problem.F[u][v] + problem.D[j][i]*problem.F[v][u] for all i,j,u,v: i<j, u!=v
            for (int i = 0; i < n; ++i) {
                for (int j = i+1; j < n; ++j) {
                    for (int u = 0; u < n; ++u) {
                        // if (x_sol[std::make_pair(i,u)] < 1e-6) continue; // only create constraints for positive x_sol to save time
                        for (int v = 0; v < n; ++v) {
                            // if (x_sol[std::make_pair(j,v)] < 1e-6) continue; // only create constraints for positive x_sol to save time
                            if (u == v) continue;
                            GRBLinExpr lhs = alpha[std::make_tuple(i,u,v)] + alpha[std::make_tuple(j,v,u)] + beta[std::make_tuple(i,u,j)] + beta[std::make_tuple(j,v,i)];
                            for (auto kv: subgraphs) {
                                int k = kv.first;
                                auto G_k = kv.second;
                                if (std::find(G_k.arcs.begin(), G_k.arcs.end(), std::make_pair(u,v)) != G_k.arcs.end()) {
                                    lhs += gamma[std::make_tuple(k,i,j)];
                                }
                                if (std::find(G_k.arcs.begin(), G_k.arcs.end(), std::make_pair(v,u)) != G_k.arcs.end()) {
                                    lhs += gamma[std::make_tuple(k,j,i)];
                                }
                            }
                            auto constr = subproblem_model->addConstr(lhs <= problem.D[i][j]*problem.F[u][v] + problem.D[j][i]*problem.F[v][u], "subprob_i" + std::to_string(i) + "_j" + std::to_string(j) + "_u" + std::to_string(u) + "_v" + std::to_string(v));
                        }
                    }
                }
            }

            // GRBLinExpr lhs = 0;
            // for (const auto &kv : alpha) {
            //     lhs+= kv.second;
            // }
            // for (const auto &kv : beta) {
            //     lhs+= kv.second;
            // }
            // for (const auto &kv : gamma) {
            //     lhs+= kv.second;
            // }
            // subproblem_model->addConstr(lhs == 0, "obj_bound"); // to avoid unboundedness
            // subproblem_model->update();

            // solve the subproblem
            subproblem_model->set(GRB_IntParam_Method, 0);
            // subproblem_model->set(GRB_IntParam_Presolve, 0); // disable presolve to get correct duals for infeasibility
            subproblem_model->set(GRB_IntParam_InfUnbdInfo, 1); // alway see unbounded info
            // subproblem_model->set(GRB_IntParam_DualReductions, 0); // alway see unbounded info

        } 
        // if(0)
        {
            std::vector<std::pair<int,int>> positive_x;
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    if (x_sol[std::make_pair(i,u)] > 1e-6) {
                        positive_x.push_back(std::make_pair(i,u));
                    }
                }
            }

            subproblem_env.set(GRB_IntParam_OutputFlag, 0); // suppress subproblem output
            subproblem_env.start();
            subproblem_model = new GRBModel(subproblem_env);
            alpha.clear();
            beta.clear();
            gamma.clear();
            
            // create y_iujv >= 0, i < j, u != v, y_jviu = y_iujv
            for (auto pair : positive_x) {
                int i = pair.first;
                int u = pair.second;
                // if (x_sol[std::make_pair(i,u)] < 1e-6) continue; // only create variables for positive x_sol to save time
                for (auto v = 0; v < n; ++v) {
                    if (u == v) continue;
                    std::string name = "alpha_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(v);
                    auto alpha_iuv = subproblem_model->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
                    alpha[std::make_tuple(i,u,v)] = alpha_iuv;
                }
                for (auto j = 0; j < n; ++j) {
                    if (j == i) continue;
                    std::string name = "beta_" + std::to_string(i) + "_" + std::to_string(u) + "_" + std::to_string(j);
                    auto beta_iuj = subproblem_model->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
                    beta[std::make_tuple(i,u,j)] = beta_iuj;
                }
            }
            for (auto kv: subgraphs) {
                int k = kv.first;
                for (auto i = 0; i < n; ++i) {
                    for (auto j = 0; j < n; ++j) {
                        if (j == i) continue;
                        if(e_sol[std::make_tuple(k,i,j)] > 1e-6) continue; 
                        std::string name = "gamma_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j);
                        auto gamma_kij = subproblem_model->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
                        gamma[std::make_tuple(k,i,j)] = gamma_kij;
                    }
                }
            }

            // add all constraints
            // alpha_iuv + alpha_jvu + beta_iuj + beta_jvi + sum_{k: (u,v) in G_k} gamma_kij = problem.D[i][j]*problem.F[u][v] + problem.D[j][i]*problem.F[v][u] for all i,j,u,v: i<j, u!=v
            auto n_reduced_constr = 0;
            for (int i = 0; i < n; ++i) {
                for (int j = i+1; j < n; ++j) {
                    for (int u = 0; u < n; ++u) {
                        // if (x_sol[std::make_pair(i,u)] < 1e-6) continue; // only create constraints for positive x_sol to save time
                        for (int v = 0; v < n; ++v) {
                            // if (x_sol[std::make_pair(j,v)] < 1e-6) continue; // only create constraints for positive x_sol to save time
                            if (u == v) continue;
                            GRBLinExpr lhs = 0.0;
                            if (x_sol[std::make_pair(i, u)] > 1e-6) {
                                lhs = alpha[std::make_tuple(i,u,v)] + beta[std::make_tuple(i,u,j)];
                            } 
                            if (x_sol[std::make_pair(j, v)] > 1e-6) {
                                lhs += alpha[std::make_tuple(j,v,u)] + beta[std::make_tuple(j,v,i)];
                            }
                            auto k1 = uv_to_k[std::make_pair(u,v)];
                            auto k2 = uv_to_k[std::make_pair(v,u)];
                            if (e_sol[std::make_tuple(k1,i,j)] > 1e-6) {
                                lhs += gamma[std::make_tuple(k1,i,j)];
                            }
                            if (e_sol[std::make_tuple(k2,j,i)] > 1e-6) {
                                lhs += gamma[std::make_tuple(k2,j,i)];
                            }
                            if (lhs.size() > 0) {
                                auto constr = subproblem_model->addConstr(lhs <= 0.0, "subprob_i" + std::to_string(i) + "_j" + std::to_string(j) + "_u" + std::to_string(u) + "_v" + std::to_string(v));
                            } else {
                                n_reduced_constr++;
                            }
                        }
                    }
                }
            }
            std::cout << "Number of constraints in the reduced model: " << n_reduced_constr << std::endl;


            // GRBLinExpr lhs = 0;
            // for (const auto &kv : alpha) {
            //     lhs+= kv.second;
            // }
            // for (const auto &kv : beta) {
            //     lhs+= kv.second;
            // }
            // for (const auto &kv : gamma) {
            //     lhs+= kv.second;
            // }
            // subproblem_model->addConstr(lhs == 1, "obj_bound"); // to avoid unboundedness
            // subproblem_model->update();

            // solve the subproblem
            subproblem_model->set(GRB_IntParam_Method, 0);
            // subproblem_model->set(GRB_IntParam_Presolve, 0); // disable presolve to get correct duals for infeasibility
            subproblem_model->set(GRB_IntParam_InfUnbdInfo, 1); // alway see unbounded info
            // subproblem_model->set(GRB_IntParam_DualReductions, 0); // alway see unbounded info

        } 
        {
            std::vector<std::pair<int,int>> positive_x;
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    if (x_sol[std::make_pair(i,u)] > 1e-6) {
                        positive_x.push_back(std::make_pair(i,u));
                    }
                }
            }

            // obj = sum_iu x_sol[i,u]*sum_v alpha_iuv + sum_ij x_sol[i,u]*sum_j beta_iuj + sum_kij e_sol[k,i,j]*gamma_kij
            GRBLinExpr obj = 0;
            for (auto pair : positive_x) {
                int i = pair.first;
                int u = pair.second;
                GRBLinExpr alpha_sum = 0;
                for (int v = 0; v < n; ++v) {
                    if (v == u) continue;
                    alpha_sum += alpha[std::make_tuple(i,u,v)];
                }
                GRBLinExpr beta_sum = 0;
                for (int j = 0; j < n; ++j) {
                    if (j == i) continue;
                    beta_sum += beta[std::make_tuple(i,u,j)];
                }
                obj += x_sol[std::make_pair(i,u)] * (alpha_sum + beta_sum);
                
            }
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (j == i) continue;
                    for (auto kv: subgraphs) {
                        int k = kv.first;
                        if (e_sol[std::make_tuple(k,i,j)] < 1e-6) continue; // only consider positive e_sol to save time
                        obj += e_sol[std::make_tuple(k,i,j)]*gamma[std::make_tuple(k,i,j)];
                    }
                }
            }
            subproblem_model->setObjective(obj, GRB_MAXIMIZE);
        }
        // subproblem_model->reset();
        subproblem_model->optimize();
        // if status is UNBOUNDED, we found a Benders cut
        std::cout << "Subproblem status: " << subproblem_model->get(GRB_IntAttr_Status) << std::endl;
        if (subproblem_model->get(GRB_IntAttr_Status) == GRB_UNBOUNDED) 
        {
            auto cut_verify = 0.0;
            
            for (int i = 0; i < n; ++i) {
                for (int u = 0; u < n; ++u) {
                    if (x_sol[std::make_pair(i,u)] < 1e-6) continue; // only consider positive x_sol to save time
                    auto coeff = 0.0;
                    for (int v = 0; v < n; ++v) {
                        if (v == u) continue;
                        coeff += -alpha[std::make_tuple(i,u,v)].get(GRB_DoubleAttr_UnbdRay);
                    }
                    for (int j = 0; j < n; ++j) {
                        if (j == i) continue;
                        coeff += -beta[std::make_tuple(i,u,j)].get(GRB_DoubleAttr_UnbdRay);
                    }
                    cut_verify += coeff*x_sol[std::make_pair(i,u)];
                    if (std::fabs(coeff) > 1e-9) {
                        cutlhs.push_back({&x[std::make_pair(i,u)], coeff});
                    }
                }
            }
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (j == i) continue;
                    for (auto kv: subgraphs) {
                        int k = kv.first;
                        auto coeff = -gamma[std::make_tuple(k,i,j)].get(GRB_DoubleAttr_UnbdRay);
                        cut_verify += coeff*e_sol[std::make_tuple(k,i,j)];
                        if (std::fabs(coeff) > 1e-9) {
                            cutlhs.push_back({&e[std::make_tuple(k,i,j)], coeff});
                        }
                    }
                }
            }
            

            // measure the time taken by cut generation
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            // if (duration > 1000 || cut_verify > 0) 
            {
                std::cout << "verification value: " << cut_verify << ", time: " << duration*0.001 << " s." <<std::endl;
            }
            return true;
        } else {
            std::cout << "Subproblem objective: " << subproblem_model->get(GRB_DoubleAttr_ObjVal) << std::endl;
        }
        return false;
    }
};

SFDGurobiSolver::SFDGurobiSolver(const SFDGurobiConfig& cfg_) : cfg(cfg_) {}

std::map<int, SubgraphData> decomposeFlowgraphValueOnly(const Problem& problem) {
    // ensure that every (u,v) is in exactly one subgraph, and that every subgraph has a unique flow value f_k
    std::map<int, SubgraphData> subgraphs;

    std::map<double, std::vector<std::pair<int,int>>> flow_to_arcs;
    for (int u = 0; u < problem.m; ++u) {
        for (int v = 0; v < problem.m; ++v) {
            if (u == v) continue;
            double flow = problem.F[u][v];
            //  if flow is not in flow_to_arcs, it will be default-initialized to empty vector
            flow_to_arcs[flow].emplace_back(u, v);
        }
    }
    int k = 0;
    for (const auto& kv : flow_to_arcs) {
        double f_k = kv.first;
        const auto& arcs = kv.second;
        std::set<int> nodes;
        for (const auto& uv : arcs) {
            nodes.insert(uv.first);
            nodes.insert(uv.second);
        }
        subgraphs.insert({k, {f_k, arcs, nodes}});
        ++k;
    }
    // for (const auto& kv : subgraphs) {
    //     int k = kv.first;
    //     const auto& G_k = kv.second;
    //     std::cout << "Subgraph " << k << ": flow=" << G_k.f_k
    //               << ", arcs=" << G_k.arcs.size()
    //               << ", nodes=" << G_k.nodes.size() << std::endl;
    // }
    return subgraphs;
}

std::tuple< 
    GRBModel*,
    std::map<std::pair<int,int>, GRBVar>,
    std::map<std::tuple<int,int,int>, GRBVar>
    >
SFDGurobiSolver::buildModel(
    const Problem& problem,
    std::map<int, SubgraphData>& subgraphs,
    const std::unordered_map<int,int>& fixed_variables
) {
    auto n = problem.n;
    auto m = problem.m;
    auto distances = problem.D;
    auto flows = problem.F;


    std::vector<int> V(n);
    std::vector<int> M(m);

    for (int i = 0; i < n; ++i)
        V[i] = i;

    for (int u = 0; u < m; ++u)
        M[u] = u;

    // -------------------------------------------------------------
    // Create environment and model
    // -------------------------------------------------------------
    GRBEnv* env = new GRBEnv();
    GRBModel* model = new GRBModel(*env);

    model->set(GRB_StringAttr_ModelName, "QAP_SFD");

    // -------------------------------------------------------------
    // x variables
    // -------------------------------------------------------------
    std::map<std::pair<int,int>, GRBVar> x;

    for (auto i : V) {
        for (auto u : M) {
            x[std::make_pair(i, u)] = model->addVar(
                0.0,
                1.0,
                0.0,
                cfg.is_relax ? GRB_CONTINUOUS : GRB_BINARY,
                "x_" + std::to_string(i) + "_" + std::to_string(u)
            );
        }
    }
    // -------------------------------------------------------------
    // e variables (subgraph edge variables) and objective
    // -------------------------------------------------------------
    // detect symmetry of flow and distance matrices
    bool is_symmetric = true;
    for (int u = 0; u < m && is_symmetric; ++u) {
        for (int v = 0; v < m; ++v) {
            if (std::fabs(flows[u][v] - flows[v][u]) > 1e-9) { is_symmetric = false; break; }
        }
    }
    for (int i = 0; i < n && is_symmetric; ++i) {
        for (int j = 0; j < n; ++j) {
            if (std::fabs(distances[i][j] - distances[j][i]) > 1e-9) { is_symmetric = false; break; }
        }
    }

    // create reduced e vars when symmetric (only i<j), then map both directions to same var
    std::map<std::tuple<int,int,int>, GRBVar> e_temp; // temporary: only created keys
    for (const auto &kv : subgraphs) {
        int k = kv.first;
        for (int i : V) {
            for (int j : V) {
                if (i == j) continue;
                if (is_symmetric && j <= i) continue; // create only for i<j
                std::string name = "e_" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j);
                e_temp[std::make_tuple(k,i,j)] = model->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, name);
            }
        }
    }

    // final e map: if symmetric, map both (i,j) and (j,i) to same var
    std::map<std::tuple<int,int,int>, GRBVar> e;
    if (is_symmetric) {
        for (const auto &kv : e_temp) {
            int k,i,j; std::tie(k,i,j) = kv.first;
            GRBVar var = kv.second;
            e[std::make_tuple(k,i,j)] = var;
            e[std::make_tuple(k,j,i)] = var; // symmetry mapping
        }
    } else {
        e = e_temp;
    }

    // Objective as linear combination: sum_{k,i,j} D[i][j] * f_k * e_{k,i,j}
    GRBLinExpr obj = 0;
    for (const auto &kv : e) {
        int k,i,j; std::tie(k,i,j) = kv.first;
        auto f_k = this->subgraphs[k].f_k;
        // if (f_k < 1e-9) continue; // skip zero-flow subgraph
        GRBVar var = kv.second;
        obj += distances[i][j] * f_k * var;
    }
    model->setObjective(obj, GRB_MINIMIZE);

    // -------------------------------------------------------------
    // Assignment constraints
    // -------------------------------------------------------------
    for (auto i : V) {

        GRBLinExpr expr = 0;

        for (auto u : M)
            expr += x[std::make_pair(i, u)];

        model->addConstr(
            expr == 1,
            "assign_fac_" + std::to_string(i)
        );
    }

    for (auto u : M) {

        GRBLinExpr expr = 0;

        for (auto i : V)
            expr += x[std::make_pair(i, u)];

        model->addConstr(
            expr == 1,
            "assign_loc_" + std::to_string(u)
        );
    }

    // -------------------------------------------------------------
    // Subgraph constraints and flow conservation (using e variables)
    // -------------------------------------------------------------

    // sum_j e^k_ij == sum_u x_iu * degree_out(u) for all i in V, for all k
    // (and similarly for incoming if not symmetric)
    int total_subgraph_constraints = 0;
    for (const auto &kv : subgraphs) {
        int k = kv.first;
        auto G_k = kv.second;

        // compute degree sequences
        std::map<int,int> degree_out, degree_in;
        for (auto &uv : G_k.arcs) { degree_out[uv.first]++; degree_in[uv.second]++; }

        // flow conservation constraints
        for (int i : V) {
            GRBLinExpr lhs_out = 0;
            for (int j : V) if (i!=j) {
                auto key = std::make_tuple(k,i,j);
                lhs_out += e[key];
            }
            GRBLinExpr rhs_out = 0;
            for (int u : G_k.nodes) rhs_out += x[std::make_pair(i,u)] * degree_out[u];
            model->addConstr(lhs_out == rhs_out, "flow_out_" + std::to_string(i) + "_" + std::to_string(k));

            if (!is_symmetric) {
                GRBLinExpr lhs_in = 0;
                for (int j : V) if (i!=j) {
                    auto key = std::make_tuple(k,j,i);
                    lhs_in += e[key];
                }
                GRBLinExpr rhs_in = 0;
                for (int u : G_k.nodes) rhs_in += x[std::make_pair(i,u)] * degree_in[u];
                model->addConstr(lhs_in == rhs_in, "flow_in_" + std::to_string(i) + "_" + std::to_string(k));
            }
        }

        if(!cfg.is_relax && !cfg.use_lazy_constraints)
        {
            for (int i : V) {
                for (int j : V) {
                    if (i==j) continue;
                    auto key_e = std::make_tuple(k,i,j);
                    for (int u : G_k.nodes) {
                        {
                            GRBLinExpr rhs = -1 + x[std::make_pair(i,u)];
                            for (int v : G_k.nodes) {
                                if (std::find(G_k.arcs.begin(), G_k.arcs.end(), std::make_pair(u,v)) != G_k.arcs.end()) {
                                    rhs += x[std::make_pair(j,v)];
                                }
                            }
                            model->addConstr(e[key_e] >= rhs, "edge_link_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(u) + "_" + std::to_string(k));
                            total_subgraph_constraints++;
                        }
                        if(0)
                        {
                            GRBLinExpr rhs = 1 - x[std::make_pair(i,u)];
                            for (int v : G_k.nodes) {
                                if (std::find(G_k.arcs.begin(), G_k.arcs.end(), std::make_pair(u,v)) != G_k.arcs.end()) {
                                    rhs += x[std::make_pair(j,v)];
                                }
                            }
                            model->addConstr(e[key_e] <= rhs, "edge_link2_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(u) + "_" + std::to_string(k));
                            total_subgraph_constraints++;
                        }
                    }
                }
            }
        }
    }
    std::cout << "Added " << total_subgraph_constraints << " subgraph constraints directly to the model." << std::endl;

    // sum_k e^k_ij == 1 for all i,j in V
    for (int i : V) {
        for (int j : V) {
            if (i == j) continue;
            GRBLinExpr lhs = 0;
            for (const auto &kv : subgraphs) {
                lhs += e[std::make_tuple(kv.first,i,j)];
            }
            model->addConstr(lhs == 1, "flow_value_" + std::to_string(i) + "_" + std::to_string(j));
        }
    }

    // -------------------------------------------------------------
    // Fixed assignments
    // -------------------------------------------------------------
    if (!problem.fixed_assignments.empty()) {

        std::cout << "Adding "
                << problem.fixed_assignments.size()
                << " fixed assignment constraints"
                << std::endl;

        for (const auto& kv : problem.fixed_assignments) {

            int i = kv.first;
            int u = kv.second;

            model->addConstr(
                x[std::make_pair(i, u)] == 1,
                "fixed_" +
                std::to_string(i) + "_" +
                std::to_string(u)
            );

            std::cout << "Fixed assignment: x["
                    << i << "," << u << "] = 1"
                    << std::endl;
        }
    }

    if(cfg.use_lazy_constraints)
        model->set(GRB_IntParam_LazyConstraints, 1);
    model->update();

    return {model, x, e};
}



Solution SFDGurobiSolver::solve(Problem& problem,
                                const std::string& instance_path,
                                const std::unordered_map<int,int>& fixed_variables)
{
    // if cfg.matrix_for_decomposition == "distance", we will swap D and F before decomposition, then swap back when interpreting solution. This is to achieve Python parity with the existing SFD implementation which was hardcoded to decompose the flow graph.
    if (cfg.matrix_for_decomposition == "distance") {
        std::cout << "Swapping distance and flow matrices for decomposition (Python parity)." << std::endl;
        std::swap(problem.D, problem.F);
    }
    // decompose flow graph into subgraphs (value_only by default)
    subgraphs = decomposeFlowgraphValueOnly(problem);

    std::cout << "Decomposed flow graph into " << subgraphs.size() << " subgraphs for SFD." << std::endl;
    try {
        // Build the model
        auto start_time = std::chrono::high_resolution_clock::now();
        auto [model, x, e] = buildModel(problem, subgraphs, fixed_variables);
        auto build_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start_time
        ).count();
        std::cout << "Model built in " << build_time << " seconds." << std::endl;

        if (cfg.lpmethod == "barrier") {
            std::cout << "Using barrier method for LP relaxations." << std::endl;
            model->set(GRB_IntParam_Method, GRB_METHOD_BARRIER);
        } else if (cfg.lpmethod == "dual") {
            std::cout << "Using dual simplex method for LP relaxations." << std::endl;
            model->set(GRB_IntParam_Method, GRB_METHOD_DUAL);
        } else if (cfg.lpmethod == "primal") {
            std::cout << "Using primal simplex method for LP relaxations." << std::endl;
            model->set(GRB_IntParam_Method, GRB_METHOD_PRIMAL);
        } else if (cfg.lpmethod == "concurrent") {
            std::cout << "Using concurrent method for LP relaxations." << std::endl;
            model->set(GRB_IntParam_Method, GRB_METHOD_CONCURRENT);
        } else if (cfg.lpmethod == "auto") {
            std::cout << "Using Gurobi's automatic LP method selection." << std::endl;
        } else {
            std::cerr << "Unknown LP method: " << cfg.lpmethod << std::endl;
        }

        // Warm-start from file (Python parity includes index swap when D/F are swapped)
        if (!cfg.warmstart.empty()) {
            int file_n = 0;
            std::vector<int> loc_to_fac;
            double ws_obj = 0.0;
            qap::read_solution(cfg.warmstart, file_n, loc_to_fac, ws_obj);
            // if the matrix is distances then F and D were swapped before decomposition, so the warmstart file (which is based on location->facility) should be interpreted as facility->location when loading x variables.
            if (cfg.matrix_for_decomposition == "distance") {
                std::cout << "Interpreting warm-start file with swapped indices (Python parity)." << std::endl;
                std::vector<int> fac_to_loc(file_n, -1);
                for (int loc = 0; loc < file_n; ++loc) {
                    int fac = loc_to_fac[loc];
                    if (fac >= 0 && fac < file_n) fac_to_loc[fac] = loc;
                }
                loc_to_fac = fac_to_loc; // swap interpretation to match model variables
            }

            if (file_n == problem.n) {
                for (int i = 0; i < problem.n; ++i) {
                    for (int u = 0; u < problem.m; ++u) {
                        bool match = false;
                        x[std::make_pair(i, u)].set(GRB_DoubleAttr_Start, loc_to_fac[i] == u ? 1.0 : 0.0);
                    }
                }

                std::vector<int> loc_of_fac(problem.m, -1);
                for (int loc = 0; loc < problem.n; ++loc) {
                    int fac = loc_to_fac[loc];
                    if (fac >= 0 && fac < problem.m) loc_of_fac[fac] = loc;
                }

                for (const auto &kv : this->subgraphs) {
                    int k = kv.first;
                    auto G_k = kv.second;
                    for (int i = 0; i < problem.n; ++i) {
                        for (int j = 0; j < problem.n; ++j) {
                            if (i == j) continue;
                            auto key = std::make_tuple(k, i, j);
                            if (e.find(key) == e.end()) continue;
                            double val = 0.0;
                            for (const auto &uv : G_k.arcs) {
                                int u = uv.first;
                                int v = uv.second;
                                if (loc_to_fac[i] == u && loc_to_fac[j] == v) { val = 1.0; break; }
                            }
                            e[key].set(GRB_DoubleAttr_Start, val);
                        }
                    }
                }
                std::cout << "Warm-start loaded from: " << cfg.warmstart << std::endl;
            } else {
                std::cerr << "Warning: warm-start size mismatch, expected " << problem.n << " got " << file_n << std::endl;
            }
        }
        
        SFDBendersCallback* cb_ptr = nullptr;
        SFDBendersCallback cb(x, e, this->subgraphs, problem, problem.n, cfg.use_lazy_constraints, cfg.use_cuts);
        if (cfg.use_lazy_constraints || cfg.use_cuts) {
            cb_ptr = &cb;
            if (cfg.use_lazy_constraints) model->set(GRB_IntParam_LazyConstraints, 1);
            // add z to model
            auto z_var = model->addVar(0.0, GRB_INFINITY, 1.0, GRB_CONTINUOUS, "z");
            model->update();
            cb.z_var = z_var;
            model->setCallback(cb_ptr);
            std::cout << "SFD callback enabled (lazy=" << cfg.use_lazy_constraints
                      << ", cuts=" << cfg.use_cuts << ")." << std::endl;
        }

        {
            auto start_solve_time = std::chrono::high_resolution_clock::now();
            // Solve the model
            model->optimize();
            auto solve_time = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_solve_time
            ).count();
            std::cout << "Model solved in " << solve_time << " seconds." << std::endl;
            
        }
        if (cb_ptr != nullptr) {
            model->setCallback(nullptr);
            std::cout << "Callback added lazy cuts: " << cb.addedLazyCuts()
                      << ", node cuts: " << cb.addedNodeCuts() << std::endl;
            std::cout << "Callback was called " << cb.n_callbacks << " times." << std::endl;
            std::cout << "Callback average time: " << 0.001*cb.callback_time/cb.n_callbacks << " seconds." << std::endl;
        }
        // print model to file for debugging
        // model->write("SFD_gurobi_cpp.lp");

        // Extract the solution
        Solution solution(instance_path, "SFD_Gurobi");
        solution.assignment.resize(problem.n, -1);
        std::cout << "Populating assignment from x variables." << std::endl;
        try {
            int sol_count = model->get(GRB_IntAttr_SolCount);
            if (sol_count <= 0) {
                int status = model->get(GRB_IntAttr_Status);
                std::cerr << "No feasible MIP solution found (status=" << status << ")." << std::endl;
                return solution;
            }

            for (const auto& kv : x) {
                auto [i, u] = kv.first;
                auto var = kv.second;
                double xv = var.get(GRB_DoubleAttr_X);
                if (xv > 0.5) {
                    solution.assignment[i] = u;
                }
            }

            solution.objective = model->get(GRB_DoubleAttr_ObjVal);
            solution.lower_bound = model->get(GRB_DoubleAttr_ObjBound);
        } catch (GRBException &e) {
            std::cerr << "Caught Gurobi exception while extracting MIP solution: " << e.getMessage() << " (code=" << e.getErrorCode() << ")" << std::endl;
            return solution;
        }
        // get non-zero e variables and recompute objective for verification
        std::map<std::tuple<int,int,int>, double> e_values;
        double recomputed_obj = 0.0;
        for (const auto &kv : e) {
            double xv = kv.second.get(GRB_DoubleAttr_X);
            e_values[kv.first] = xv;
            if (xv > 1e-6) {
                int k,i,j; std::tie(k,i,j) = kv.first;
                double f_k = this->subgraphs[k].f_k;
                auto coeff = problem.D[i][j] * f_k;
                recomputed_obj += coeff * xv;
                // std::cout << "e[" << k << "," << i << "," << j << "] = " << xv << ", mat_ij = " << used_matrix[i][j] << ", f_k = " << f_k << std::endl;
            }
        }
        std::cout << "Non-zero e variables:" << e_values.size() << std::endl;
        std::cout << "Objective from model: " << solution.objective << std::endl;
        std::cout << "Objective recomputed from e variables: " << recomputed_obj << std::endl;
        {
            auto recomputed_obj_from_x = 0.0;
            bool valid = true;
            for (auto i = 0; i < solution.assignment.size(); ++i) {
                auto u = solution.assignment[i];
                if (u < 0 || u >= problem.m) { valid = false; break; }
                for (auto j = 0; j < solution.assignment.size(); ++j) {
                    auto v = solution.assignment[j];
                    recomputed_obj_from_x += problem.D[i][j] * problem.F[u][v];
                }
            }
            if (valid) std::cout << "Objective recomputed from x variables: " << recomputed_obj_from_x << std::endl;
            else std::cout << "Assignment incomplete; cannot recompute objective from x variables." << std::endl;
        }
        // std::map<std::pair<int,int>, double> x_values;
        // for (const auto& kv : x) {
        //     double xv = kv.second.get(GRB_DoubleAttr_X);
        //     if (xv > 1e-6) {
        //         x_values[kv.first] = xv;
        //         // std::cout << "x[" << kv.first.first << "," << kv.first.second << "] = " << xv << std::endl;
        //     }
        // }
        return solution;
    }
    catch (GRBException e) {
        std::cerr << "Gurobi error code: " << e.getErrorCode() << std::endl;
        std::cerr << "Gurobi error message: " << e.getMessage() << std::endl;
        return Solution(instance_path, "SFD_Gurobi");
    }
    catch (...) {
        std::cerr << "Unknown error during optimization." << std::endl;
        return Solution(instance_path, "SFD_Gurobi");
    }
}