#pragma once
#include "IncrementalRMP.h"
#include <optional>

class PricingEngine {
public:
    struct Result {
        double best_rc = 0.0;
        std::vector<TripleKey> negative_cols;
        std::vector<TripleKey> most_negative_cols;
    };

    struct ReducedCost {
        TripleKey key;
        double rc;
        ReducedCost(TripleKey key, double rc) : key(key), rc(rc) {}
        ReducedCost(int k, int i, int j, double rc) : key(k,i,j), rc(rc) {}

        // compare by rc for sorting
        bool operator<(const ReducedCost& other) const {
            return rc < other.rc;
        }
    };

    PricingEngine(){};

    Result price(const IncrementalRMP& rmp);
    std::vector<ReducedCost> all_pricing(const IncrementalRMP& rmp);
    

private:
    double eps = 1e-7;
    std::vector<ReducedCost> all_negative_pricing(const IncrementalRMP& rmp);
    std::vector<ReducedCost> one_negative_pricing(const IncrementalRMP& rmp);

};