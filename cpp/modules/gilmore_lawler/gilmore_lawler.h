#ifndef QAP_GILMORE_LAWLER_H
#define QAP_GILMORE_LAWLER_H

#include "../../core/problem.h"
#include <vector>
#include <algorithm>
#include <limits>

struct GLBResult {
    double value;
    std::vector<int> assignment;           // assignment[i] = location assigned to facility i
    std::vector<std::vector<double>> C;      // the GLB cost matrix
};
/**
 * Gilmore-Lawler Bound computation for QAP.
 *
 * Computes the GLB by solving n assignment problems of size (n-1)x(n-1).
 */
class GilmoreLawler {
public:
    /**
     * Compute the Gilmore-Lawler bound for a QAP instance.
     * @param problem QAP instance
     * @return lower bound (double)
     */
    GLBResult compute(Problem& problem);
};

#endif // QAP_GILMORE_LAWLER_H
