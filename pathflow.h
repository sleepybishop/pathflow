#ifndef PATHFLOW_H
#define PATHFLOW_H

#include "fpmath.h"
#include <stddef.h>

#define MAX_LINKS 16

typedef struct {
    fp_t b;   // throughput (pkt/s)
    fp_t p;   // loss rate
    fp_t l;   // latency
    size_t q; // packets in queue
    size_t m; // packets allocated to this link
    size_t x; // redundant packets for estimated loss
    fp_t t;   // estimated transfer time for this link
} path_t;

typedef struct {
    fp_t b_ewma;
    fp_t l_ewma;
    fp_t p_ewma;
    size_t q_ewma;
    int initialized;
} path_state_t;

void pathflow_update_state(path_state_t *state, fp_t b, fp_t l, fp_t p,
                           size_t q, fp_t alpha);
typedef enum {
    PATHFLOW_SOLVER_DE,
    PATHFLOW_SOLVER_SA,
    PATHFLOW_SOLVER_PSO,
    PATHFLOW_SOLVER_GA,
    PATHFLOW_SOLVER_ACO,
    PATHFLOW_SOLVER_TS
} pathflow_solver_t;

fp_t pathflow_optimize(size_t N, size_t K, path_t *path, fp_t penalty_weight,
                       size_t Ps, pathflow_solver_t solver_type);

#endif // PATHFLOW_H
