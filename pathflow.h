#ifndef PATHFLOW_H
#define PATHFLOW_H

#include <stddef.h>

#define MAX_LINKS 16

typedef struct {
    float b;  // throughput (pkt/s)
    float p;  // loss rate
    float l;  // latency
    size_t q; // packets in queue
    size_t m; // packets allocated to this link
    size_t x; // redundant packets for estimated loss
    float t;  // estimated transfer time for this link
} path_t;

typedef struct {
    float b_ewma;
    float l_ewma;
    float p_ewma;
    size_t q_ewma;
    int initialized;
} path_state_t;

void pathflow_update_state(path_state_t *state, float b, float l, float p,
                           size_t q, float alpha);
float pathflow_optimize(size_t N, size_t K, path_t *path, float penalty_weight,
                        size_t Ps);

#endif // PATHFLOW_H
