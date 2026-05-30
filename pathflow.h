#ifndef PATHFLOW_H
#define PATHFLOW_H

#include <stddef.h>

typedef struct {
    float b;  // throughput (pkt/s)
    float p;  // loss rate
    float l;  // latency
    size_t q; // packets in queue
    size_t m; // packets allocated to this link
    size_t x; // redundant packets for estimated loss
    float t;  // estimated transfer time for this link
} path_t;

float pathflow_optimize(size_t N, size_t K, path_t *path, float penalty_weight, size_t Ps);

#endif // PATHFLOW_H
