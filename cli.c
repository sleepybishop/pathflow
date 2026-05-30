#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pathflow.h"

#define CLAMP(x, a, b) ((x < (a)) ? (a) : ((x > (b)) ? (b) : (x)))

int main() {
    path_t path[16] = {0};
    size_t N = 0, K = 0;
    float Ps_f = 0.0, deadline = 60.0;

    FILE *in = fopen("problem.txt", "r");
    if (!in) exit(-1);

    fscanf(in, "N: %zu\n", &N);
    fscanf(in, "K: %zu\n", &K);
    fscanf(in, "Ps: %f\n", &Ps_f);
    N = (N > 16) ? 16 : N;
    K = (K < 1) ? 1 : ((K > 1000) ? 1000 : K);
    Ps_f = CLAMP(Ps_f, 0.01f, 0.99f);
    size_t Ps = (size_t)roundf(Ps_f * 100.0f);
    printf("N: %zu, K: %zu, Ps: %.2f\n", N, K, Ps_f);

    size_t i = 0;
    for (i = 0; i < N && !feof(in); i++) {
        int got = fscanf(in, "%f %f %f %zu\n", &path[i].b, &path[i].l, &path[i].p, &path[i].q);
        if (got != 4) break;
        printf("path[%zu] %f %f %f %zu\n", i, path[i].b, path[i].l, path[i].p, path[i].q);
    }
    fclose(in);
    if (i != N) exit(-1);

    float total_time = pathflow_optimize(N, K, path, deadline, Ps);
    printf("estimated transfer time: %.2fs\n", total_time);
    size_t alloc = 0, extra = 0;
    printf("%5s: %5s + %5s  [ %6s | %6s | %6s ] -> %8s\n", "path", "alloc", "extra", "tput", "lat", "loss", "xfer");
    for (size_t j = 0; j < N; j++) {
        alloc += path[j].m;
        extra += path[j].x;
        printf("m[%2zu]: %5zu + %5zu  [ %6.0f | %6.3f | %6.3f ] -> %8.3f\n", j, path[j].m, path[j].x - path[j].m, path[j].b, path[j].l, path[j].p, path[j].t);
    }
    printf("m[xx]: %5zu   %5zu\n", alloc, extra);

    return 0;
}
