#include "pathflow.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CLAMP(x, a, b) ((x < (a)) ? (a) : ((x > (b)) ? (b) : (x)))

int main(int argc, char **argv) {
    fp_t max_p = FP_FROM_FLOAT(0.20f);
    fp_t max_l = FP_FROM_FLOAT(2.0f);
    if (argc >= 3) {
        max_p = FP_FROM_FLOAT(atof(argv[1]));
        max_l = FP_FROM_FLOAT(atof(argv[2]));
    }

    path_t path[MAX_LINKS] = {0};
    size_t N = 0, K = 0;
    float Ps_f = 0.0;
    fp_t deadline = FP_FROM_FLOAT(60.0f);

    FILE *in = fopen("problem.txt", "r");
    if (!in)
        exit(-1);

    if (fscanf(in, "N: %zu\n", &N) != 1 || fscanf(in, "K: %zu\n", &K) != 1 ||
        fscanf(in, "Ps: %f\n", &Ps_f) != 1) {
        fprintf(stderr, "Failed to parse problem.txt headers\n");
        exit(-1);
    }
    N = (N > MAX_LINKS) ? MAX_LINKS : N;
    K = (K < 1) ? 1 : ((K > 1000) ? 1000 : K);
    Ps_f = CLAMP(Ps_f, 0.01f, 0.99f);
    size_t Ps = (size_t)roundf(Ps_f * 100.0f);
    printf("N: %zu, K: %zu, Ps: %.2f\n", N, K, Ps_f);
    path_state_t states[MAX_LINKS] = {0};
    fp_t alpha = FP_FROM_FLOAT(0.25f);

    size_t i = 0;
    for (i = 0; i < N; i++) {
        float raw_b, raw_l, raw_p;
        size_t raw_q;
        int got = fscanf(in, "%f %f %f %zu\n", &raw_b, &raw_l, &raw_p, &raw_q);
        if (got != 4)
            break;
        raw_p = CLAMP(raw_p, 0.0f, 0.99f);

        pathflow_update_state(&states[i], FP_FROM_FLOAT(raw_b),
                              FP_FROM_FLOAT(raw_l), FP_FROM_FLOAT(raw_p), raw_q,
                              alpha);

        path[i].b = states[i].b_ewma;
        path[i].l = states[i].l_ewma;
        path[i].p = states[i].p_ewma;
        path[i].q = states[i].q_ewma;
    }
    fclose(in);
    if (i != N)
        exit(-1);

    path_t active_paths[MAX_LINKS] = {0};
    size_t active_N = 0;
    size_t map[MAX_LINKS] = {0};

    for (size_t j = 0; j < N; j++) {
        if (path[j].p >= max_p || path[j].l >= max_l) {
            printf("path[%zu] %f %f %f %zu (DROPPED)\n", j,
                   FP_TO_FLOAT(path[j].b), FP_TO_FLOAT(path[j].l),
                   FP_TO_FLOAT(path[j].p), path[j].q);
        } else {
            active_paths[active_N] = path[j];
            map[active_N] = j;
            printf("path[%zu] %f %f %f %zu\n", j, FP_TO_FLOAT(path[j].b),
                   FP_TO_FLOAT(path[j].l), FP_TO_FLOAT(path[j].p), path[j].q);
            active_N++;
        }
    }

    if (active_N == 0) {
        printf("ERROR: All paths dropped by fast-fail criteria.\n");
        exit(-1);
    }

    pathflow_solver_t solver_type = PATHFLOW_SOLVER_DE;
    const char *env_solver = getenv("PATHFLOW_SOLVER");
    if (env_solver != NULL) {
        if (strcmp(env_solver, "sa") == 0) {
            solver_type = PATHFLOW_SOLVER_SA;
        } else if (strcmp(env_solver, "pso") == 0) {
            solver_type = PATHFLOW_SOLVER_PSO;
        } else if (strcmp(env_solver, "ga") == 0) {
            solver_type = PATHFLOW_SOLVER_GA;
        } else if (strcmp(env_solver, "aco") == 0) {
            solver_type = PATHFLOW_SOLVER_ACO;
        } else if (strcmp(env_solver, "ts") == 0) {
            solver_type = PATHFLOW_SOLVER_TS;
        }
    }

    fp_t total_time =
        pathflow_optimize(active_N, K, active_paths, FP_FROM_FLOAT(10000.0f),
                          Ps, solver_type); // Use a fixed large penalty for DE

    for (size_t j = 0; j < active_N; j++) {
        size_t orig = map[j];
        path[orig] = active_paths[j];
    }

    printf("estimated transfer time: %.2fs\n", FP_TO_FLOAT(total_time));
    if (total_time > deadline) {
        printf(
            "WARNING: Estimated transfer time exceeds the deadline of %.2fs!\n",
            FP_TO_FLOAT(deadline));
    }

    size_t alloc = 0, extra = 0;
    printf("%5s: %5s + %5s  [ %6s | %6s | %6s ] -> %8s\n", "path", "alloc",
           "extra", "tput", "lat", "loss", "xfer");
    for (size_t j = 0; j < N; j++) {
        alloc += path[j].m;
        extra += path[j].x;

        printf("m[%2zu]: %5zu + %5zu  [ %6.0f | %6.3f | %6.3f ] -> %8.3f\n", j,
               (size_t)path[j].m, path[j].x - path[j].m, FP_TO_FLOAT(path[j].b),
               FP_TO_FLOAT(path[j].l), FP_TO_FLOAT(path[j].p),
               FP_TO_FLOAT(path[j].t));
    }
    printf("m[xx]: %5zu   %5zu\n", alloc, extra);

    return 0;
}
