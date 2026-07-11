#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CLAMP(x, a, b) ((x < (a)) ? (a) : ((x > (b)) ? (b) : (x)))

#include "pathflow.h"
#include <stdint.h>

/* Static memory arena for solvers to avoid dynamic allocation */
#ifndef PATHFLOW_ARENA_SIZE
#define PATHFLOW_ARENA_SIZE                                                    \
    262144 /* 256 KB should be plenty for N=16 pop=320 */
#endif

static uint8_t pathflow_arena[PATHFLOW_ARENA_SIZE];
static size_t pathflow_arena_offset = 0;

static void *pathflow_alloc(size_t sz) {
    /* Align to 8 bytes */
    size_t remainder = sz % 8;
    if (remainder != 0)
        sz += (8 - remainder);

    if (pathflow_arena_offset + sz > PATHFLOW_ARENA_SIZE)
        return NULL;
    void *ptr = &pathflow_arena[pathflow_arena_offset];
    pathflow_arena_offset += sz;
    return ptr;
}

static void pathflow_free(void *p) {
    (void)p; /* No-op, we reset the entire arena per optimization pass */
}

static void pathflow_arena_reset(void) { pathflow_arena_offset = 0; }

#define DE_ALLOC(sz) pathflow_alloc(sz)
#define DE_FREE(p) pathflow_free(p)
#define SA_ALLOC(sz) pathflow_alloc(sz)
#define SA_FREE(p) pathflow_free(p)
#define PSO_ALLOC(sz) pathflow_alloc(sz)
#define PSO_FREE(p) pathflow_free(p)
#define GA_ALLOC(sz) pathflow_alloc(sz)
#define GA_FREE(p) pathflow_free(p)
#define ACO_ALLOC(sz) pathflow_alloc(sz)
#define ACO_FREE(p) pathflow_free(p)
#define TS_ALLOC(sz) pathflow_alloc(sz)
#define TS_FREE(p) pathflow_free(p)

#include "solvers.h"

static inline size_t calculate_n(fp_t c1, fp_t c2, fp_t m) {
    fp_t val = FP_MUL(c1, fp_sqrt(m)) + FP_MUL(c2, m);
    return (val <= 0) ? 0 : (size_t)FP_TO_INT(fp_ceil(val));
}

static inline fp_t calculate_time(fp_t b, fp_t l, size_t q, fp_t x) {
    return FP_DIV(x, b) + l + FP_DIV(FP_FROM_INT(q), b);
}
/* return zscore given probability of success */
static fp_t qnorm(size_t P) {
    if (P == 50)
        return FP_FROM_FLOAT(0.0f);
    if (P <= 0)
        return -3.090f;
    if (P >= 100)
        return 3.090f;
    static const fp_t z_c[] = {FP_FROM_FLOAT(0.025f),
                               FP_FROM_FLOAT(0.050f),
                               0.075f,
                               0.100f,
                               0.126f,
                               0.151f,
                               0.176f,
                               0.202f,
                               0.228f,
                               0.253f,
                               0.279f,
                               0.305f,
                               0.332f,
                               0.358f,
                               0.385f,
                               0.412f,
                               0.440f,
                               0.468f,
                               0.496f,
                               0.524f,
                               0.553f,
                               0.583f,
                               0.613f,
                               0.643f,
                               0.674f,
                               0.706f,
                               0.739f,
                               0.772f,
                               0.806f,
                               0.842f,
                               0.878f,
                               0.915f,
                               0.954f,
                               0.994f,
                               1.036f,
                               1.080f,
                               1.126f,
                               1.175f,
                               1.227f,
                               1.282f,
                               1.341f,
                               1.405f,
                               1.476f,
                               1.555f,
                               1.645f,
                               1.751f,
                               FP_FROM_FLOAT(1.881f),
                               FP_FROM_FLOAT(2.054f),
                               FP_FROM_FLOAT(2.326f)};
    return (P > 50) ? z_c[P - 51] : -z_c[49 - P];
}

static fp_t greedy_solver(size_t N, size_t K, path_t *path, const fp_t *c1,
                          const fp_t *c2, const fp_t *link_time) {
    N = (N > MAX_LINKS) ? MAX_LINKS : N;
    fp_t next_times[MAX_LINKS];

    for (size_t i = 0; i < N; i++) {
        path[i].m = 0;
        if (link_time != NULL) {
            next_times[i] = link_time[i * (K + 1) + 1];
        } else {
            size_t n = calculate_n(c1[i], c2[i], FP_FROM_INT(1));
            next_times[i] = calculate_time(path[i].b, path[i].l, path[i].q,
                                           FP_FROM_INT(1 + n));
        }
    }

    for (size_t k = 0; k < K; k++) {
        size_t best_idx = 0;
        fp_t min_time = FP_MAX;
        for (size_t i = 0; i < N; i++) {
            if (next_times[i] < min_time) {
                min_time = next_times[i];
                best_idx = i;
            }
        }
        path[best_idx].m++;
        size_t next_m = path[best_idx].m + 1;
        if (next_m <= K) {
            if (link_time != NULL) {
                next_times[best_idx] = link_time[best_idx * (K + 1) + next_m];
            } else {
                size_t n = calculate_n(c1[best_idx], c2[best_idx],
                                       FP_FROM_INT(next_m));
                next_times[best_idx] =
                    calculate_time(path[best_idx].b, path[best_idx].l,
                                   path[best_idx].q, FP_FROM_INT(next_m + n));
            }
        } else {
            next_times[best_idx] = FP_MAX;
        }
    }

    fp_t slowest = FP_FROM_FLOAT(0.0f);
    for (size_t i = 0; i < N; i++) {
        if (path[i].m > 0) {
            size_t n = calculate_n(c1[i], c2[i], FP_FROM_INT(path[i].m));
            path[i].x = path[i].m + n;
            if (link_time != NULL) {
                path[i].t = link_time[i * (K + 1) + path[i].m];
            } else {
                path[i].t = calculate_time(path[i].b, path[i].l, path[i].q,
                                           FP_FROM_INT(path[i].x));
            }
            if (path[i].t > slowest) {
                slowest = path[i].t;
            }
        } else {
            path[i].x = 0;
            path[i].t = FP_FROM_FLOAT(0.0f);
        }
    }
    return slowest;
}

static fp_t transfer_time(size_t N, size_t K, fp_t penalty_weight,
                          const fp_t *m, const path_t *path, const fp_t *c1,
                          const fp_t *c2, const fp_t *link_time) {
    fp_t num = FP_FROM_FLOAT(0.0f), slowest = FP_FROM_FLOAT(0.0f);
    for (size_t i = 0; i < N; i++) {
        fp_t clamped_m = CLAMP(m[i], FP_FROM_FLOAT(0.0f), FP_FROM_INT(K));
        size_t int_m = FP_TO_INT(fp_round(clamped_m));
        num += FP_FROM_INT(int_m);
        if (int_m > 0) {
            if (path[i].b <= FP_FROM_FLOAT(0.0f)) {
                return FP_MAX;
            }
            fp_t time;
            if (link_time != NULL) {
                time = link_time[i * (K + 1) + int_m];
            } else {
                size_t n = calculate_n(c1[i], c2[i], FP_FROM_INT(int_m));
                time = calculate_time(path[i].b, path[i].l, path[i].q,
                                      FP_FROM_INT(int_m + n));
            }
            if (time > slowest) {
                slowest = time;
            }
        }
    }
    return FP_MUL(fp_abs(FP_FROM_INT(K) - num), penalty_weight) + slowest;
}

void pathflow_update_state(path_state_t *state, fp_t b, fp_t l, fp_t p,
                           size_t q, fp_t alpha) {
    alpha =
        CLAMP(alpha, FP_FROM_FLOAT(FP_FROM_FLOAT(0.0f)), FP_FROM_FLOAT(1.0f));
    if (!state->initialized) {
        state->b_ewma = b;
        state->l_ewma = l;
        state->p_ewma = p;
        state->q_ewma = q;
        state->initialized = 1;
    } else {
        state->b_ewma = FP_MUL(alpha, b) +
                        FP_MUL((FP_FROM_FLOAT(1.0f) - alpha), state->b_ewma);
        state->l_ewma = FP_MUL(alpha, l) +
                        FP_MUL((FP_FROM_FLOAT(1.0f) - alpha), state->l_ewma);
        state->p_ewma = FP_MUL(alpha, p) +
                        FP_MUL((FP_FROM_FLOAT(1.0f) - alpha), state->p_ewma);
        state->q_ewma = (size_t)FP_TO_INT(fp_round(
            FP_MUL(alpha, FP_FROM_INT(q)) +
            FP_MUL((FP_FROM_FLOAT(1.0f) - alpha), FP_FROM_INT(state->q_ewma))));
    }
}

fp_t pathflow_optimize(size_t N, size_t K, path_t *path, fp_t penalty_weight,
                       size_t Ps, pathflow_solver_t solver_type) {
    N = (N > MAX_LINKS) ? MAX_LINKS : N;

    fp_t c1[MAX_LINKS];
    fp_t c2[MAX_LINKS];
    fp_t q = qnorm(Ps);

    for (size_t i = 0; i < N; i++) {
        fp_t p = CLAMP(path[i].p, FP_FROM_FLOAT(0.0f), FP_FROM_FLOAT(0.999f));
        fp_t one_minus_p = FP_FROM_FLOAT(1.0f) - p;
        c1[i] = FP_DIV(FP_MUL(q, fp_sqrt(p)), one_minus_p);
        c2[i] = FP_DIV(p, one_minus_p);
    }

    pathflow_arena_reset();
    fp_t *link_time = NULL;
    size_t table_sz = N * (K + 1) * sizeof(fp_t);
    if (table_sz <= PATHFLOW_ARENA_SIZE - pathflow_arena_offset) {
        link_time = pathflow_alloc(table_sz);
        if (link_time != NULL) {
            for (size_t i = 0; i < N; i++) {
                link_time[i * (K + 1) + 0] = 0;
                for (size_t m = 1; m <= K; m++) {
                    size_t n = calculate_n(c1[i], c2[i], FP_FROM_INT(m));
                    link_time[i * (K + 1) + m] = calculate_time(
                        path[i].b, path[i].l, path[i].q, FP_FROM_INT(m + n));
                }
            }
        }
    }

    if (solver_type == PATHFLOW_SOLVER_GREEDY) {
        return greedy_solver(N, K, path, c1, c2, link_time);
    }

    /* Select solver plugin based on solver_type */
    const solver_interface_t *plugin;
    switch (solver_type) {
    case PATHFLOW_SOLVER_SA:
        plugin = &sa_plugin;
        break;
    case PATHFLOW_SOLVER_PSO:
        plugin = &pso_plugin;
        break;
    case PATHFLOW_SOLVER_GA:
        plugin = &ga_plugin;
        break;
    case PATHFLOW_SOLVER_ACO:
        plugin = &aco_plugin;
        break;
    case PATHFLOW_SOLVER_TS:
        plugin = &ts_plugin;
        break;
    case PATHFLOW_SOLVER_DE:
    default:
        plugin = &de_plugin;
        break;
    }

    void *solver = plugin->init(&(solver_settings_t){
        .dimension_count = N,
        .population_count = 20 * N,
        .lower_bound = FP_FROM_FLOAT(0.0f),
        .upper_bound = FP_FROM_INT(K),
        .random_seed = 0x5eed,
    });
    fp_t candidate[MAX_LINKS];

    if (solver == NULL) {
        return -1;
    }

    fp_t greedy_fitness = greedy_solver(N, K, path, c1, c2, link_time);
    for (size_t i = 0; i < N; i++) {
        candidate[i] = FP_FROM_INT(path[i].m);
    }
    plugin->tell(solver, 0, candidate, greedy_fitness);

    fp_t Z = FP_MAX;
    fp_t epsilon = FP_FROM_FLOAT(1e-6);
    size_t iters = 1000000, unchanged = 0;
    for (size_t i = 0; i < iters; i++) {
        int id = plugin->ask(solver, candidate);
        fp_t fitness = transfer_time(N, K, penalty_weight, candidate, path, c1,
                                     c2, link_time);
        plugin->tell(solver, id, candidate, fitness);

        fp_t current_best = plugin->best(solver, NULL);
        if (current_best + epsilon < Z) {
            unchanged = 0;
            Z = current_best;
        } else {
            unchanged++;
            if (unchanged == iters / 10)
                break;
        }
    }
    Z = plugin->best(solver, candidate);
    for (size_t i = 0; i < N; i++) {
        fp_t clamped_candidate =
            CLAMP(candidate[i], FP_FROM_FLOAT(0.0f), FP_FROM_INT(K));
        path[i].m = (size_t)FP_TO_INT(fp_round(clamped_candidate));
        if (path[i].m > 0) {
            size_t n = calculate_n(c1[i], c2[i], FP_FROM_INT(path[i].m));
            path[i].x = path[i].m + n;
            if (link_time != NULL) {
                path[i].t = link_time[i * (K + 1) + path[i].m];
            } else {
                path[i].t = calculate_time(path[i].b, path[i].l, path[i].q,
                                           FP_FROM_INT(path[i].x));
            }
        } else {
            path[i].x = 0;
            path[i].t = FP_FROM_FLOAT(0.0f);
        }
    }
    plugin->deinit(solver);

    return Z;
}
