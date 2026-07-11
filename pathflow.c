#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CLAMP(x, a, b) ((x < (a)) ? (a) : ((x > (b)) ? (b) : (x)))

#include "pathflow.h"
#include "solvers.h"

static inline size_t calculate_n(float c1, float c2, float m) {
    float val = c1 * sqrtf(m) + c2 * m;
    return (val < 0.0f) ? 0 : (size_t)ceilf(val);
}

static inline float calculate_time(float b, float l, size_t q, float x) {
    return x / b + l + (float)q / b;
}
/* return zscore given probability of success */
static float qnorm(size_t P) {
    if (P == 50)
        return 0.0f;
    if (P <= 0)
        return -3.090f;
    if (P >= 100)
        return 3.090f;
    static const float z_c[] = {
        0.025f, 0.050f, 0.075f, 0.100f, 0.126f, 0.151f, 0.176f, 0.202f, 0.228f,
        0.253f, 0.279f, 0.305f, 0.332f, 0.358f, 0.385f, 0.412f, 0.440f, 0.468f,
        0.496f, 0.524f, 0.553f, 0.583f, 0.613f, 0.643f, 0.674f, 0.706f, 0.739f,
        0.772f, 0.806f, 0.842f, 0.878f, 0.915f, 0.954f, 0.994f, 1.036f, 1.080f,
        1.126f, 1.175f, 1.227f, 1.282f, 1.341f, 1.405f, 1.476f, 1.555f, 1.645f,
        1.751f, 1.881f, 2.054f, 2.326f};
    return (P > 50) ? z_c[P - 51] : -z_c[49 - P];
}

static float greedy_solver(size_t N, size_t K, path_t *path, const float *c1,
                           const float *c2) {
    N = (N > MAX_LINKS) ? MAX_LINKS : N;
    float next_times[MAX_LINKS];

    for (size_t i = 0; i < N; i++) {

        path[i].m = 0;
        size_t n = calculate_n(c1[i], c2[i], 1.0f);
        next_times[i] =
            calculate_time(path[i].b, path[i].l, path[i].q, 1.0f + (float)n);
    }

    for (size_t k = 0; k < K; k++) {
        size_t best_idx = 0;
        float min_time = FLT_MAX;
        for (size_t i = 0; i < N; i++) {
            if (next_times[i] < min_time) {
                min_time = next_times[i];
                best_idx = i;
            }
        }
        path[best_idx].m++;
        size_t next_m = path[best_idx].m + 1;
        size_t n = calculate_n(c1[best_idx], c2[best_idx], (float)next_m);
        next_times[best_idx] =
            calculate_time(path[best_idx].b, path[best_idx].l, path[best_idx].q,
                           (float)next_m + (float)n);
    }

    float slowest = 0.0f;
    for (size_t i = 0; i < N; i++) {
        if (path[i].m > 0) {
            size_t n = calculate_n(c1[i], c2[i], (float)path[i].m);
            path[i].x = path[i].m + n;
            path[i].t = calculate_time(path[i].b, path[i].l, path[i].q,
                                       (float)path[i].x);
            if (path[i].t > slowest) {
                slowest = path[i].t;
            }
        } else {
            path[i].x = 0;
            path[i].t = 0.0f;
        }
    }
    return slowest;
}

static float transfer_time(size_t N, size_t K, float penalty_weight,
                           const float *m, const path_t *path, const float *c1,
                           const float *c2) {
    float num = 0.0f, slowest = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float clamped_m = CLAMP(m[i], 0.0f, (float)K);
        float rounded_m = roundf(clamped_m);
        num += rounded_m;
        if (rounded_m > 0.0f) {
            if (path[i].b <= 0.0f) {
                return FLT_MAX;
            }
            size_t n = calculate_n(c1[i], c2[i], rounded_m);
            float time = calculate_time(path[i].b, path[i].l, path[i].q,
                                        rounded_m + (float)n);
            if (time > slowest) {
                slowest = time;
            }
        }
    }
    return fabsf((float)K - num) * penalty_weight + slowest;
}

void pathflow_update_state(path_state_t *state, float b, float l, float p,
                           size_t q, float alpha) {
    alpha = CLAMP(alpha, 0.0f, 1.0f);
    if (!state->initialized) {
        state->b_ewma = b;
        state->l_ewma = l;
        state->p_ewma = p;
        state->q_ewma = q;
        state->initialized = 1;
    } else {
        state->b_ewma = alpha * b + (1.0f - alpha) * state->b_ewma;
        state->l_ewma = alpha * l + (1.0f - alpha) * state->l_ewma;
        state->p_ewma = alpha * p + (1.0f - alpha) * state->p_ewma;
        state->q_ewma = (size_t)roundf(alpha * (float)q +
                                       (1.0f - alpha) * (float)state->q_ewma);
    }
}

float pathflow_optimize(size_t N, size_t K, path_t *path, float penalty_weight,
                        size_t Ps) {
    N = (N > MAX_LINKS) ? MAX_LINKS : N;

    float c1[MAX_LINKS];
    float c2[MAX_LINKS];
    float q = qnorm(Ps);

    for (size_t i = 0; i < N; i++) {
        float p = CLAMP(path[i].p, 0.0f, 0.999f);
        float one_minus_p = 1.0f - p;
        c1[i] = (q * sqrtf(p)) / one_minus_p;
        c2[i] = p / one_minus_p;
    }

    /* Select solver plugin based on environment variable PATHFLOW_SOLVER */
    const solver_interface_t *plugin = &de_plugin;
    const char *env_solver = getenv("PATHFLOW_SOLVER");
    if (env_solver != NULL) {
        if (strcmp(env_solver, "sa") == 0) {
            plugin = &sa_plugin;
        } else if (strcmp(env_solver, "pso") == 0) {
            plugin = &pso_plugin;
        } else if (strcmp(env_solver, "ga") == 0) {
            plugin = &ga_plugin;
        } else if (strcmp(env_solver, "aco") == 0) {
            plugin = &aco_plugin;
        } else if (strcmp(env_solver, "ts") == 0) {
            plugin = &ts_plugin;
        }
    }

    void *solver = plugin->init(&(solver_settings_t){
        .dimension_count = N,
        .population_count = 20 * N,
        .lower_bound = 0.0f,
        .upper_bound = (float)K,
        .random_seed = 0x5eed,
    });
    float candidate[MAX_LINKS];

    if (solver == NULL) {
        return -1;
    }

    float greedy_fitness = greedy_solver(N, K, path, c1, c2);
    for (size_t i = 0; i < N; i++) {
        candidate[i] = (float)path[i].m;
    }
    plugin->tell(solver, 0, candidate, greedy_fitness);

    float Z = FLT_MAX;
    float epsilon = 1e-6;
    size_t iters = 1000000, unchanged = 0;
    for (size_t i = 0; i < iters; i++) {
        int id = plugin->ask(solver, candidate);
        float fitness =
            transfer_time(N, K, penalty_weight, candidate, path, c1, c2);
        plugin->tell(solver, id, candidate, fitness);

        float current_best = plugin->best(solver, NULL);
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
        float clamped_candidate = CLAMP(candidate[i], 0.0f, (float)K);
        path[i].m = (size_t)round(clamped_candidate);
        if (path[i].m > 0) {
            size_t n = calculate_n(c1[i], c2[i], (float)path[i].m);
            path[i].x = path[i].m + n;
            path[i].t = calculate_time(path[i].b, path[i].l, path[i].q,
                                       (float)path[i].x);
        } else {
            path[i].x = 0;
            path[i].t = 0.0f;
        }
    }
    plugin->deinit(solver);

    return Z;
}
