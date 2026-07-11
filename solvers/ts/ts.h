/* Copyright 2026 Antigravity TS Solver
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TABU_SEARCH_H
#define TABU_SEARCH_H

#if (!defined(TS_ALLOC) && defined(TS_FREE)) ||                                \
    (defined(TS_ALLOC) && !defined(TS_FREE))
#error "Must define both or neither of TS_ALLOC and TS_FREE."
#endif

#ifndef TS_ALLOC
#define TS_ALLOC(sz) malloc(sz)
#define TS_FREE(p) free(p)
#endif

#include <stdint.h>

typedef struct ts_settings {
    int dimension_count;  /* Number of dimensions in the optimisation problem */
    int population_count; /* Number of agents in the population */
    float lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    float upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int random_seed; /* Seed for the optimiser's pseudo random number generator
                      */
} ts_settings;

typedef struct ts_optimiser {
    int dimension_count;  /* Number of dimensions in the optimisation problem */
    int population_count; /* Number of agents in the population */
    float lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    float upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int best;             /* Index of the agent with the lowest fitness */

    float *fitnesses;  /* Per-agent fitness */
    float *candidates; /* Per-agent candidate vectors (population_count *
                          dimension_count) */
    int *tabu_list;    /* Tabu tenure matrix (population_count * N * N) */
    int *step_counts;  /* Step counter for each agent */
    int *consecutive_rejects; /* Consecutive rejected moves for each agent */
    int *proposed_source;     /* Proposed source index for each agent in ask */
    int *proposed_dest; /* Proposed destination index for each agent in ask */
    float *best_candidate; /* Global best candidate vector (dimension_count) */
    float best_fitness;    /* Global best fitness */
    uint32_t rng[4];       /* PRNG state */
} ts_optimiser;

#define TS_MEMORY_REQUIRED(dimensions, population) ( \
    sizeof(ts_optimiser) +                           \
    (sizeof(float) * (population)) +                 \
    (sizeof(float) * (dimensions) * (population)) +  \
    (sizeof(int) * (population) * (dimensions) * (dimensions)) + \
    (sizeof(int) * (population)) +                   \
    (sizeof(int) * (population)) +                   \
    (sizeof(int) * (population)) +                   \
    (sizeof(int) * (population)) +                   \
    (sizeof(float) * (dimensions))                   \
)

/* Initialise the optimiser. Returns NULL if any allocation failed. */
ts_optimiser *ts_init(ts_settings *settings);

/* Ask the optimiser to generate a candidate solution for evaluation */
int ts_ask(ts_optimiser *opt, float *out_candidate);

/* Tell the optimiser the fitness of a candidate solution */
void ts_tell(ts_optimiser *opt, int id, const float *candidate, float fitness);

/* Query the optimiser for the current best fitness and corresponding candidate
 * solution. */
float ts_best(ts_optimiser *opt, float *out_candidate);

/* Free the optimiser and its memory pools */
void ts_deinit(ts_optimiser *opt);

#endif /* TABU_SEARCH_H */

/* Implementation */

#ifdef TABU_SEARCH_IMPL

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Random number generation, courtesy of
 * https://prng.di.unimi.it/xoshiro128plus.c */

static inline uint32_t ts__rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint32_t ts__next(uint32_t s[4]) {
    const uint32_t result = s[0] + s[3];
    const uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = ts__rotl(s[3], 11);

    return result;
}

static float ts__next_float(uint32_t s[4]) {
    /* Only the upper 28 bits are high entropy enough */
    const uint32_t max_draw = (~(uint32_t)0) >> 4;
    const float divisor = 1.0f / (float)max_draw;
    return (float)(ts__next(s) >> 4) * divisor;
}

ts_optimiser *ts_init(ts_settings *settings) {
    const int dimension_count = settings->dimension_count;
    const int population_count = settings->population_count;
    const float lower_bound = settings->lower_bound;
    const float upper_bound = settings->upper_bound;
    const int random_seed = settings->random_seed;
    float range;
    uint32_t rng[4];
    uint32_t sm_state;
    int i;

    /* Allocate the optimiser and its memory pools */
    ts_optimiser *opt = (ts_optimiser *)TS_ALLOC(sizeof(ts_optimiser));
    float *fitnesses = (float *)TS_ALLOC(sizeof(float) * population_count);
    float *candidates =
        (float *)TS_ALLOC(sizeof(float) * dimension_count * population_count);
    int *tabu_list = (int *)TS_ALLOC(sizeof(int) * population_count *
                                     dimension_count * dimension_count);
    int *step_counts = (int *)TS_ALLOC(sizeof(int) * population_count);
    int *consecutive_rejects = (int *)TS_ALLOC(sizeof(int) * population_count);
    int *proposed_source = (int *)TS_ALLOC(sizeof(int) * population_count);
    int *proposed_dest = (int *)TS_ALLOC(sizeof(int) * population_count);
    float *best_candidate = (float *)TS_ALLOC(sizeof(float) * dimension_count);

    if (!opt || !fitnesses || !candidates || !tabu_list || !step_counts ||
        !consecutive_rejects || !proposed_source || !proposed_dest ||
        !best_candidate) {
        TS_FREE(opt);
        TS_FREE(fitnesses);
        TS_FREE(candidates);
        TS_FREE(tabu_list);
        TS_FREE(step_counts);
        TS_FREE(consecutive_rejects);
        TS_FREE(proposed_source);
        TS_FREE(proposed_dest);
        TS_FREE(best_candidate);
        return NULL;
    }

    /* Seed the xoshiro128+ state using splitmix32 */
    sm_state = random_seed;
    for (i = 0; i < 4; i++) {
        uint32_t z = (sm_state += 0x9e3779b9);
        z = (z ^ (z >> 16)) * 0x85ebca6b;
        z = (z ^ (z >> 13)) * 0xc2b2ae35;
        rng[i] = z ^ (z >> 16);
    }

    range = upper_bound - lower_bound;

    for (i = 0; i < population_count; i++) {
        fitnesses[i] = INFINITY;
        step_counts[i] = 0;
        consecutive_rejects[i] = 0;
        proposed_source[i] = -1;
        proposed_dest[i] = -1;
    }

    memset(tabu_list, 0,
           sizeof(int) * population_count * dimension_count * dimension_count);

    /* Initialise candidates randomly and project to hyperplane */
    for (i = 0; i < population_count; i++) {
        int d;
        for (d = 0; d < dimension_count; d++) {
            candidates[i * dimension_count + d] =
                lower_bound + ts__next_float(rng) * range;
        }

        /* Project to sum to K (which is upper_bound) */
        if (dimension_count > 1) {
            int iter;
            for (iter = 0; iter < 3; iter++) {
                float sum = 0.0f;
                for (d = 0; d < dimension_count; d++) {
                    sum += candidates[i * dimension_count + d];
                }
                float error = sum - upper_bound;
                for (d = 0; d < dimension_count; d++) {
                    candidates[i * dimension_count + d] -=
                        error / (float)dimension_count;
                    if (candidates[i * dimension_count + d] < lower_bound) {
                        candidates[i * dimension_count + d] = lower_bound;
                    }
                    if (candidates[i * dimension_count + d] > upper_bound) {
                        candidates[i * dimension_count + d] = upper_bound;
                    }
                }
            }
        }
    }

    /* Fill out the optimiser struct and return the pointer to it */
    *opt = (ts_optimiser){
        .dimension_count = dimension_count,
        .population_count = population_count,
        .lower_bound = lower_bound,
        .upper_bound = upper_bound,
        .best = 0,

        .fitnesses = fitnesses,
        .candidates = candidates,
        .tabu_list = tabu_list,
        .step_counts = step_counts,
        .consecutive_rejects = consecutive_rejects,
        .proposed_source = proposed_source,
        .proposed_dest = proposed_dest,
        .best_candidate = best_candidate,
        .best_fitness = INFINITY,
        .rng = {rng[0], rng[1], rng[2], rng[3]},
    };

    return opt;
}

int ts_ask(ts_optimiser *opt, float *out_candidate) {
    const int population_count = opt->population_count;
    const int dimension_count = opt->dimension_count;
    const float lower_bound = opt->lower_bound;
    const float upper_bound = opt->upper_bound;

    int id = ts__next(opt->rng) % population_count;
    const float *x = &opt->candidates[id * dimension_count];

    /* In the initialization phase, return initial random candidates one by one
     */
    if (opt->best_fitness == INFINITY && opt->fitnesses[id] == INFINITY) {
        memcpy(out_candidate, x, sizeof(float) * dimension_count);
        return id;
    }

    memcpy(out_candidate, x, sizeof(float) * dimension_count);

    if (dimension_count <= 1) {
        return id;
    }

    /* Try to propose a valid, non-tabu neighbor move */
    int d1 = -1, d2 = -1;
    int attempts;
    for (attempts = 0; attempts < 15; attempts++) {
        d1 = ts__next(opt->rng) % dimension_count;
        d2 = ts__next(opt->rng) % dimension_count;
        if (d1 == d2)
            continue;

        /* Source must have at least 1 packet to transfer */
        if (x[d1] < 1.0f)
            continue;

        /* Check tabu status of move (d1 -> d2) */
        int tabu_expiry =
            opt->tabu_list[id * dimension_count * dimension_count +
                           d1 * dimension_count + d2];
        if (tabu_expiry <= opt->step_counts[id]) {
            /* Not tabu */
            break;
        } else if (ts__next_float(opt->rng) < 0.10f) {
            /* Aspiration criterion: allow tabu move occasionally */
            break;
        }
    }

    if (d1 != -1 && d2 != -1 && d1 != d2 && x[d1] >= 1.0f) {
        out_candidate[d1] = x[d1] - 1.0f;
        out_candidate[d2] = x[d2] + 1.0f;

        /* Project and clamp to keep it exact */
        int d;
        for (d = 0; d < dimension_count; d++) {
            if (out_candidate[d] < lower_bound)
                out_candidate[d] = lower_bound;
            if (out_candidate[d] > upper_bound)
                out_candidate[d] = upper_bound;
        }

        opt->proposed_source[id] = d1;
        opt->proposed_dest[id] = d2;
    } else {
        opt->proposed_source[id] = -1;
        opt->proposed_dest[id] = -1;
    }

    return id;
}

void ts_tell(ts_optimiser *opt, int id, const float *candidate, float fitness) {
    const int dimension_count = opt->dimension_count;
    const int TabuTenure = 8;
    int accept = 0;

    if (opt->fitnesses[id] == INFINITY) {
        accept = 1;
    } else if (fitness < opt->fitnesses[id]) {
        accept = 1;
        opt->consecutive_rejects[id] = 0;
    } else {
        opt->consecutive_rejects[id]++;
        /* If we are stuck (e.g. rejected 20 moves), force acceptance of
         * non-improving move to escape local minima */
        if (opt->consecutive_rejects[id] >= 20) {
            accept = 1;
            opt->consecutive_rejects[id] = 0;
        }
    }

    if (accept) {
        memcpy(&opt->candidates[id * dimension_count], candidate,
               sizeof(float) * dimension_count);
        opt->fitnesses[id] = fitness;
        opt->best = (fitness < opt->fitnesses[opt->best]) ? id : opt->best;

        /* Set the reverse move (dest -> source) as tabu to prevent cycling */
        int src = opt->proposed_source[id];
        int dst = opt->proposed_dest[id];
        if (src != -1 && dst != -1) {
            opt->tabu_list[id * dimension_count * dimension_count +
                           dst * dimension_count + src] =
                opt->step_counts[id] + TabuTenure;
        }
    }

    /* Update global best */
    if (opt->best_fitness == INFINITY || fitness < opt->best_fitness) {
        memcpy(opt->best_candidate, candidate, sizeof(float) * dimension_count);
        opt->best_fitness = fitness;
    }

    opt->step_counts[id]++;
}

float ts_best(ts_optimiser *opt, float *out_candidate) {
    const int dimension_count = opt->dimension_count;
    const int candidate_bytes = sizeof(float) * dimension_count;

    if (out_candidate && opt->best_fitness != INFINITY) {
        memcpy(out_candidate, opt->best_candidate, candidate_bytes);
    } else if (out_candidate) {
        memcpy(out_candidate, &opt->candidates[opt->best * dimension_count],
               candidate_bytes);
    }

    return opt->best_fitness == INFINITY ? opt->fitnesses[opt->best]
                                         : opt->best_fitness;
}

void ts_deinit(ts_optimiser *opt) {
    TS_FREE(opt->fitnesses);
    TS_FREE(opt->candidates);
    TS_FREE(opt->tabu_list);
    TS_FREE(opt->step_counts);
    TS_FREE(opt->consecutive_rejects);
    TS_FREE(opt->proposed_source);
    TS_FREE(opt->proposed_dest);
    TS_FREE(opt->best_candidate);
    TS_FREE(opt);
}

#endif /* TABU_SEARCH_IMPL */
