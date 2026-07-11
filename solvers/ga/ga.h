/* Copyright 2026 Antigravity GA Solver
 *
 * Licensed under the Apache License, Version FP_FROM_FLOAT(2.0) (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-FP_FROM_FLOAT(2.0)
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#if (!defined(GA_ALLOC) && defined(GA_FREE)) ||                                \
    (defined(GA_ALLOC) && !defined(GA_FREE))
#error "Must define both or neither of GA_ALLOC and GA_FREE."
#endif

#ifndef GA_ALLOC
#define GA_ALLOC(sz) malloc(sz)
#define GA_FREE(p) free(p)
#endif

#include <stdint.h>

typedef struct ga_settings {
    int dimension_count;  /* Number of dimensions in the optimisation problem */
    int population_count; /* Number of agents in the population */
    fp_t lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    fp_t upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int random_seed; /* Seed for the optimiser's pseudo random number generator
                      */
} ga_settings;

typedef struct ga_optimiser {
    int dimension_count;  /* Number of dimensions in the optimisation problem */
    int population_count; /* Number of agents in the population */
    fp_t lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    fp_t upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int best;             /* Index of the agent with the lowest fitness */

    fp_t *fitnesses;  /* Per-agent fitness */
    fp_t *candidates; /* Per-agent candidate vectors (population_count *
                          dimension_count) */
    int total_steps;   /* Global step counter */
    uint32_t rng[4];   /* PRNG state */
} ga_optimiser;

#define GA_MEMORY_REQUIRED(dimensions, population) ( \
    sizeof(ga_optimiser) +                           \
    (sizeof(fp_t) * (population)) +                 \
    (sizeof(fp_t) * (dimensions) * (population))    \
)

/* Initialise the optimiser. Returns NULL if any allocation failed. */
ga_optimiser *ga_init(ga_settings *settings);

/* Ask the optimiser to generate a candidate solution for evaluation */
int ga_ask(ga_optimiser *opt, fp_t *out_candidate);

/* Tell the optimiser the fitness of a candidate solution */
void ga_tell(ga_optimiser *opt, int id, const fp_t *candidate, fp_t fitness);

/* Query the optimiser for the current best fitness and corresponding candidate
 * solution. */
fp_t ga_best(ga_optimiser *opt, fp_t *out_candidate);

/* Free the optimiser and its memory pools */
void ga_deinit(ga_optimiser *opt);

#endif /* GENETIC_ALGORITHM_H */

/* Implementation */

#ifdef GENETIC_ALGORITHM_IMPL

#include "fpmath.h"
#include <stdlib.h>
#include <string.h>

/* Random number generation, courtesy of
 * https://prng.di.unimi.it/xoshiro128plus.c */

static inline uint32_t ga__rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint32_t ga__next(uint32_t s[4]) {
    const uint32_t result = s[0] + s[3];
    const uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = ga__rotl(s[3], 11);

    return result;
}

static inline fp_t ga__next_fp(uint32_t s[4]) {
    return (fp_t)(ga__next(s) % (uint32_t)FP_ONE);
}

ga_optimiser *ga_init(ga_settings *settings) {
    const int dimension_count = settings->dimension_count;
    const int population_count = settings->population_count;
    const fp_t lower_bound = settings->lower_bound;
    const fp_t upper_bound = settings->upper_bound;
    const int random_seed = settings->random_seed;
    fp_t range;
    uint32_t rng[4];
    uint32_t sm_state;
    int i;

    /* Allocate the optimiser and its memory pools */
    ga_optimiser *opt = (ga_optimiser *)GA_ALLOC(sizeof(ga_optimiser));
    fp_t *fitnesses = (fp_t *)GA_ALLOC(sizeof(fp_t) * population_count);
    fp_t *candidates =
        (fp_t *)GA_ALLOC(sizeof(fp_t) * dimension_count * population_count);

    if (!opt || !fitnesses || !candidates) {
        GA_FREE(opt);
        GA_FREE(fitnesses);
        GA_FREE(candidates);
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
        fitnesses[i] = FP_MAX;
    }

    /* Initialise candidates randomly and project to hyperplane */
    for (i = 0; i < population_count; i++) {
        int d;
        for (d = 0; d < dimension_count; d++) {
            candidates[i * dimension_count + d] =
                lower_bound + FP_MUL(ga__next_fp(rng), range);
        }

        /* Project to sum to K (which is upper_bound) */
        if (dimension_count > 1) {
            int iter;
            for (iter = 0; iter < 3; iter++) {
                fp_t sum = FP_FROM_FLOAT(0.0f);
                for (d = 0; d < dimension_count; d++) {
                    sum += candidates[i * dimension_count + d];
                }
                fp_t error = sum - upper_bound;
                for (d = 0; d < dimension_count; d++) {
                    candidates[i * dimension_count + d] -=
                        FP_DIV(error, (fp_t)dimension_count);
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

    *opt = (ga_optimiser){
        .dimension_count = dimension_count,
        .population_count = population_count,
        .lower_bound = lower_bound,
        .upper_bound = upper_bound,
        .best = 0,
        .fitnesses = fitnesses,
        .candidates = candidates,
        .total_steps = 0,
        .rng = {rng[0], rng[1], rng[2], rng[3]},
    };

    return opt;
}

int ga_ask(ga_optimiser *opt, fp_t *out_candidate) {
    const int population_count = opt->population_count;
    const int dimension_count = opt->dimension_count;
    const fp_t lower_bound = opt->lower_bound;
    const fp_t upper_bound = opt->upper_bound;

    /* In the initialization phase, return initial random candidates one by one
     */
    if (opt->total_steps < population_count) {
        memcpy(out_candidate,
               &opt->candidates[opt->total_steps * dimension_count],
               sizeof(fp_t) * dimension_count);
        return opt->total_steps;
    }

    /* Tournament selection for parent 1 */
    int p1_id = ga__next(opt->rng) % population_count;
    int i;
    for (i = 0; i < 2; i++) {
        int competitor = ga__next(opt->rng) % population_count;
        if (opt->fitnesses[competitor] < opt->fitnesses[p1_id]) {
            p1_id = competitor;
        }
    }

    /* Tournament selection for parent 2 */
    int p2_id = ga__next(opt->rng) % population_count;
    for (i = 0; i < 2; i++) {
        int competitor = ga__next(opt->rng) % population_count;
        if (opt->fitnesses[competitor] < opt->fitnesses[p2_id]) {
            p2_id = competitor;
        }
    }

    const fp_t *p1 = &opt->candidates[p1_id * dimension_count];
    const fp_t *p2 = &opt->candidates[p2_id * dimension_count];

    /* Uniform Crossover */
    int d;
    for (d = 0; d < dimension_count; d++) {
        if (ga__next_fp(opt->rng) < FP_FROM_FLOAT(0.50f)) {
            out_candidate[d] = p1[d];
        } else {
            out_candidate[d] = p2[d];
        }
    }

    /* Mutation */
    if (ga__next_fp(opt->rng) < FP_FROM_FLOAT(0.20f)) {
        fp_t range = upper_bound - lower_bound;
        /* Linearly decay mutation step size */
        fp_t progress = FP_DIV(FP_FROM_INT(opt->total_steps), FP_FROM_FLOAT(1000000.0f));
        fp_t max_step, min_step, step_size;
        if (progress > FP_FROM_FLOAT(1.0f))
            progress = FP_FROM_FLOAT(1.0f);
        max_step = FP_MUL(FP_FROM_FLOAT(0.10f), range);
        min_step = FP_FROM_FLOAT(1.0f);
        if (min_step > FP_MUL(FP_FROM_FLOAT(0.01f), range))
            min_step = FP_MUL(FP_FROM_FLOAT(0.01f), range);
        step_size = min_step + FP_MUL((max_step - min_step), (FP_FROM_FLOAT(1.0f) - progress));

        if (dimension_count > 1 && ga__next_fp(opt->rng) < FP_FROM_FLOAT(0.70f)) {
            /* Sum-preserving mutation by transferring between two genes */
            int d1 = ga__next(opt->rng) % dimension_count;
            int d2;
            fp_t val1, val2, v_min, v_max, v;
            do {
                d2 = ga__next(opt->rng) % dimension_count;
            } while (d2 == d1);

            val1 = out_candidate[d1];
            val2 = out_candidate[d2];

            v_min = (lower_bound - val1 > val2 - upper_bound)
                        ? (lower_bound - val1)
                        : (val2 - upper_bound);
            v_max = (upper_bound - val1 < val2 - lower_bound)
                        ? (upper_bound - val1)
                        : (val2 - lower_bound);

            v = FP_MUL((FP_MUL(ga__next_fp(opt->rng), FP_FROM_FLOAT(2.0f)) - FP_FROM_FLOAT(1.0f)), step_size);
            if (v < v_min)
                v = v_min;
            if (v > v_max)
                v = v_max;

            out_candidate[d1] = val1 + v;
            out_candidate[d2] = val2 - v;
        } else {
            /* Coordinate-wise mutation */
            int d_mut = ga__next(opt->rng) % dimension_count;
            fp_t v = FP_MUL((FP_MUL(ga__next_fp(opt->rng), FP_FROM_FLOAT(2.0f)) - FP_FROM_FLOAT(1.0f)), step_size);
            fp_t val = out_candidate[d_mut] + v;
            if (val < lower_bound)
                val = lower_bound;
            if (val > upper_bound)
                val = upper_bound;
            out_candidate[d_mut] = val;
        }
    }

    /* Project candidate to sum to K (which is upper_bound) */
    if (dimension_count > 1) {
        int iter;
        for (iter = 0; iter < 3; iter++) {
            fp_t sum = FP_FROM_FLOAT(0.0f);
            for (d = 0; d < dimension_count; d++) {
                sum += out_candidate[d];
            }
            fp_t error = sum - upper_bound;
            for (d = 0; d < dimension_count; d++) {
                out_candidate[d] -= FP_DIV(error, (fp_t)dimension_count);
                if (out_candidate[d] < lower_bound) {
                    out_candidate[d] = lower_bound;
                }
                if (out_candidate[d] > upper_bound) {
                    out_candidate[d] = upper_bound;
                }
            }
        }
    }

    /* Return the worst individual in the population to be replaced */
    int worst_id = 0;
    fp_t max_fitness = -FP_FROM_FLOAT(1.0f);
    for (i = 0; i < population_count; i++) {
        if (opt->fitnesses[i] > max_fitness) {
            max_fitness = opt->fitnesses[i];
            worst_id = i;
        }
    }

    return worst_id;
}

void ga_tell(ga_optimiser *opt, int id, const fp_t *candidate, fp_t fitness) {
    const int dimension_count = opt->dimension_count;

    /* Replace if better, or if initial loading */
    if (opt->fitnesses[id] == FP_MAX || fitness < opt->fitnesses[id]) {
        memcpy(&opt->candidates[id * dimension_count], candidate,
               sizeof(fp_t) * dimension_count);
        opt->fitnesses[id] = fitness;
        opt->best = (fitness < opt->fitnesses[opt->best]) ? id : opt->best;
    }

    opt->total_steps++;
}

fp_t ga_best(ga_optimiser *opt, fp_t *out_candidate) {
    const int dimension_count = opt->dimension_count;
    const int candidate_bytes = sizeof(fp_t) * dimension_count;

    if (out_candidate) {
        memcpy(out_candidate, &opt->candidates[opt->best * dimension_count],
               candidate_bytes);
    }

    return opt->fitnesses[opt->best];
}

void ga_deinit(ga_optimiser *opt) {
    GA_FREE(opt->fitnesses);
    GA_FREE(opt->candidates);
    GA_FREE(opt);
}

#endif /* GENETIC_ALGORITHM_IMPL */
