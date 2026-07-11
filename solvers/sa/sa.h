/* Copyright 2026 Antigravity SA Solver
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
#ifndef SIMULATED_ANNEALING_H
#define SIMULATED_ANNEALING_H

#if (!defined(SA_ALLOC) && defined(SA_FREE)) ||                                \
    (defined(SA_ALLOC) && !defined(SA_FREE))
#error "Must define both or neither of SA_ALLOC and SA_FREE."
#endif

#ifndef SA_ALLOC
#define SA_ALLOC(sz) malloc(sz)
#define SA_FREE(p) free(p)
#endif

#include <stdint.h>

typedef struct sa_settings {
    int dimension_count;  /* Number of dimensions in the optimisation problem */
    int population_count; /* Number of agents in the population */
    fp_t lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    fp_t upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int random_seed; /* Seed for the optimiser's pseudo random number generator
                      */
} sa_settings;

typedef struct sa_optimiser {
    int dimension_count;  /* Number of dimensions in the optimisation problem */
    int population_count; /* Number of agents in the population */
    fp_t lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    fp_t upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int best;             /* Index of the agent with the lowest fitness */

    fp_t *fitnesses;      /* Per-agent fitness */
    fp_t *candidates;     /* Per-agent candidate vectors (population_count *
                              dimension_count) */
    fp_t *temperatures;   /* Per-agent temperature */
    int *step_counts;      /* Per-agent step counts */
    fp_t cooling_factor;  /* Cooling decay multiplier */
    fp_t *best_candidate; /* Historic best candidate vector */
    fp_t best_fitness;    /* Historic best fitness */
    uint32_t rng[4];       /* PRNG state */
} sa_optimiser;

#define SA_MEMORY_REQUIRED(dimensions, population) ( \
    sizeof(sa_optimiser) +                           \
    (sizeof(fp_t) * (population)) +                 \
    (sizeof(fp_t) * (dimensions) * (population)) +  \
    (sizeof(fp_t) * (population)) +                 \
    (sizeof(int) * (population)) +                   \
    (sizeof(fp_t) * (dimensions))                   \
)

/* Initialise the optimiser. Returns NULL if any allocation failed. */
sa_optimiser *sa_init(sa_settings *settings);

/* Ask the optimiser to generate a candidate solution for evaluation */
int sa_ask(sa_optimiser *opt, fp_t *out_candidate);

/* Tell the optimiser the fitness of a candidate solution */
void sa_tell(sa_optimiser *opt, int id, const fp_t *candidate, fp_t fitness);

/* Query the optimiser for the current best fitness and corresponding candidate
 * solution. */
fp_t sa_best(sa_optimiser *opt, fp_t *out_candidate);

/* Free the optimiser and its memory pools */
void sa_deinit(sa_optimiser *opt);

#endif /* SIMULATED_ANNEALING_H */

/* Implementation */

#ifdef SIMULATED_ANNEALING_IMPL

#include "fpmath.h"
#include <stdlib.h>
#include <string.h>

/* Random number generation, courtesy of
 * https://prng.di.unimi.it/xoshiro128plus.c */

static inline uint32_t sa__rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint32_t sa__next(uint32_t s[4]) {
    const uint32_t result = s[0] + s[3];
    const uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = sa__rotl(s[3], 11);

    return result;
}

static inline fp_t sa__next_fp(uint32_t s[4]) {
    return (fp_t)(sa__next(s) % (uint32_t)FP_ONE);
}

sa_optimiser *sa_init(sa_settings *settings) {
    const int dimension_count = settings->dimension_count;
    const int population_count = settings->population_count;
    const fp_t lower_bound = settings->lower_bound;
    const fp_t upper_bound = settings->upper_bound;
    const int random_seed = settings->random_seed;
    fp_t cooling_factor = FP_FROM_FLOAT(0.995f);
    uint32_t rng[4];
    uint32_t sm_state;
    int i;

    /* Allocate the optimiser and its memory pools */
    sa_optimiser *opt = (sa_optimiser *)SA_ALLOC(sizeof(sa_optimiser));
    fp_t *fitnesses = (fp_t *)SA_ALLOC(sizeof(fp_t) * population_count);
    fp_t *candidates =
        (fp_t *)SA_ALLOC(sizeof(fp_t) * dimension_count * population_count);
    fp_t *temperatures = (fp_t *)SA_ALLOC(sizeof(fp_t) * population_count);
    int *step_counts = (int *)SA_ALLOC(sizeof(int) * population_count);
    fp_t *best_candidate = (fp_t *)SA_ALLOC(sizeof(fp_t) * dimension_count);

    if (!opt || !fitnesses || !candidates || !temperatures || !step_counts ||
        !best_candidate) {
        SA_FREE(opt);
        SA_FREE(fitnesses);
        SA_FREE(candidates);
        SA_FREE(temperatures);
        SA_FREE(step_counts);
        SA_FREE(best_candidate);
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

    /* Initialise fitnesses to infinity and temperatures */
    for (i = 0; i < population_count; i++) {
        fitnesses[i] = FP_MAX;
        temperatures[i] = FP_FROM_FLOAT(10.0f);
        step_counts[i] = 0;
    }

    /* Initialise the candidates to random points in the search space */
    for (i = 0; i < dimension_count * population_count; i++) {
        candidates[i] =
            lower_bound + FP_MUL(sa__next_fp(rng), (upper_bound - lower_bound));
    }

    /* Fill out the optimiser struct and return the pointer to it */
    *opt = (sa_optimiser){
        .dimension_count = dimension_count,
        .population_count = population_count,
        .lower_bound = lower_bound,
        .upper_bound = upper_bound,
        .best = 0,

        .fitnesses = fitnesses,
        .candidates = candidates,
        .temperatures = temperatures,
        .step_counts = step_counts,
        .cooling_factor = cooling_factor,
        .best_candidate = best_candidate,
        .best_fitness = FP_MAX,
        .rng = {rng[0], rng[1], rng[2], rng[3]},
    };

    return opt;
}

int sa_ask(sa_optimiser *opt, fp_t *out_candidate) {
    const int population_count = opt->population_count;
    const int dimension_count = opt->dimension_count;
    const fp_t lower_bound = opt->lower_bound;
    const fp_t upper_bound = opt->upper_bound;

    int id = sa__next(opt->rng) % population_count;
    const fp_t *x = &opt->candidates[id * dimension_count];

    /* Calculate step size based on current temperature */
    fp_t t_ratio = FP_DIV(opt->temperatures[id], FP_FROM_FLOAT(10.0f));
    fp_t range, max_step, min_step, step_size;
    int i;

    if (t_ratio > FP_FROM_FLOAT(1.0f))
        t_ratio = FP_FROM_FLOAT(1.0f);
    if (t_ratio < FP_FROM_FLOAT(0.0f))
        t_ratio = FP_FROM_FLOAT(0.0f);

    range = upper_bound - lower_bound;
    max_step = FP_MUL(FP_FROM_FLOAT(0.20f), range);
    min_step = FP_FROM_FLOAT(1.0f);
    if (min_step > FP_MUL(FP_FROM_FLOAT(0.01f), range))
        min_step = FP_MUL(FP_FROM_FLOAT(0.01f), range);
    step_size = min_step + FP_MUL((max_step - min_step), t_ratio);

    if (dimension_count <= 1) {
        fp_t val = x[0] + FP_MUL((FP_MUL(sa__next_fp(opt->rng), FP_FROM_FLOAT(2.0f)) - FP_FROM_FLOAT(1.0f)), step_size);
        if (val < lower_bound)
            val = lower_bound;
        if (val > upper_bound)
            val = upper_bound;
        out_candidate[0] = val;
        return id;
    }

    if (sa__next_fp(opt->rng) < FP_FROM_FLOAT(0.70f)) {
        int d1 = sa__next(opt->rng) % dimension_count;
        int d2;
        fp_t x1, x2, v_min, v_max, v;
        do {
            d2 = sa__next(opt->rng) % dimension_count;
        } while (d2 == d1);

        x1 = x[d1];
        x2 = x[d2];

        v_min = (lower_bound - x1 > x2 - upper_bound) ? (lower_bound - x1)
                                                      : (x2 - upper_bound);
        v_max = (upper_bound - x1 < x2 - lower_bound) ? (upper_bound - x1)
                                                      : (x2 - lower_bound);

        v = FP_MUL((FP_MUL(sa__next_fp(opt->rng), FP_FROM_FLOAT(2.0f)) - FP_FROM_FLOAT(1.0f)), step_size);
        if (v < v_min)
            v = v_min;
        if (v > v_max)
            v = v_max;

        for (i = 0; i < dimension_count; i++) {
            if (i == d1) {
                out_candidate[i] = x1 + v;
            } else if (i == d2) {
                out_candidate[i] = x2 - v;
            } else {
                out_candidate[i] = x[i];
            }
        }
    } else {
        int d = sa__next(opt->rng) % dimension_count;
        fp_t v = FP_MUL((FP_MUL(sa__next_fp(opt->rng), FP_FROM_FLOAT(2.0f)) - FP_FROM_FLOAT(1.0f)), step_size);
        for (i = 0; i < dimension_count; i++) {
            if (i == d) {
                fp_t val = x[i] + v;
                if (val < lower_bound)
                    val = lower_bound;
                if (val > upper_bound)
                    val = upper_bound;
                out_candidate[i] = val;
            } else {
                out_candidate[i] = x[i];
            }
        }
    }

    return id;
}

void sa_tell(sa_optimiser *opt, int id, const fp_t *candidate, fp_t fitness) {
    const int dimension_count = opt->dimension_count;
    int accept = 0;

    if (opt->fitnesses[id] == FP_MAX) {
        accept = 1;
    } else if (fitness < opt->fitnesses[id]) {
        accept = 1;
    } else {
        fp_t diff = fitness - opt->fitnesses[id];
        fp_t p = fp_exp(FP_DIV(-diff, opt->temperatures[id]));
        if (sa__next_fp(opt->rng) < p) {
            accept = 1;
        }
    }

    if (accept) {
        memcpy(&opt->candidates[id * dimension_count], candidate,
               sizeof(fp_t) * dimension_count);
        opt->fitnesses[id] = fitness;
        opt->best = (fitness < opt->fitnesses[opt->best]) ? id : opt->best;
    }

    if (fitness < opt->best_fitness) {
        memcpy(opt->best_candidate, candidate, sizeof(fp_t) * dimension_count);
        opt->best_fitness = fitness;
    }

    opt->step_counts[id]++;
    opt->temperatures[id] = FP_MUL(opt->temperatures[id], opt->cooling_factor);
}

fp_t sa_best(sa_optimiser *opt, fp_t *out_candidate) {
    const int dimension_count = opt->dimension_count;
    const int candidate_bytes = sizeof(fp_t) * dimension_count;

    if (out_candidate && opt->best_fitness != FP_MAX) {
        memcpy(out_candidate, opt->best_candidate, candidate_bytes);
    } else if (out_candidate) {
        memcpy(out_candidate, &opt->candidates[opt->best * dimension_count],
               candidate_bytes);
    }

    return opt->best_fitness == FP_MAX ? opt->fitnesses[opt->best]
                                         : opt->best_fitness;
}

void sa_deinit(sa_optimiser *opt) {
    SA_FREE(opt->fitnesses);
    SA_FREE(opt->candidates);
    SA_FREE(opt->temperatures);
    SA_FREE(opt->step_counts);
    SA_FREE(opt->best_candidate);
    SA_FREE(opt);
}

#endif /* SIMULATED_ANNEALING_IMPL */
