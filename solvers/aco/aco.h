/* Copyright 2026 Antigravity ACO Solver
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
#ifndef ANT_COLONY_OPTIMISATION_H
#define ANT_COLONY_OPTIMISATION_H

#if (!defined(ACO_ALLOC) && defined(ACO_FREE)) ||                              \
    (defined(ACO_ALLOC) && !defined(ACO_FREE))
#error "Must define both or neither of ACO_ALLOC and ACO_FREE."
#endif

#ifndef ACO_ALLOC
#define ACO_ALLOC(sz) malloc(sz)
#define ACO_FREE(p) free(p)
#endif

#include <stdint.h>

typedef struct aco_settings {
    int dimension_count;  /* Number of dimensions in the optimisation problem */
    int population_count; /* Number of agents in the population */
    float lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    float upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int random_seed; /* Seed for the optimiser's pseudo random number generator
                      */
} aco_settings;

typedef struct aco_optimiser {
    int dimension_count;  /* Number of dimensions in the optimisation problem */
    int population_count; /* Number of agents in the colony */
    float lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    float upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int best; /* Index of the ant with the lowest fitness in the current
                 generation */

    float *fitnesses;      /* Per-ant fitness (population_count) */
    float *candidates;     /* Per-ant candidate vectors (population_count *
                              dimension_count) */
    float *pheromones;     /* Pheromone matrix ((K + 1) * dimension_count) */
    float *best_candidate; /* Global best candidate vector (dimension_count) */
    float best_fitness;    /* Global best fitness */
    int evaluated_count;   /* Number of ants evaluated in total */
    uint32_t rng[4];       /* PRNG state */
} aco_optimiser;

#define ACO_MEMORY_REQUIRED(dimensions, population, upper_bound) ( \
    sizeof(aco_optimiser) +                                        \
    (sizeof(float) * (population)) +                               \
    (sizeof(float) * (dimensions) * (population)) +                \
    (sizeof(float) * ((int)(upper_bound) + 1) * (dimensions)) +    \
    (sizeof(float) * (dimensions))                                 \
)

/* Initialise the optimiser. Returns NULL if any allocation failed. */
aco_optimiser *aco_init(aco_settings *settings);

/* Ask the optimiser to generate a candidate solution for evaluation */
int aco_ask(aco_optimiser *opt, float *out_candidate);

/* Tell the optimiser the fitness of a candidate solution */
void aco_tell(aco_optimiser *opt, int id, const float *candidate,
              float fitness);

/* Query the optimiser for the current best fitness and corresponding candidate
 * solution. */
float aco_best(aco_optimiser *opt, float *out_candidate);

/* Free the optimiser and its memory pools */
void aco_deinit(aco_optimiser *opt);

#endif /* ANT_COLONY_OPTIMISATION_H */

/* Implementation */

#ifdef ANT_COLONY_OPTIMISATION_IMPL

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Random number generation, courtesy of
 * https://prng.di.unimi.it/xoshiro128plus.c */

static inline uint32_t aco__rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint32_t aco__next(uint32_t s[4]) {
    const uint32_t result = s[0] + s[3];
    const uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = aco__rotl(s[3], 11);

    return result;
}

static float aco__next_float(uint32_t s[4]) {
    /* Only the upper 28 bits are high entropy enough */
    const uint32_t max_draw = (~(uint32_t)0) >> 4;
    const float divisor = 1.0f / (float)max_draw;
    return (float)(aco__next(s) >> 4) * divisor;
}

aco_optimiser *aco_init(aco_settings *settings) {
    const int dimension_count = settings->dimension_count;
    const int population_count = settings->population_count;
    const float lower_bound = settings->lower_bound;
    const float upper_bound = settings->upper_bound;
    const int random_seed = settings->random_seed;
    const int K = (int)upper_bound;
    const float tau_max = 10.0f;
    float range;
    uint32_t rng[4];
    uint32_t sm_state;
    int i;

    /* Allocate the optimiser and its memory pools */
    aco_optimiser *opt = (aco_optimiser *)ACO_ALLOC(sizeof(aco_optimiser));
    float *fitnesses = (float *)ACO_ALLOC(sizeof(float) * population_count);
    float *candidates =
        (float *)ACO_ALLOC(sizeof(float) * dimension_count * population_count);
    float *pheromones =
        (float *)ACO_ALLOC(sizeof(float) * (K + 1) * dimension_count);
    float *best_candidate = (float *)ACO_ALLOC(sizeof(float) * dimension_count);

    if (!opt || !fitnesses || !candidates || !pheromones || !best_candidate) {
        ACO_FREE(opt);
        ACO_FREE(fitnesses);
        ACO_FREE(candidates);
        ACO_FREE(pheromones);
        ACO_FREE(best_candidate);
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
    }

    /* Initialise pheromones to tau_max */
    for (i = 0; i < (K + 1) * dimension_count; i++) {
        pheromones[i] = tau_max;
    }

    /* Initialise candidates randomly and project to hyperplane */
    for (i = 0; i < population_count; i++) {
        int d;
        for (d = 0; d < dimension_count; d++) {
            candidates[i * dimension_count + d] =
                lower_bound + aco__next_float(rng) * range;
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
    *opt = (aco_optimiser){
        .dimension_count = dimension_count,
        .population_count = population_count,
        .lower_bound = lower_bound,
        .upper_bound = upper_bound,
        .best = 0,

        .fitnesses = fitnesses,
        .candidates = candidates,
        .pheromones = pheromones,
        .best_candidate = best_candidate,
        .best_fitness = INFINITY,
        .evaluated_count = 0,
        .rng = {rng[0], rng[1], rng[2], rng[3]},
    };

    return opt;
}

int aco_ask(aco_optimiser *opt, float *out_candidate) {
    const int population_count = opt->population_count;
    const int dimension_count = opt->dimension_count;
    const float upper_bound = opt->upper_bound;

    /* In the initialization phase, return initial random candidates one by one
     */
    if (opt->best_fitness == INFINITY &&
        opt->evaluated_count < population_count) {
        memcpy(out_candidate,
               &opt->candidates[opt->evaluated_count * dimension_count],
               sizeof(float) * dimension_count);
        return opt->evaluated_count;
    }

    /* Ant constructs a solution step-by-step */
    int K = (int)upper_bound;
    int m[16] = {0};
    float probs[16];
    int k;

    for (k = 0; k < K; k++) {
        float sum = 0.0f;
        int d;
        for (d = 0; d < dimension_count; d++) {
            int current_m = m[d];
            if (current_m > K)
                current_m = K;
            probs[d] = opt->pheromones[current_m * dimension_count + d];
            sum += probs[d];
        }

        /* Choose a link based on probabilities */
        float r = aco__next_float(opt->rng) * sum;
        float accumulator = 0.0f;
        int chosen_d = dimension_count - 1;
        for (d = 0; d < dimension_count; d++) {
            accumulator += probs[d];
            if (r <= accumulator) {
                chosen_d = d;
                break;
            }
        }
        m[chosen_d]++;
    }

    /* Write to out_candidate */
    int d;
    for (d = 0; d < dimension_count; d++) {
        out_candidate[d] = (float)m[d];
    }

    return opt->evaluated_count % population_count;
}

void aco_tell(aco_optimiser *opt, int id, const float *candidate,
              float fitness) {
    const int dimension_count = opt->dimension_count;
    const int population_count = opt->population_count;
    const int K = (int)opt->upper_bound;

    /* Store the ant's solution and fitness */
    memcpy(&opt->candidates[id * dimension_count], candidate,
           sizeof(float) * dimension_count);
    opt->fitnesses[id] = fitness;

    /* Update generation best index */
    if (opt->evaluated_count % population_count == 0) {
        opt->best = id;
    } else if (fitness < opt->fitnesses[opt->best]) {
        opt->best = id;
    }

    /* Update global best */
    if (opt->best_fitness == INFINITY || fitness < opt->best_fitness) {
        memcpy(opt->best_candidate, candidate, sizeof(float) * dimension_count);
        opt->best_fitness = fitness;
    }

    opt->evaluated_count++;

    /* Once all ants in the colony have been evaluated, we complete a generation
     */
    if (opt->evaluated_count % population_count == 0) {
        const float rho = 0.10f;
        const float tau_min = 0.01f;
        const float tau_max = 10.0f;
        float delta_tau = 1.0f;
        int i, j, d;

        /* Evaporate pheromones (10%) and clamp */
        for (i = 0; i < (K + 1) * dimension_count; i++) {
            opt->pheromones[i] *= (1.0f - rho);
            if (opt->pheromones[i] < tau_min)
                opt->pheromones[i] = tau_min;
        }

        /* Deposit pheromones on the paths chosen by the best ant in this
         * generation */
        const float *best_candidate =
            &opt->candidates[opt->best * dimension_count];
        float best_fitness = opt->fitnesses[opt->best];
        if (best_fitness > 0.0f) {
            delta_tau = 10.0f / best_fitness;
        }

        for (d = 0; d < dimension_count; d++) {
            int allocated = (int)best_candidate[d];
            for (j = 0; j < allocated; j++) {
                if (j <= K) {
                    opt->pheromones[j * dimension_count + d] += delta_tau;
                    if (opt->pheromones[j * dimension_count + d] > tau_max) {
                        opt->pheromones[j * dimension_count + d] = tau_max;
                    }
                }
            }
        }
    }
}

float aco_best(aco_optimiser *opt, float *out_candidate) {
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

void aco_deinit(aco_optimiser *opt) {
    ACO_FREE(opt->fitnesses);
    ACO_FREE(opt->candidates);
    ACO_FREE(opt->pheromones);
    ACO_FREE(opt->best_candidate);
    ACO_FREE(opt);
}

#endif /* ANT_COLONY_OPTIMISATION_IMPL */
