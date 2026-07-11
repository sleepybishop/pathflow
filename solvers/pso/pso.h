/* Copyright 2026 Antigravity PSO Solver
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
#ifndef PARTICLE_SWARM_OPTIMISATION_H
#define PARTICLE_SWARM_OPTIMISATION_H

#if (!defined(PSO_ALLOC) && defined(PSO_FREE)) ||                              \
    (defined(PSO_ALLOC) && !defined(PSO_FREE))
#error "Must define both or neither of PSO_ALLOC and PSO_FREE."
#endif

#ifndef PSO_ALLOC
#define PSO_ALLOC(sz) malloc(sz)
#define PSO_FREE(p) free(p)
#endif

#include <stdint.h>

typedef struct pso_settings {
    int dimension_count;  /* Number of dimensions in the optimisation problem */
    int population_count; /* Number of particles in the population */
    fp_t lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    fp_t upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int random_seed; /* Seed for the optimiser's pseudo random number generator
                      */
} pso_settings;

typedef struct pso_optimiser {
    int dimension_count;  /* Number of dimensions in the optimisation problem */
    int population_count; /* Number of particles in the population */
    fp_t lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    fp_t upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int best; /* Index of the particle with the lowest personal best fitness */

    fp_t *fitnesses;  /* Current fitness of each particle */
    fp_t *candidates; /* Current position of each particle (population_count *
                          dimension_count) */
    fp_t *velocities; /* Current velocity of each particle (population_count *
                          dimension_count) */
    fp_t *proposed_velocities; /* Temporary storage for velocities proposed in
                                   ask (population_count * dimension_count) */
    fp_t *pbests; /* Personal best position of each particle (population_count
                      * dimension_count) */
    fp_t *pbest_fitnesses; /* Personal best fitness of each particle */
    fp_t *gbest;           /* Global best position (dimension_count) */
    fp_t gbest_fitness;    /* Global best fitness */
    int *step_counts;       /* Per-particle step counts */
    int next_id;            /* Next particle ID to update in ask */
    uint32_t rng[4];        /* PRNG state */
} pso_optimiser;

#define PSO_MEMORY_REQUIRED(dimensions, population) ( \
    sizeof(pso_optimiser) +                           \
    (sizeof(fp_t) * (population)) +                  \
    (sizeof(fp_t) * (dimensions) * (population)) +   \
    (sizeof(fp_t) * (dimensions) * (population)) +   \
    (sizeof(fp_t) * (dimensions) * (population)) +   \
    (sizeof(fp_t) * (dimensions) * (population)) +   \
    (sizeof(fp_t) * (population)) +                  \
    (sizeof(fp_t) * (dimensions)) +                  \
    (sizeof(int) * (population))                      \
)

/* Initialise the optimiser. Returns NULL if any allocation failed. */
pso_optimiser *pso_init(pso_settings *settings);

/* Ask the optimiser to generate a candidate solution for evaluation */
int pso_ask(pso_optimiser *opt, fp_t *out_candidate);

/* Tell the optimiser the fitness of a candidate solution */
void pso_tell(pso_optimiser *opt, int id, const fp_t *candidate,
              fp_t fitness);

/* Query the optimiser for the current best fitness and corresponding candidate
 * solution. */
fp_t pso_best(pso_optimiser *opt, fp_t *out_candidate);

/* Free the optimiser and its memory pools */
void pso_deinit(pso_optimiser *opt);

#endif /* PARTICLE_SWARM_OPTIMISATION_H */

/* Implementation */

#ifdef PARTICLE_SWARM_OPTIMISATION_IMPL

#include "fpmath.h"
#include <stdlib.h>
#include <string.h>

/* Random number generation, courtesy of
 * https://prng.di.unimi.it/xoshiro128plus.c */

static inline uint32_t pso__rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint32_t pso__next(uint32_t s[4]) {
    const uint32_t result = s[0] + s[3];
    const uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = pso__rotl(s[3], 11);

    return result;
}

static inline fp_t pso__next_fp(uint32_t s[4]) {
    return (fp_t)(pso__next(s) % (uint32_t)FP_ONE);
}

pso_optimiser *pso_init(pso_settings *settings) {
    const int dimension_count = settings->dimension_count;
    const int population_count = settings->population_count;
    const fp_t lower_bound = settings->lower_bound;
    const fp_t upper_bound = settings->upper_bound;
    const int random_seed = settings->random_seed;
    fp_t range, v_max;
    uint32_t rng[4];
    uint32_t sm_state;
    int i;

    /* Allocate the optimiser and its memory pools */
    pso_optimiser *opt = (pso_optimiser *)PSO_ALLOC(sizeof(pso_optimiser));
    fp_t *fitnesses = (fp_t *)PSO_ALLOC(sizeof(fp_t) * population_count);
    fp_t *candidates =
        (fp_t *)PSO_ALLOC(sizeof(fp_t) * dimension_count * population_count);
    fp_t *velocities =
        (fp_t *)PSO_ALLOC(sizeof(fp_t) * dimension_count * population_count);
    fp_t *proposed_velocities =
        (fp_t *)PSO_ALLOC(sizeof(fp_t) * dimension_count * population_count);
    fp_t *pbests =
        (fp_t *)PSO_ALLOC(sizeof(fp_t) * dimension_count * population_count);
    fp_t *pbest_fitnesses =
        (fp_t *)PSO_ALLOC(sizeof(fp_t) * population_count);
    fp_t *gbest = (fp_t *)PSO_ALLOC(sizeof(fp_t) * dimension_count);
    int *step_counts = (int *)PSO_ALLOC(sizeof(int) * population_count);

    if (!opt || !fitnesses || !candidates || !velocities ||
        !proposed_velocities || !pbests || !pbest_fitnesses || !gbest ||
        !step_counts) {
        PSO_FREE(opt);
        PSO_FREE(fitnesses);
        PSO_FREE(candidates);
        PSO_FREE(velocities);
        PSO_FREE(proposed_velocities);
        PSO_FREE(pbests);
        PSO_FREE(pbest_fitnesses);
        PSO_FREE(gbest);
        PSO_FREE(step_counts);
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
    v_max = FP_MUL(FP_FROM_FLOAT(0.20f), range);

    for (i = 0; i < population_count; i++) {
        fitnesses[i] = FP_MAX;
        pbest_fitnesses[i] = FP_MAX;
        step_counts[i] = 0;
    }

    /* Initialise the candidates to random points and project to sum to K */
    for (i = 0; i < population_count; i++) {
        int d;
        for (d = 0; d < dimension_count; d++) {
            candidates[i * dimension_count + d] =
                lower_bound + FP_MUL(pso__next_fp(rng), range);
        }

        /* Project to sum to K (which is upper_bound) */
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

        for (d = 0; d < dimension_count; d++) {
            pbests[i * dimension_count + d] =
                candidates[i * dimension_count + d];
            velocities[i * dimension_count + d] =
                FP_MUL((FP_MUL(pso__next_fp(rng), FP_FROM_FLOAT(2.0f)) - FP_FROM_FLOAT(1.0f)), v_max);
            proposed_velocities[i * dimension_count + d] =
                velocities[i * dimension_count + d];
        }
    }

    for (i = 0; i < dimension_count; i++) {
        gbest[i] = FP_FROM_FLOAT(0.0f);
    }

    /* Fill out the optimiser struct and return the pointer to it */
    *opt = (pso_optimiser){
        .dimension_count = dimension_count,
        .population_count = population_count,
        .lower_bound = lower_bound,
        .upper_bound = upper_bound,
        .best = 0,

        .fitnesses = fitnesses,
        .candidates = candidates,
        .velocities = velocities,
        .proposed_velocities = proposed_velocities,
        .pbests = pbests,
        .pbest_fitnesses = pbest_fitnesses,
        .gbest = gbest,
        .gbest_fitness = FP_MAX,
        .step_counts = step_counts,
        .next_id = 0,
        .rng = {rng[0], rng[1], rng[2], rng[3]},
    };

    return opt;
}

int pso_ask(pso_optimiser *opt, fp_t *out_candidate) {
    const int population_count = opt->population_count;
    const int dimension_count = opt->dimension_count;
    const fp_t lower_bound = opt->lower_bound;
    const fp_t upper_bound = opt->upper_bound;

    int id = opt->next_id;
    opt->next_id = (opt->next_id + 1) % population_count;

    const fp_t *x = &opt->candidates[id * dimension_count];
    const fp_t *pbest = &opt->pbests[id * dimension_count];
    const fp_t *gbest = opt->gbest;
    const fp_t *v = &opt->velocities[id * dimension_count];

    fp_t range = upper_bound - lower_bound;
    fp_t v_max = FP_MUL(FP_FROM_FLOAT(0.20f), range);

    /* Inertia weight decays from FP_FROM_FLOAT(0.8) to FP_FROM_FLOAT(0.4) based on steps */
    fp_t w = FP_FROM_FLOAT(0.8f) - FP_MUL(FP_FROM_FLOAT(0.4f), FP_DIV((fp_t)opt->step_counts[id], FP_FROM_FLOAT(5000.0f)));
    int d;

    if (w < FP_FROM_FLOAT(0.4f))
        w = FP_FROM_FLOAT(0.4f);

    const fp_t c1 = FP_FROM_FLOAT(1.49618f);
    const fp_t c2 = FP_FROM_FLOAT(1.49618f);

    for (d = 0; d < dimension_count; d++) {
        fp_t r1 = pso__next_fp(opt->rng);
        fp_t r2 = pso__next_fp(opt->rng);

        /* Cognitive and social components */
        fp_t cog = FP_MUL(FP_MUL(c1, r1), (pbest[d] - x[d]));
        fp_t soc = FP_FROM_FLOAT(0.0f);
        if (opt->gbest_fitness != FP_MAX) {
            soc = FP_MUL(FP_MUL(c2, r2), (gbest[d] - x[d]));
        }

        fp_t v_new = FP_MUL(w, v[d]) + cog + soc;

        /* Clamp velocity */
        if (v_new < -v_max)
            v_new = -v_max;
        if (v_new > v_max)
            v_new = v_max;

        out_candidate[d] = x[d] + v_new;
    }

    /* Project candidate to sum to K (which is upper_bound) */
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

    /* Compute effective corrected velocity based on projection */
    for (d = 0; d < dimension_count; d++) {
        opt->proposed_velocities[id * dimension_count + d] =
            out_candidate[d] - x[d];
    }

    return id;
}

void pso_tell(pso_optimiser *opt, int id, const fp_t *candidate,
              fp_t fitness) {
    const int dimension_count = opt->dimension_count;

    /* Always update candidate position and velocity in PSO */
    memcpy(&opt->candidates[id * dimension_count], candidate,
           sizeof(fp_t) * dimension_count);
    memcpy(&opt->velocities[id * dimension_count],
           &opt->proposed_velocities[id * dimension_count],
           sizeof(fp_t) * dimension_count);
    opt->fitnesses[id] = fitness;

    /* Update personal best */
    if (opt->pbest_fitnesses[id] == FP_MAX ||
        fitness < opt->pbest_fitnesses[id]) {
        memcpy(&opt->pbests[id * dimension_count], candidate,
               sizeof(fp_t) * dimension_count);
        opt->pbest_fitnesses[id] = fitness;
    }

    /* Update global best */
    if (opt->gbest_fitness == FP_MAX || fitness < opt->gbest_fitness) {
        memcpy(opt->gbest, candidate, sizeof(fp_t) * dimension_count);
        opt->gbest_fitness = fitness;
        opt->best = id;
    }

    opt->step_counts[id]++;
}

fp_t pso_best(pso_optimiser *opt, fp_t *out_candidate) {
    const int dimension_count = opt->dimension_count;
    const int candidate_bytes = sizeof(fp_t) * dimension_count;

    if (out_candidate && opt->gbest_fitness != FP_MAX) {
        memcpy(out_candidate, opt->gbest, candidate_bytes);
    } else if (out_candidate) {
        memcpy(out_candidate, &opt->candidates[opt->best * dimension_count],
               candidate_bytes);
    }

    return opt->gbest_fitness == FP_MAX ? opt->fitnesses[opt->best]
                                          : opt->gbest_fitness;
}

void pso_deinit(pso_optimiser *opt) {
    PSO_FREE(opt->fitnesses);
    PSO_FREE(opt->candidates);
    PSO_FREE(opt->velocities);
    PSO_FREE(opt->proposed_velocities);
    PSO_FREE(opt->pbests);
    PSO_FREE(opt->pbest_fitnesses);
    PSO_FREE(opt->gbest);
    PSO_FREE(opt->step_counts);
    PSO_FREE(opt);
}

#endif /* PARTICLE_SWARM_OPTIMISATION_IMPL */
