/* Copyright 2026 Antigravity PSO Solver
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
    float lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    float upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int random_seed; /* Seed for the optimiser's pseudo random number generator
                      */
} pso_settings;

typedef struct pso_optimiser {
    int dimension_count;  /* Number of dimensions in the optimisation problem */
    int population_count; /* Number of particles in the population */
    float lower_bound;    /* Lower bound of the search space (same in all
                             dimensions) */
    float upper_bound;    /* Upper bound of the search space (same in all
                             dimensions) */
    int best; /* Index of the particle with the lowest personal best fitness */

    float *fitnesses;  /* Current fitness of each particle */
    float *candidates; /* Current position of each particle (population_count *
                          dimension_count) */
    float *velocities; /* Current velocity of each particle (population_count *
                          dimension_count) */
    float *proposed_velocities; /* Temporary storage for velocities proposed in
                                   ask (population_count * dimension_count) */
    float *pbests; /* Personal best position of each particle (population_count
                      * dimension_count) */
    float *pbest_fitnesses; /* Personal best fitness of each particle */
    float *gbest;           /* Global best position (dimension_count) */
    float gbest_fitness;    /* Global best fitness */
    int *step_counts;       /* Per-particle step counts */
    int next_id;            /* Next particle ID to update in ask */
    uint32_t rng[4];        /* PRNG state */
} pso_optimiser;

/* Initialise the optimiser. Returns NULL if any allocation failed. */
pso_optimiser *pso_init(pso_settings *settings);

/* Ask the optimiser to generate a candidate solution for evaluation */
int pso_ask(pso_optimiser *opt, float *out_candidate);

/* Tell the optimiser the fitness of a candidate solution */
void pso_tell(pso_optimiser *opt, int id, const float *candidate,
              float fitness);

/* Query the optimiser for the current best fitness and corresponding candidate
 * solution. */
float pso_best(pso_optimiser *opt, float *out_candidate);

/* Free the optimiser and its memory pools */
void pso_deinit(pso_optimiser *opt);

#endif /* PARTICLE_SWARM_OPTIMISATION_H */

/* Implementation */

#ifdef PARTICLE_SWARM_OPTIMISATION_IMPL

#include <math.h>
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

static float pso__next_float(uint32_t s[4]) {
    /* Only the upper 28 bits are high entropy enough */
    const uint32_t max_draw = (~(uint32_t)0) >> 4;
    const float divisor = 1.0f / (float)max_draw;
    return (float)(pso__next(s) >> 4) * divisor;
}

pso_optimiser *pso_init(pso_settings *settings) {
    const int dimension_count = settings->dimension_count;
    const int population_count = settings->population_count;
    const float lower_bound = settings->lower_bound;
    const float upper_bound = settings->upper_bound;
    const int random_seed = settings->random_seed;
    float range, v_max;
    uint32_t rng[4];
    uint32_t sm_state;
    int i;

    /* Allocate the optimiser and its memory pools */
    pso_optimiser *opt = (pso_optimiser *)PSO_ALLOC(sizeof(pso_optimiser));
    float *fitnesses = (float *)PSO_ALLOC(sizeof(float) * population_count);
    float *candidates =
        (float *)PSO_ALLOC(sizeof(float) * dimension_count * population_count);
    float *velocities =
        (float *)PSO_ALLOC(sizeof(float) * dimension_count * population_count);
    float *proposed_velocities =
        (float *)PSO_ALLOC(sizeof(float) * dimension_count * population_count);
    float *pbests =
        (float *)PSO_ALLOC(sizeof(float) * dimension_count * population_count);
    float *pbest_fitnesses =
        (float *)PSO_ALLOC(sizeof(float) * population_count);
    float *gbest = (float *)PSO_ALLOC(sizeof(float) * dimension_count);
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
    v_max = 0.20f * range;

    for (i = 0; i < population_count; i++) {
        fitnesses[i] = INFINITY;
        pbest_fitnesses[i] = INFINITY;
        step_counts[i] = 0;
    }

    /* Initialise the candidates to random points and project to sum to K */
    for (i = 0; i < population_count; i++) {
        int d;
        for (d = 0; d < dimension_count; d++) {
            candidates[i * dimension_count + d] =
                lower_bound + pso__next_float(rng) * range;
        }

        /* Project to sum to K (which is upper_bound) */
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

        for (d = 0; d < dimension_count; d++) {
            pbests[i * dimension_count + d] =
                candidates[i * dimension_count + d];
            velocities[i * dimension_count + d] =
                (pso__next_float(rng) * 2.0f - 1.0f) * v_max;
            proposed_velocities[i * dimension_count + d] =
                velocities[i * dimension_count + d];
        }
    }

    for (i = 0; i < dimension_count; i++) {
        gbest[i] = 0.0f;
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
        .gbest_fitness = INFINITY,
        .step_counts = step_counts,
        .next_id = 0,
        .rng = {rng[0], rng[1], rng[2], rng[3]},
    };

    return opt;
}

int pso_ask(pso_optimiser *opt, float *out_candidate) {
    const int population_count = opt->population_count;
    const int dimension_count = opt->dimension_count;
    const float lower_bound = opt->lower_bound;
    const float upper_bound = opt->upper_bound;

    int id = opt->next_id;
    opt->next_id = (opt->next_id + 1) % population_count;

    const float *x = &opt->candidates[id * dimension_count];
    const float *pbest = &opt->pbests[id * dimension_count];
    const float *gbest = opt->gbest;
    const float *v = &opt->velocities[id * dimension_count];

    float range = upper_bound - lower_bound;
    float v_max = 0.20f * range;

    /* Inertia weight decays from 0.8 to 0.4 based on steps */
    float w = 0.8f - 0.4f * ((float)opt->step_counts[id] / 5000.0f);
    int d;

    if (w < 0.4f)
        w = 0.4f;

    const float c1 = 1.49618f;
    const float c2 = 1.49618f;

    for (d = 0; d < dimension_count; d++) {
        float r1 = pso__next_float(opt->rng);
        float r2 = pso__next_float(opt->rng);

        /* Cognitive and social components */
        float cog = c1 * r1 * (pbest[d] - x[d]);
        float soc = 0.0f;
        if (opt->gbest_fitness != INFINITY) {
            soc = c2 * r2 * (gbest[d] - x[d]);
        }

        float v_new = w * v[d] + cog + soc;

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
        float sum = 0.0f;
        for (d = 0; d < dimension_count; d++) {
            sum += out_candidate[d];
        }
        float error = sum - upper_bound;
        for (d = 0; d < dimension_count; d++) {
            out_candidate[d] -= error / (float)dimension_count;
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

void pso_tell(pso_optimiser *opt, int id, const float *candidate,
              float fitness) {
    const int dimension_count = opt->dimension_count;

    /* Always update candidate position and velocity in PSO */
    memcpy(&opt->candidates[id * dimension_count], candidate,
           sizeof(float) * dimension_count);
    memcpy(&opt->velocities[id * dimension_count],
           &opt->proposed_velocities[id * dimension_count],
           sizeof(float) * dimension_count);
    opt->fitnesses[id] = fitness;

    /* Update personal best */
    if (opt->pbest_fitnesses[id] == INFINITY ||
        fitness < opt->pbest_fitnesses[id]) {
        memcpy(&opt->pbests[id * dimension_count], candidate,
               sizeof(float) * dimension_count);
        opt->pbest_fitnesses[id] = fitness;
    }

    /* Update global best */
    if (opt->gbest_fitness == INFINITY || fitness < opt->gbest_fitness) {
        memcpy(opt->gbest, candidate, sizeof(float) * dimension_count);
        opt->gbest_fitness = fitness;
        opt->best = id;
    }

    opt->step_counts[id]++;
}

float pso_best(pso_optimiser *opt, float *out_candidate) {
    const int dimension_count = opt->dimension_count;
    const int candidate_bytes = sizeof(float) * dimension_count;

    if (out_candidate && opt->gbest_fitness != INFINITY) {
        memcpy(out_candidate, opt->gbest, candidate_bytes);
    } else if (out_candidate) {
        memcpy(out_candidate, &opt->candidates[opt->best * dimension_count],
               candidate_bytes);
    }

    return opt->gbest_fitness == INFINITY ? opt->fitnesses[opt->best]
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
