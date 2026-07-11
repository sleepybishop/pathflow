#ifndef PATHFLOW_SOLVERS_H
#define PATHFLOW_SOLVERS_H

#define DIFFERENTIAL_EVOLUTION_IMPL
#include "de/de.h"

#define SIMULATED_ANNEALING_IMPL
#include "sa/sa.h"

#define PARTICLE_SWARM_OPTIMISATION_IMPL
#include "pso/pso.h"

#define GENETIC_ALGORITHM_IMPL
#include "ga/ga.h"

#define ANT_COLONY_OPTIMISATION_IMPL
#include "aco/aco.h"

#define TABU_SEARCH_IMPL
#include "ts/ts.h"

typedef struct {
    int dimension_count;
    int population_count;
    fp_t lower_bound;
    fp_t upper_bound;
    int random_seed;
} solver_settings_t;

typedef struct {
    void *(*init)(const solver_settings_t *settings);
    int (*ask)(void *opt, fp_t *out_candidate);
    void (*tell)(void *opt, int id, const fp_t *candidate, fp_t fitness);
    fp_t (*best)(void *opt, fp_t *out_candidate);
    void (*deinit)(void *opt);
} solver_interface_t;

#define DEFINE_SOLVER_PLUGIN(name)                                             \
    static void *name##_init_wrap(const solver_settings_t *s) {                \
        return name##_init(&(name##_settings){                                 \
            s->dimension_count, s->population_count, s->lower_bound,           \
            s->upper_bound, s->random_seed});                                  \
    }                                                                          \
    static int name##_ask_wrap(void *opt, fp_t *out) {                         \
        return name##_ask((name##_optimiser *)opt, out);                       \
    }                                                                          \
    static void name##_tell_wrap(void *opt, int id, const fp_t *candidate,     \
                                 fp_t fitness) {                               \
        name##_tell((name##_optimiser *)opt, id, candidate, fitness);          \
    }                                                                          \
    static fp_t name##_best_wrap(void *opt, fp_t *out) {                       \
        return name##_best((name##_optimiser *)opt, out);                      \
    }                                                                          \
    static void name##_deinit_wrap(void *opt) {                                \
        name##_deinit((name##_optimiser *)opt);                                \
    }                                                                          \
    static const solver_interface_t name##_plugin = {                          \
        name##_init_wrap, name##_ask_wrap, name##_tell_wrap, name##_best_wrap, \
        name##_deinit_wrap};

DEFINE_SOLVER_PLUGIN(de)
DEFINE_SOLVER_PLUGIN(sa)
DEFINE_SOLVER_PLUGIN(pso)
DEFINE_SOLVER_PLUGIN(ga)
DEFINE_SOLVER_PLUGIN(aco)
DEFINE_SOLVER_PLUGIN(ts)

#endif /* PATHFLOW_SOLVERS_H */
