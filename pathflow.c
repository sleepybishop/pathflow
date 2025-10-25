#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#define DIFFERENTIAL_EVOLUTION_IMPL
#include "de.h"

#define CLAMP(x, a, b) ((x < (a)) ? (a) : ((x > b) ? b : x))

typedef struct {
    float b;  // throughput (pkt/s)
    float p;  // loss rate
    float l;  // latency
    size_t q; // packets in queue
    size_t m; // packets allocated to this link
    size_t x; // redundant packets for estimated loss
    float t;  // estimated transfer time for this link
} path_t;

/**
 * return zscore given probability of success
 **/

float qnorm(size_t P)
{
    if (P == 50)
        return 0.0f;
    if (P <= 0)
        return -3.090f;
    if (P >= 100)
        return 3.090f;
    static const float z_c[] = {0.025f, 0.050f, 0.075f, 0.100f, 0.126f, 0.151f, 0.176f, 0.202f, 0.228f, 0.253f,
                                0.279f, 0.305f, 0.332f, 0.358f, 0.385f, 0.412f, 0.440f, 0.468f, 0.496f, 0.524f,
                                0.553f, 0.583f, 0.613f, 0.643f, 0.674f, 0.706f, 0.739f, 0.772f, 0.806f, 0.842f,
                                0.878f, 0.915f, 0.954f, 0.994f, 1.036f, 1.080f, 1.126f, 1.175f, 1.227f, 1.282f,
                                1.341f, 1.405f, 1.476f, 1.555f, 1.645f, 1.751f, 1.881f, 2.054f, 2.326f};
    return (P > 50) ? z_c[P - 51] : -z_c[49 - P];
}

size_t psi(size_t Ps, size_t m, float p)
{
    float mp = m * p;
    size_t n = ceil((qnorm(Ps) * sqrt(mp) + mp) / (1 - p));
    return m + ((n < 0) ? 0 : n);
}

float path_time(size_t m, path_t *path, size_t Ps)
{
    return psi(Ps, m, path->p) / path->b + path->l + path->q / path->b;
};

float transfer_time(size_t N, size_t K, float penalty, float *m, path_t *path, size_t Ps)
{
    float num = 0.0, slowest = 0.0;
    for (size_t i = 0; i < N; i++) {
        if (isnan(m[i]) || isinf(m[i]) || m[i] < 0.0 || m[i] > K) {
            m[i] = 0.0;
            continue;
        }
        num += m[i];
        slowest = fmax(slowest, path_time(m[i], &path[i], Ps));
    }
    return fabs(K - num) * penalty + slowest;
}

float optimizer(size_t N, size_t K, path_t *path, float deadline, size_t Ps)
{
    int ret = 0;

    de_optimiser *solver = de_init(&(de_settings){
        .dimension_count = N,
        .population_count = 20.0 * N,
        .lower_bound = 0.0f,
        .upper_bound = deadline,
        .random_seed = 0x5eed,
    });
    float *candidate = calloc(N, sizeof(float));

    if (solver == NULL || candidate == NULL)
        return -1;

    float Z = FLT_MAX;
    float epsilon = 1e-6;
    size_t iters = 1000000, unchanged = 0;
    for (size_t i = 0; i < iters; i++) {
        int id = de_ask(solver, candidate);
        float fitness = transfer_time(N, K, deadline, candidate, path, Ps);

        Z = de_best(solver, NULL);
        if (fitness + epsilon < Z) {
            unchanged = 0;
            // printf("step [%08zu] fitness: %16.6f\n", i, Z);
        } else {
            unchanged++;
            if (unchanged == iters / 10)
                break;
        }
        de_tell(solver, id, candidate, fitness);
    }
    for (int i = 0; i < N; i++) {
        path[i].m = (size_t)round(candidate[i]);
        path[i].x = psi(Ps, path[i].m, path[i].p);
        path[i].t = path_time(path[i].m, &path[i], Ps);
    }
    free(candidate);
    de_deinit(solver);

    return Z;
}

int main(int argc, char *arvg[])
{
    path_t path[16] = {0};
    size_t N = 0, K = 0;
    float Ps_f = 0.0;

    FILE *in = fopen("problem.txt", "r");
    if (!in)
        exit(-1);

    fscanf(in, "N: %zu\n", &N);
    fscanf(in, "K: %zu\n", &K);
    fscanf(in, "Ps: %f\n", &Ps_f);
    N = CLAMP(N, 0, 16);
    K = CLAMP(K, 10, 1000);
    Ps_f = CLAMP(Ps_f, 0.01, 0.99);
    size_t Ps = Ps_f * 100;
    printf("N: %zu, K: %zu, Ps: %.2f\n", N, K, Ps_f);
    int i = 0;
    for (i = 0; i < N && !feof(in); i++) {
        int got = fscanf(in, "%f %f %f %zu\n", &path[i].b, &path[i].l, &path[i].p, &path[i].q);
        if (got != 4)
            break;
        printf("path[%d] %f %f %f %zu\n", i, path[i].b, path[i].l, path[i].p, path[i].q);
    }
    fclose(in);
    if (i != N)
        exit(-1);

    float total_time = optimizer(N, K, path, 60.0, Ps);
    printf("estimated transfer time: %.2fs\n", total_time);
    size_t alloc = 0, extra = 0;
    printf("%5s: %5s + %5s  [ %6s | %6s | %6s ] -> %8s\n", "path", "alloc", "extra", "tput", "lat", "loss", "xfer");
    for (int j = 0; j < N; j++) {
        alloc += path[j].m;
        extra += path[j].x;

        printf("m[%2d]: %5zu + %5zu  [ %6.0f | %6.1f | %6.1f ] -> %8.3f\n", j, (size_t)path[j].m, path[j].x - path[j].m, path[j].b,
               path[j].l, path[j].p, path[j].t);
    }
    printf("m[xx]: %5zu   %5zu\n", alloc, extra);

    return 0;
}
