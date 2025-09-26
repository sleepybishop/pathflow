#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define AM_IMPLEMENTATION
#include "amoeba.h"

#define CLAMP(x, a, b) ((x < (a)) ? (a) : ((x > b) ? b : x))

typedef struct {
    float b;  // throughput (pkt/s)
    float p;  // loss rate
    float l;  // latency
    size_t q; // packets in queue
    size_t m; // packets allocated to this link
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

float optimizer(size_t N, size_t Kval, path_t *path)
{
    int ret = 0;
    float Zval = 0.0;
    am_Solver *solver;
    am_Variable *m[N], *Z;
    am_Constraint *c[N], *K;

    solver = am_newsolver(NULL, NULL);
    if (solver == NULL)
        return -1;
    Z = am_newvariable(solver);
    K = am_newconstraint(solver, AM_REQUIRED);
    am_addconstant(K, Kval);
    am_setrelation(K, AM_EQUAL);

    for (int i = 0; i < N; i++) {
        m[i] = am_newvariable(solver);
        c[i] = am_newconstraint(solver, AM_REQUIRED);
        am_addterm(c[i], Z, 1.0);
        am_setrelation(c[i], AM_GREATEQUAL);
        am_addterm(c[i], m[i], 1.0 / path[i].b);
        am_addconstant(c[i], path[i].l + path[i].q / path[i].b);
        am_add(c[i]);
        am_addterm(K, m[i], 1.0);
    }
    am_add(K);

    am_updatevars(solver);
    for (int i = 0; i < N; i++) {
        path[i].m = (size_t)ceil(am_value(m[i]));
    }
    Zval = am_value(Z);
    am_delsolver(solver);
    return Zval;
}

int main(int argc, char *arvg[])
{
    path_t path[16] = {0};
    size_t N = 0, K = 0;
    float Ps = 0.0;

    FILE *in = fopen("problem.txt", "r");
    if (!in)
        exit(-1);

    fscanf(in, "N: %zu\n", &N);
    fscanf(in, "K: %zu\n", &K);
    fscanf(in, "Ps: %f\n", &Ps);
    N = CLAMP(N, 0, 16);
    K = CLAMP(K, 10, 1000);
    Ps = CLAMP(Ps, 0.01, 0.99);
    printf("N: %zu, K: %zu, Ps: %.2f\n", N, K, Ps);
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

    float total_time = optimizer(N, K, path);
    printf("estimated transfer time: %.2fs\n", total_time);
    for (i = 0; i < N; i++) {
        printf("m[%2d]: %4zu with %4.1f%% loss -> %4zu\n", i, path[i].m, 100.0 * (path[i].p), (size_t)psi(Ps*100, path[i].m, path[i].p));
    }

    return 0;
}
