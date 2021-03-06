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

float qnorm(float P) {
  float z_c[] = {0.000, 0.025, 0.050, 0.075, 0.100, 0.126, 0.151, 0.176, 0.202,
                 0.228, 0.253, 0.279, 0.305, 0.332, 0.358, 0.385, 0.412, 0.440,
                 0.468, 0.496, 0.524, 0.553, 0.583, 0.613, 0.643, 0.674, 0.706,
                 0.739, 0.772, 0.806, 0.842, 0.878, 0.915, 0.954, 0.994, 1.036,
                 1.080, 1.126, 1.175, 1.227, 1.282, 1.341, 1.405, 1.476, 1.555,
                 1.645, 1.751, 1.881, 2.054, 2.326};
  int Pi = (int)(P * 100.0);
  return (Pi >= 50) ? z_c[Pi - 50] : -z_c[50 - Pi - 1];
}

size_t psi(float Ps, size_t m, float p) {
  float mp = m * p;
  size_t n = ceil((qnorm(Ps) * sqrt(mp) + mp) / (1 - p));
  return m + ((n < 0) ? 0 : n);
}

float optimizer(size_t N, size_t Kval, path_t *path) {
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

int main(int argc, char *arvg[]) {
  path_t path[16] = {0};
  size_t N = 0, K = 0;
  float Ps = 0.0;

  FILE *in = fopen("problem.txt", "r");
  if (!in)
    exit(-1);

  fscanf(in, "N: %u\n", &N);
  fscanf(in, "K: %u\n", &K);
  fscanf(in, "Ps: %f\n", &Ps);
  N = CLAMP(N, 0, 16);
  K = CLAMP(K, 10, 1000);
  Ps = CLAMP(Ps, 0.01, 0.99);
  printf("N: %u, K: %u, Ps: %.2f\n", N, K, Ps);
  int i = 0;
  for (i = 0; i < N && !feof(in); i++) {
    int got = fscanf(in, "%f %f %f %u\n", &path[i].b, &path[i].l, &path[i].p,
                     &path[i].q);
    if (got != 4)
      break;
    printf("path[%d] %f %f %f %u\n", i, path[i].b, path[i].l, path[i].p,
           path[i].q);
  }
  fclose(in);
  if (i != N)
    exit(-1);

  float total_time = optimizer(N, K, path);
  printf("estimated transfer time: %.2fs\n", total_time);
  for (i = 0; i < N; i++) {
    printf("m[%2d]: %4d with %4.1f%% loss -> %4d\n", i, path[i].m,
           100.0 * (path[i].p), (size_t)psi(Ps, path[i].m, path[i].p));
  }

  return 0;
}
