#ifndef FPMATH_H
#define FPMATH_H

#include <stdint.h>

typedef int64_t fp_t;

#ifndef MAX_PACKETS
#define MAX_PACKETS 1000
#endif

/*
 * Calculate max safe fraction bits to avoid 64-bit overflow during FP_MUL.
 * Max internal value is roughly (MAX_LINKS * MAX_PACKETS * 10000.0) during
 * penalty calculations. To safely multiply them without 128-bit math: (value *
 * FP_ONE) < INT64_MAX
 */
#if MAX_PACKETS <= 1000
#define FP_FRAC_BITS 17
#elif MAX_PACKETS <= 10000
#define FP_FRAC_BITS 15
#elif MAX_PACKETS <= 100000
#define FP_FRAC_BITS 14
#elif MAX_PACKETS <= 1000000
#define FP_FRAC_BITS 12
#else
#define FP_FRAC_BITS 10
#endif

#define FP_ONE ((fp_t)(1 << FP_FRAC_BITS))
#define FP_MAX INT64_MAX

#define FP_FROM_INT(x) ((fp_t)((int64_t)(x) * (1LL << FP_FRAC_BITS)))
#define FP_TO_INT(x) ((x) >> FP_FRAC_BITS)
#define FP_FROM_FLOAT(x) ((fp_t)((x) * (float)FP_ONE))
#define FP_TO_FLOAT(x) ((float)(x) / (float)FP_ONE)

#define FP_MUL(a, b) ((fp_t)(((int64_t)(a) * (b)) >> FP_FRAC_BITS))
#define FP_DIV(a, b) ((fp_t)((((int64_t)(a) << FP_FRAC_BITS) / (b))))

/* Absolute value */
static inline fp_t fp_abs(fp_t a) { return (a < 0) ? -a : a; }

/* Floor, Ceil, Round */
static inline fp_t fp_floor(fp_t a) { return a & ~(FP_ONE - 1); }
static inline fp_t fp_ceil(fp_t a) { return (a + FP_ONE - 1) & ~(FP_ONE - 1); }
static inline fp_t fp_round(fp_t a) {
    return (a + (FP_ONE >> 1)) & ~(FP_ONE - 1);
}

/* Integer Square Root (isqrt) adapted for Fixed Point */
static inline fp_t fp_sqrt(fp_t val) {
    if (val <= 0)
        return 0;
    uint64_t res = 0;
    uint64_t bit = 1ULL << ((63 - __builtin_clzll((uint64_t)val)) & ~1);

    while (bit != 0) {
        if ((uint64_t)val >= res + bit) {
            val -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    /* The result is integer sqrt. We need to shift it because the input was
     * scaled. */
    /* If x is Q16.16, sqrt(x * 2^16) = sqrt(x) * 2^8. We need to shift it left
     * by 8 to get back to Q16.16. */
    return (fp_t)(res << (FP_FRAC_BITS / 2));
}

/*
 * Exponential function for negative values (used in SA cooling).
 * exp(x) approx 1 + x + x^2/2 + x^3/6
 * We clamp x to avoid underflow/overflow.
 */
static inline fp_t fp_exp(fp_t x) {
    if (x >= 0)
        return FP_ONE; /* We only need negative x for SA */
    if (x < FP_FROM_INT(-10))
        return 0; /* exp(-10) is tiny */

    /* Taylor series: 1 + x + x^2/2 + x^3/6 + x^4/24 */
    fp_t x2 = FP_MUL(x, x);
    fp_t x3 = FP_MUL(x2, x);
    fp_t x4 = FP_MUL(x3, x);

    fp_t res = FP_ONE + x + (x2 >> 1) + FP_DIV(x3, FP_FROM_INT(6)) +
               FP_DIV(x4, FP_FROM_INT(24));
    if (res < 0)
        return 0;
    return res;
}

#endif /* FPMATH_H */
