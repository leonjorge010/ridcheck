/* Integer inference for the spoofed-Remote-ID detector.
 * The weights live in model_weights.h, written by `python -m ridcheck train`.
 * Hidden layers are int8. The last accumulator is a float logit.
 */

#include "ridcheck_infer.h"

#include "model_weights.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int32_t sd_qround(float x) {
    if (x >= 0.f) {
        return (int32_t)floorf(x + 0.5f);
    }
    return (int32_t)ceilf(x - 0.5f);
}

static int8_t sd_requant(int32_t acc, int32_t mult, int relu) {
    int64_t scaled = ((int64_t)acc * (int64_t)mult + (1 << 15)) >> 16;
    if (relu) {
        if (scaled < 0) {
            scaled = 0;
        }
        if (scaled > 127) {
            scaled = 127;
        }
    } else {
        if (scaled < -128) {
            scaled = -128;
        }
        if (scaled > 127) {
            scaled = 127;
        }
    }
    return (int8_t)scaled;
}

static void sd_layer(const int8_t *weight, const int32_t *bias, int32_t mult,
                     int inputs, int outputs, int relu,
                     const int8_t *x, int8_t *y) {
    for (int o = 0; o < outputs; o++) {
        int32_t acc = bias[o];
        const int8_t *row = weight + (size_t)o * (size_t)inputs;
        for (int i = 0; i < inputs; i++) {
            acc += (int32_t)row[i] * (int32_t)x[i];
        }
        y[o] = sd_requant(acc, mult, relu);
    }
}

int sd_quantize(const float *features, int8_t *out) {
    for (int t = 0; t < SD_WINDOW; t++) {
        for (int f = 0; f < SD_FEATURES; f++) {
            float scale = sd_feature_scale[f];
            float q = features[t * SD_FEATURES + f] / scale;
            if (q > 127.f) {
                q = 127.f;
            }
            if (q < -127.f) {
                q = -127.f;
            }
            out[t * SD_FEATURES + f] = (int8_t)sd_qround(q);
        }
    }
    return SD_INPUTS;
}

float sd_score_i8(const int8_t *x, int32_t *last_acc) {
    int8_t hidden1[SD_H1];
    int8_t hidden2[SD_H2];
    sd_layer(sd_w1, sd_b1, sd_m1, SD_INPUTS, SD_H1, 1, x, hidden1);
    sd_layer(sd_w2, sd_b2, sd_m2, SD_H1, SD_H2, 1, hidden1, hidden2);

    int32_t acc = sd_b3[0];
    for (int i = 0; i < SD_H2; i++) {
        acc += (int32_t)sd_w3[i] * (int32_t)hidden2[i];
    }
    if (last_acc) {
        *last_acc = acc;
    }
    float logit = (float)acc * sd_scale_acc3;
    return 1.f / (1.f + expf(-logit));
}

float sd_score(const float *features) {
    int8_t quantized[SD_INPUTS];
    sd_quantize(features, quantized);
    return sd_score_i8(quantized, NULL);
}

const char *sd_verdict(float probability) {
    if (probability >= SD_THRESHOLD) {
        return "spoof";
    }
    if (probability >= SD_WATCH) {
        return "watch";
    }
    return "real";
}

int sd_selftest(void) {
    int failed = 0;
    for (int n = 0; n < SD_N_SELFTEST; n++) {
        const float *feat = sd_selftest_feat + n * SD_INPUTS;
        const int8_t *expect_q = sd_selftest_q + n * SD_INPUTS;
        int8_t got[SD_INPUTS];
        sd_quantize(feat, got);
        if (memcmp(got, expect_q, SD_INPUTS) != 0) {
            fprintf(stderr, "selftest %d quantize mismatch\n", n);
            failed++;
            continue;
        }
        int32_t acc = 0;
        float p = sd_score_i8(got, &acc);
        if (acc != sd_selftest_acc[n]) {
            fprintf(stderr, "selftest %d acc %ld != %ld\n", n,
                    (long)acc, (long)sd_selftest_acc[n]);
            failed++;
        }
        float dp = p - sd_selftest_p[n];
        if (dp < 0.f) {
            dp = -dp;
        }
        if (dp > 1e-4f) {
            fprintf(stderr, "selftest %d p %f != %f\n", n, p, sd_selftest_p[n]);
            failed++;
        }
    }
    return failed;
}
