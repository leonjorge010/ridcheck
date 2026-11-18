#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "model_weights.h"

/* Quantize one window of SD_WINDOW * SD_FEATURES floats, oldest fix first. */
int sd_quantize(const float *features, int8_t *out);

/* Score an already quantized window. last_acc may be NULL. */
float sd_score_i8(const int8_t *quantized, int32_t *last_acc);

/* Quantize and score. */
float sd_score(const float *features);

/* "spoof" above the 1% cut, "watch" above the 5% cut, otherwise "real". */
const char *sd_verdict(float probability);

/* Returns the number of failing vectors. Zero means the weights match the host. */
int sd_selftest(void);

#ifdef __cplusplus
}
#endif
