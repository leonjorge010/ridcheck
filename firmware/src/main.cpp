/* Serial scorer for the Wio-S3 board (ESP32-S3 + SX1262, SAM-M10Q, LSM6DSV16X).

   One line in, one line out. The line is 128 comma-separated floats: 8 location
   fixes, oldest first, 16 features each, in the order documented in
   model_weights.h. The sketch prints the spoof probability.

   The SX1262 is the board's LoRa radio. Remote ID is 2.4 GHz, so a later
   tracker should fill this window from the S3's Wi-Fi and BLE radios plus the
   GNSS fix and the IMU speed. This sketch is the piece that runs the model.

   SELFTEST on its own line re-runs the baked vectors.
*/

#include <Arduino.h>

#include <cstdlib>
#include <cstring>

#include "spoofdet_infer.h"

static void score_line(char *line) {
    if (strcmp(line, "SELFTEST") == 0) {
        int failed = sd_selftest();
        Serial.printf("selftest %s (%d)\n", failed ? "FAIL" : "ok", failed);
        return;
    }
    float features[SD_INPUTS];
    int count = 0;
    char *cursor = line;
    while (count < SD_INPUTS && *cursor) {
        char *end = nullptr;
        features[count] = strtof(cursor, &end);
        if (end == cursor) {
            break;
        }
        count++;
        cursor = end;
        if (*cursor == ',') {
            cursor++;
        }
    }
    if (count != SD_INPUTS) {
        Serial.printf("need %d floats, got %d\n", SD_INPUTS, count);
        return;
    }
    float probability = sd_score(features);
    Serial.printf("p=%.3f %s\n", probability, sd_verdict(probability));
}

void setup() {
    Serial.begin(115200);
    delay(200);
    int failed = sd_selftest();
    Serial.printf("spoofdet selftest %s\n", failed ? "FAIL" : "ok");
}

void loop() {
    static char line[4096];
    static size_t used = 0;
    while (Serial.available()) {
        char byte = (char)Serial.read();
        if (byte == '\r') {
            continue;
        }
        if (byte != '\n') {
            if (used + 1 < sizeof(line)) {
                line[used++] = byte;
            }
            continue;
        }
        line[used] = '\0';
        used = 0;
        if (line[0] == '\0' || line[0] == '#') {
            continue;
        }
        score_line(line);
    }
}
