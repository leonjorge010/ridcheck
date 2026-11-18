/* Serial scorer for the XIAO ESP32-S3 (SX1262, SAM-M10Q, LSM6DSV16X).

   One line in, one line out. The line is 128 comma-separated floats: 8 location
   fixes, oldest first, 16 features each, in the order documented in
   model_weights.h. The sketch prints the score and shows it on the orange
   user LED: solid for a real track, blinking for a false one, off until the
   first score.

   The LED is GPIO21 and active low (high is dark). The SX1262 is the board's
   LoRa radio. Remote ID is 2.4 GHz, so a later tracker should fill this window
   from the S3's Wi-Fi and BLE radios plus the GNSS fix and the IMU speed.

   SELFTEST on its own line re-runs the baked vectors and leaves the LED as it
   was.
*/

#include <Arduino.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "ridcheck_infer.h"

// XIAO ESP32-S3 orange user LED. A high level turns it off.
static const int kUserLed = 21;
static const uint8_t kLedOn = LOW;
static const uint8_t kLedOff = HIGH;
static const uint32_t kBlinkMs = 250;

enum Lamp : uint8_t { kLampOff, kLampSolid, kLampBlink };

static Lamp lamp = kLampOff;

static void paint_led() {
    bool on = false;
    if (lamp == kLampSolid) {
        on = true;
    } else if (lamp == kLampBlink) {
        on = ((millis() / kBlinkMs) & 1u) == 0u;
    }
    digitalWrite(kUserLed, on ? kLedOn : kLedOff);
}

static void show_verdict(float probability) {
    lamp = probability >= SD_THRESHOLD ? kLampBlink : kLampSolid;
    paint_led();
}

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
    show_verdict(probability);
    Serial.printf("p=%.3f %s\n", probability, sd_verdict(probability));
}

void setup() {
    pinMode(kUserLed, OUTPUT);
    paint_led();
    Serial.begin(115200);
    delay(200);
    int failed = sd_selftest();
    Serial.printf("ridcheck selftest %s\n", failed ? "FAIL" : "ok");
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
    paint_led();
}
