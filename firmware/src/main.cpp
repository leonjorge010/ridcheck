/* Remote ID scorer for a generic ESP32-S3 dev board (ESP32-S3-DevKitC-1).

   One line in, one line out. The line is 128 comma-separated floats: 8 location
   fixes, oldest first, 16 features each, in the order documented in
   model_weights.h. The sketch prints the score and shows it on the onboard
   RGB LED: solid for a real track, blinking for a false one, off until the
   first score.

   DevKitC-1 v1.1 wires that LED to GPIO48. Older v1.0 boards use GPIO38;
   build with -DRIDCHECK_LED_PIN=38 for those. The LED is a single-wire RGB
   pixel, so the sketch writes it with rgbLedWrite rather than digitalWrite.

   SELFTEST on its own line re-runs the baked vectors and leaves the LED as it
   was.
*/

#include <Arduino.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "ridcheck_infer.h"

#ifndef RIDCHECK_LED_PIN
#ifdef RGB_BUILTIN
#define RIDCHECK_LED_PIN RGB_BUILTIN
#else
#define RIDCHECK_LED_PIN 48
#endif
#endif

static const int kUserLed = RIDCHECK_LED_PIN;
static const uint8_t kLedLevel = 48;
static const uint32_t kBlinkMs = 250;

enum Lamp : uint8_t { kLampOff, kLampSolid, kLampBlink };

static Lamp lamp = kLampOff;
static bool led_on = false;

static void write_led(bool on) {
    if (on == led_on) {
        return;
    }
    led_on = on;
    uint8_t level = on ? kLedLevel : 0;
    rgbLedWrite(kUserLed, level, level, level);
}

static void paint_led() {
    bool on = false;
    if (lamp == kLampSolid) {
        on = true;
    } else if (lamp == kLampBlink) {
        on = ((millis() / kBlinkMs) & 1u) == 0u;
    }
    write_led(on);
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
    led_on = true;
    write_led(false);
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
