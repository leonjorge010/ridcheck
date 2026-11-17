#include "../firmware/src/ridcheck_infer.h"

#include <stdio.h>

int main(void) {
    int failed = sd_selftest();
    if (failed) {
        fprintf(stderr, "selftest failures: %d\n", failed);
        return 1;
    }
    printf("selftest ok\n");
    return 0;
}
