/* Binary stream harness for the identical production inference kernel. */
#include <stdio.h>
#include <stdint.h>
#include "inference.h"
#include "model_config.h"
int main(void) {
    uint8_t input[MODEL_INPUT];
    int32_t out[3];
    size_t n;
    while ((n = fread(input, 1, sizeof input, stdin)) != 0) {
        if (n != sizeof input) return 2;
        uint32_t spikes;
        keyword_infer(input, out, &spikes);
        out[2] = (int32_t)spikes;
        if (fwrite(out, sizeof out, 1, stdout) != 1) return 3;
    }
    return ferror(stdin) ? 4 : 0;
}
