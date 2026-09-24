#include <stddef.h>
/* Freestanding GCC may lower aggregate initialization to memset. */
void *memset(void *dest, int value, size_t length) {
    volatile unsigned char *p = dest;
    for (size_t i=0; i<length; ++i) p[i]=(unsigned char)value;
    return dest;
}
