#include <cpuid.h>
#include <string.h>
#include <stdio.h>

int main() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    unsigned int model = ((eax >> 4) & 0xF) | ((eax >> 12) & 0xF0);

    char eax_str[9];
    sprintf(eax_str, "%08x", eax);

    if (strcmp(eax_str,"000c0662") == 0 && model == 198) {
        return 1;
    }

    return 0;
}
