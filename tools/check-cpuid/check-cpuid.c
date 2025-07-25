#include <stdio.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>

int main() {
    unsigned int eax, ebx, ecx, edx;

    // Calling CPUID with EAX=7, ECX=1
    if (__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
        if (edx & (1 << 4)) {
            return 1;
        }
    }

    return 0;
}
#else
int main() { // skip arm architecture
    return 0;
}
#endif
