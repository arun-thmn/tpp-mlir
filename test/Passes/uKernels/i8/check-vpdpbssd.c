#include <immintrin.h>
#include <signal.h>
#include <setjmp.h>

static sigjmp_buf jump_buffer;

void handle_sigill(int sig) {
    siglongjmp(jump_buffer, 1);
}

int main() {
    signal(SIGILL, handle_sigill);
    if (sigsetjmp(jump_buffer, 1)) {
        return 1; // VPDPBSSD not supported (SIGILL triggered)
    }

    // Setup dummy vectors
    __m256i a = _mm256_set1_epi8(1);
    __m256i b = _mm256_set1_epi8(2);
    __m256i c = _mm256_setzero_si256();

    // Inline assembly to ensure VPDPBSSD is used
    asm volatile("vpdpbssd %1, %2, %0" : "+x"(c) : "x"(a), "x"(b));

    (void)c;
    return 0; // Success, instruction executed
}

