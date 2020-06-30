#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define N_MAX 4294967296ULL

typedef uint8_t byte;

// Algorithm: https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
int main(void) {
    uint64_t n;
    scanf(" %lu", &n);
    if (n < 2 || n >= N_MAX) {
        // Wrong N
        return 1;
    }
    byte *prime = malloc(n * sizeof(byte));

    uint64_t sqrt_n = (uint64_t)ceil(sqrt(n));
#pragma omp parallel for
    for (uint64_t i = 2; i <= n; i++) {
        prime[i] = 1;
    }

    size_t count = 0;
    for (uint64_t i = 2; i <= n; i++) {
        if (prime[i]) {
            count++;
            if (i <= sqrt_n) {
#pragma omp parallel for
                for (uint64_t j = i * i; j <= n; j += i)
                    prime[j] = 0;
            }
        }
    }

    free(prime);
    printf("%zd\n", count);
    return 0;
}
