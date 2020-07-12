#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

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
    const double starttime = omp_get_wtime();
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
    const double endtime = omp_get_wtime();
    printf("%zd\n", count);
    const char *log_time_file = getenv("LOG_TIME_FILE");
    if (log_time_file != NULL) {
        FILE *fp = fopen(log_time_file, "a");
        fprintf(fp, "%lf\n", endtime - starttime);
        fclose(fp);
    }
    return 0;
}
