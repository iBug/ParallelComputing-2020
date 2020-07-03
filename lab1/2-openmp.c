#include <stdio.h>

#define N_MAX 4294967296UL

int main(void) {
    unsigned long n;
    scanf(" %lu", &n);
    if (n < 2 || n >= N_MAX) {
        // Wrong N
        return 1;
    }
    double sum = 0;

    #pragma omp parallel for reduction(+: sum)
    for (unsigned long i = 0; i < n; i++) {
        double x = (i + 0.5) / n;
        sum += 4.0 / (1.0 + x * x);
    }

    printf("%.12lf\n", sum / n);
    return 0;
}
