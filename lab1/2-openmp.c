#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N_MAX 4294967296UL

int main(void) {
    unsigned long n;
    scanf(" %lu", &n);
    if (n < 2 || n >= N_MAX) {
        // Wrong N
        return 1;
    }
    const double starttime = omp_get_wtime();
    double sum = 0;

    #pragma omp parallel for reduction(+: sum)
    for (unsigned long i = 0; i < n; i++) {
        double x = (i + 0.5) / n;
        sum += 4.0 / (1.0 + x * x);
    }

    const double endtime = omp_get_wtime();
    printf("%.12lf\n", sum / n);
    const char *log_time_file = getenv("LOG_TIME_FILE");
    if (log_time_file != NULL) {
        FILE *fp = fopen(log_time_file, "a");
        fprintf(fp, "%lf\n", endtime - starttime);
        fclose(fp);
    }
    return 0;
}
