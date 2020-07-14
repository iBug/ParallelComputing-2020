#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void qsort_recurse(int *a, int n) {
    if (n <= 5)
        return;
    int i = 0, j = n - 1, flag = 0;
    while (i < j) {
        if (a[i] > a[j]) {
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
            flag = !flag;
        }
        flag ? j-- : i++;
    }
    qsort_recurse(a, i);
    qsort_recurse(a + i + 1, n - i - 1);
}

void quick_sort(int *a, int n) {
    // Mixed sort actually
    qsort_recurse(a, n);
    for (int i = 1, j; i < n; i++) {
        int key = a[i];
        for (j = i - 1; j >= 0; j--) {
            if (a[j] < key)
                break;
            a[j + 1] = a[j];
        }
        a[j + 1] = key;
    }
}

int main() {
    unsigned long n;
    scanf(" %lu", &n);
    int *a = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        scanf(" %d", &a[i]);
    }

    const double starttime = omp_get_wtime();
    quick_sort(a, n);
    const double endtime = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        printf("%d\n", a[i]);
    }
    free(a);

    const char *log_time_file = getenv("LOG_TIME_FILE");
    if (log_time_file != NULL) {
        FILE *fp = fopen(log_time_file, "a");
        fprintf(fp, "%lf\n", endtime - starttime);
        fclose(fp);
    }
    return 0;
}
