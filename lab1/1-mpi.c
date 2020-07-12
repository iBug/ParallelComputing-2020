#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

typedef unsigned char byte;

int main(int argc, char **argv) {
    unsigned n;
    int rank, size; // MPI

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        scanf(" %u", &n);
        n += 1;
    }
    if (n < size * size) {
        fprintf(stderr, "Unexpected number %u, need at least %d for %d processes.\n", n, size * size, size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const double starttime = MPI_Wtime();
    MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned start, end, length;
    start = n * rank / size;
    end = n * (rank + 1) / size;
    length = end - start;

    byte *prime = malloc(length);
    memset(prime, 1, length);
    if (rank == 0) {
        prime[0] = prime[1] = 0;
    }
    unsigned i, k = 2;
    while (k * k <= n) {
        if (k * k >= start) {
            i = k * k - start;
        } else {
            i = ((start - 1) / k + 1) * k - start;
        }
        for (; i < length; i += k)
            prime[i] = 0;
        if (rank == 0) {
            // Find next prime
            for (i = k + 1; prime[i] == 0; i++);
            k = i;
        }
        MPI_Bcast(&k, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    unsigned cnt = 0, total = 0;
    for (unsigned i = 0; i < length; i++) {
        if (prime[i])
            cnt++;
    }
    free(prime);

    MPI_Reduce(&cnt, &total, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
    const double endtime = MPI_Wtime();
    MPI_Finalize();

    if (rank == 0) {
        printf("%u\n", total);
        const char *log_time_file = getenv("LOG_TIME_FILE");
        if (log_time_file != NULL) {
            FILE *fp = fopen(log_time_file, "a");
            fprintf(fp, "%lf\n", endtime - starttime);
            fclose(fp);
        }
    }
    return 0;
}
