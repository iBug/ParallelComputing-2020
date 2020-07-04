#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

typedef unsigned char byte;

int main(int argc, char **argv) {
    unsigned n;
    double sum = 0.;
    int rank, size; // MPI

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        scanf(" %u", &n);
        n += 1;
    }
    MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    double step = 1.0 / n;
    for (unsigned i = rank; i < n; i += size) {
        double x = (i + 0.5) / n;
        sum += 4.0 / (1.0 + x * x);
    }
    double total;
    MPI_Reduce(&sum, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    total /= n;

    if (rank == 0) {
        printf("%.12lf\n", total);
    }
    MPI_Finalize();
    return 0;
}
