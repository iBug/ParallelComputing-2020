#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define G 6.67e-11
#define MASS 1e4
#define FRAME 1e-3

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

int main(int argc, char **argv) {
    int rank, size; // MPI
    int n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Load data
    int *root_data = NULL, *root_samples = NULL;
    if (rank == 0) {
        scanf(" %d", &n);
        root_data = malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            scanf(" %d", &root_data[i]);
        }
        root_samples = malloc(size * size * sizeof(int));
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Prepare parameters
    int *block_sizes = malloc(n * sizeof(int)), *displs = malloc((n + 1) * sizeof(int));
    displs[0] = 0;
    for (int i = 0; i < n; i++) {
        displs[i + 1] = n * (i + 1) / size;
        block_sizes[i] = displs[i + 1] - displs[i];
    }
    const int this_start = displs[rank];
    const int this_size = block_sizes[rank];
    int *this_data = malloc(this_size * sizeof(int));
    int *pivots = malloc((size - 1) * sizeof(int));

    // Dispatch first batch of jobs
    MPI_Scatterv(root_data, block_sizes, displs, MPI_INT, this_data, this_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Local sort
    quick_sort(this_data, this_size);

    // Sampling
    // Unfinished

    MPI_Finalize();
    free(block_sizes);
    free(displs);
    free(this_data);
    free(pivots);

    if (rank == 0) {
        printf("%d", root_data[0]);
        for (int i = 1; i < n; i++) {
            printf(" %d", root_data[i]);
        }
        printf("\n");
        free(root_data);
        free(root_samples);
    }
    return 0;
}
