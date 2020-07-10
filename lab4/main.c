#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define HLEFT(index) (2 * ((index) + 1))
#define HRIGHT(index) (2 * ((index) + 2))

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

void multi_merge(int *data, int k, const int *sizes) {
    int total = 0;
    for (int i = 0; i < k; i++) {
        total += sizes[i];
    }
    int *list = malloc(total * sizeof(int));    // Scratch area
    int listsize = 0;
    int *index = malloc((k + 1) * sizeof(int)); // Start index of each "way"
    int *ptr = malloc(k * sizeof(int));         // Current index in each "way"
    int *heap = malloc(k * sizeof(int));
    int heapsize = k; // If a way has been merged completely we need to decrement heap size
    index[0] = 0;
    for (int i = 0; i < k; i++) {
        ptr[i] = index[i];
        index[i + 1] = index[i] + sizes[i];
        heap[i] = i;
    }

    // Multi-way merge using heap
    // heap[i] means the "head" element of way i
    // So the actual element is data[ptr[heap[i]]]
    for (int i = (heapsize - 1) / 2; i > 0; i--) {
        int smallest = i;
        if (HLEFT(i) < heapsize && data[ptr[heap[smallest]]] > data[ptr[heap[HLEFT(i)]]]) {
            smallest = HLEFT(i);
        }
        if (HRIGHT(i) < heapsize && data[ptr[heap[smallest]]] > data[ptr[heap[HRIGHT(i)]]]) {
            smallest = HRIGHT(i);
        }
        if (smallest != i) {
            int t = heap[i];
            heap[i] = heap[smallest];
            heap[smallest] = t;
        }
    }
    while (1) {
        list[listsize++] = data[ptr[heap[0]]];;
        ptr[heap[0]]++;
        if (ptr[heap[0]] >= index[heap[0] + 1]) {
            // This "way" reached its end
            heap[0] = heap[--heapsize];
            if (heapsize == 0)
                break;
        }

        for (int i = 0, smallest;; i = smallest) {
            smallest = i;
            if (HLEFT(i) < heapsize && data[ptr[heap[smallest]]] > data[ptr[heap[HLEFT(i)]]]) {
                smallest = HLEFT(i);
            }
            if (HRIGHT(i) < heapsize && data[ptr[heap[smallest]]] > data[ptr[heap[HRIGHT(i)]]]) {
                smallest = HRIGHT(i);
            }
            if (smallest == i)
                break;
            int t = heap[i];
            heap[i] = heap[smallest];
            heap[smallest] = t;
        }
    }
    free(index);
    free(ptr);
    free(heap);

    for (int i = 0; i < listsize; i++) {
        data[i] = list[i];
    }
    free(list);
}

void multi_merge_flat(int *data, int k, int eachsize) {
    int *sizes = malloc(k * sizeof(int));
    for (int i = 0; i < k; i++)
        sizes[i] = eachsize;
    multi_merge(data, k, sizes);
    free(sizes);
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

    // Dispatch first batch of jobs
    MPI_Scatterv(root_data, block_sizes, displs, MPI_INT, this_data, this_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Local sort
    quick_sort(this_data, this_size);

    // Sampling
    int *samples = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        samples[i] = this_data[this_size * (i + 1) / (size + 1)];
    }
    MPI_Gather(samples, size, MPI_INT, root_samples, size * size, MPI_INT, 0, MPI_COMM_WORLD);
    free(samples);

    int *pivots = malloc((size - 1) * sizeof(int));
    if (rank == 0) {
        multi_merge_flat(root_samples, size, size);
        for (int i = 0; i < size - 1; i++) {
            pivots[i] = root_samples[size * (i + 1)];
        }
    }
    MPI_Bcast(pivots, size - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Partition by pivots

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
