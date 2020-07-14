#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define HLEFT(index) (2 * (index) + 1)
#define HRIGHT(index) (2 * (index) + 2)

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
    int heapsize; // If a way has been merged completely we need to decrement heap size
    index[0] = 0;
    for (int i = 0; i < k; i++) {
        ptr[i] = index[i];
        index[i + 1] = index[i] + sizes[i];
        heap[i] = i;
    }

    // Precautionary cleanup for sizes[i] = 0
    int missing = 0;
    for (int i = 0; i < k - missing; i++) {
        if (sizes[i] == 0) {
            missing++;
        }
        heap[i] = heap[i + missing];
    }
    heapsize = k - missing;
    if (heapsize <= 0)
        return;

    // Multi-way merge using heap
    // heap[i] means the "head" element of way i
    // So the actual element is data[ptr[heap[i]]]
    for (int i = (heapsize - 1) / 2; i >= 0; i--) {
        int j = i, smallest;
        while (1) {
            smallest = j;
            if (HLEFT(j) < heapsize && data[ptr[heap[smallest]]] > data[ptr[heap[HLEFT(j)]]]) {
                smallest = HLEFT(j);
            }
            if (HRIGHT(j) < heapsize && data[ptr[heap[smallest]]] > data[ptr[heap[HRIGHT(j)]]]) {
                smallest = HRIGHT(j);
            }
            if (smallest == j)
                break;
            int t = heap[j];
            heap[j] = heap[smallest];
            heap[smallest] = t;
            j = smallest;
        }
    }
    while (1) {
        list[listsize++] = data[ptr[heap[0]]];
        ptr[heap[0]]++;
        if (ptr[heap[0]] >= index[heap[0] + 1]) {
            // This "way" reached its end
            heap[0] = heap[--heapsize];
            if (heapsize == 0)
                break;
        }

        int i = 0, smallest;
        while (1) {
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
            i = smallest;
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
    int mpi_rank, mpi_size; // MPI
    int n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Load data
    int *root_data = NULL, *root_samples = NULL,
        *block_sizes = NULL, *displs = NULL;
    if (mpi_rank == 0) {
        scanf(" %d", &n);
        root_data = malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            scanf(" %d", &root_data[i]);
        }
        root_samples = malloc(mpi_size * mpi_size * sizeof(int));
    }
    const double starttime = MPI_Wtime();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Prepare parameters
    if (mpi_rank == 0) {
        block_sizes = malloc(n * sizeof(int));
        displs = malloc((n + 1) * sizeof(int));
        displs[0] = 0;
        for (int i = 0; i < n; i++) {
            displs[i + 1] = n * (i + 1) / mpi_size;
            block_sizes[i] = displs[i + 1] - displs[i];
        }
    }
    int this_start = n * mpi_rank / mpi_size;;
    int this_size = n * (mpi_rank + 1) / mpi_size - this_start;
    int *this_data = malloc(this_size * sizeof(int));

    // Dispatch first batch of jobs
    MPI_Scatterv(root_data, block_sizes, displs, MPI_INT, this_data, this_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Local sort
    quick_sort(this_data, this_size);

    // Sampling
    int *samples = malloc(mpi_size * sizeof(int));
    for (int i = 0; i < mpi_size; i++) {
        samples[i] = this_data[this_size * (i + 1) / (mpi_size + 1)];
    }
    MPI_Gather(samples, mpi_size, MPI_INT, root_samples, mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
    free(samples);

    // Merge samples and select pivots
    int *pivots = malloc(mpi_size * sizeof(int));
    if (mpi_rank == 0) {
        multi_merge_flat(root_samples, mpi_size, mpi_size);
        for (int i = 0; i < mpi_size - 1; i++) {
            pivots[i] = root_samples[mpi_size * (i + 1)];
        }
        free(root_samples);
    }
    MPI_Bcast(pivots, mpi_size - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Partition by pivots
    int *class_index = malloc((mpi_size + 1) * sizeof(int)),
        *rclass_index = malloc((mpi_size + 1) * sizeof(int)),
        *class_sizes = malloc(mpi_size * sizeof(int)),
        *rclass_sizes = malloc(mpi_size * sizeof(int));
    class_index[0] = 0;
    class_index[mpi_size] = this_size;
    for (int i = 0, class_i = 1; i < this_size && class_i < mpi_size; i++) {
        while (this_data[i] >= pivots[class_i - 1]) {
            class_index[class_i] = i;
            class_i++;
            if (class_i >= mpi_size)
                break;
        }
    }
    for (int i = 0; i < mpi_size; i++) {
        class_sizes[i] = class_index[i + 1] - class_index[i];
    }
    MPI_Alltoall(class_sizes, 1, MPI_INT, rclass_sizes, 1, MPI_INT, MPI_COMM_WORLD);
    free(pivots);
    pivots = NULL;

    // Exchange classes of data
    rclass_index[0] = 0;
    for (int i = 0; i < mpi_size; i++) {
        rclass_index[i + 1] = rclass_index[i] + rclass_sizes[i];
    }
    int *rdata = malloc(rclass_index[mpi_size] * sizeof(int));
    MPI_Alltoallv(this_data, class_sizes, class_index, MPI_INT,
                  rdata, rclass_sizes, rclass_index, MPI_INT, MPI_COMM_WORLD);
    free(this_data);
    free(class_index);
    free(class_sizes);
    this_data = rdata;
    this_size = rclass_index[mpi_size];
    class_index = rclass_index;
    class_sizes = rclass_sizes;
    rdata = rclass_index = rclass_sizes = NULL;

    // Merge and gather
    multi_merge(this_data, mpi_size, class_sizes);
    MPI_Gather(&this_size, 1, MPI_INT, block_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < mpi_size; i++) {
            displs[i] = displs[i - 1] + block_sizes[i - 1];
        }
    }
    MPI_Gatherv(this_data, this_size, MPI_INT, root_data, block_sizes, displs, MPI_INT, 0, MPI_COMM_WORLD);
    free(class_index);
    free(class_sizes);
    free(this_data);

    const double endtime = MPI_Wtime();
    MPI_Finalize();

    if (mpi_rank == 0) {
        for (int i = 0; i < n; i++) {
            printf("%d\n", root_data[i]);
        }
        free(root_data);
        free(block_sizes);
        free(displs);

        const char *log_time_file = getenv("LOG_TIME_FILE");
        if (log_time_file != NULL) {
            FILE *fp = fopen(log_time_file, "a");
            fprintf(fp, "%lf\n", endtime - starttime);
            fclose(fp);
        }
    }
    return 0;
}
