#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define V_MAX 50
#define P 0.2

typedef struct _Car {
    int v, d;
} Car;

void init_random(void) {
    FILE *fp = fopen("/dev/urandom", "rb");
    if (fp == NULL) {
        // Too bad it isn't available, use time
        srand((unsigned)time(NULL));
    } else {
        int x;
        fread(&x, sizeof(x), 1, fp);
        fclose(fp);
        srand(x);
    }
}

int main(int argc, char **argv) {
    int rank, size; // MPI
    int p_cap = (int)(P * RAND_MAX); // rand() <= p_cap
    int n, rounds;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    init_random();

    // Construct datatype
    MPI_Datatype MPI_Car;
    {
        int blocklengths[2] = {1, 1};
        MPI_Datatype types[2] = {MPI_INT, MPI_INT};
        MPI_Aint offsets[2] = {offsetof(Car, v), offsetof(Car, d)};
        MPI_Type_create_struct(2, blocklengths, offsets, types, &MPI_Car);
        MPI_Type_commit(&MPI_Car);
    }

    if (rank == 0) {
        scanf(" %d %d", &n, &rounds);
    }
    const double starttime = MPI_Wtime();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rounds, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int this_start = n * rank / size;
    int this_size = n * (rank + 1) / size - this_start;
    Car *car = malloc(this_size * sizeof(Car));
    for (int i = 0; i < this_size; i++) {
        car[i].v = 0;
        car[i].d = 1;
    }
    if (rank == 0) {
        car[0].d = V_MAX + 1;
    }

    // Do simulation
    for (int round = 0; round < rounds; round++) {
        // Speed change
        for (int i = 0; i < this_size; i++) {
            if (car[i].d - 1 <= car[i].v) {
                car[i].v = car[i].d - 1;
            } else if (car[i].v < V_MAX) {
                car[i].v++;
            }
            if (car[i].v > 0 && rand() <= p_cap) {
                car[i].v--;
            }
        }

        // Distance change
        for (int j = 1; j < this_size; j++) {
            car[j].d += car[j - 1].v - car[j].v;
        }

        // Communicate
        if (rank != size - 1) {
            MPI_Send(&car[this_size - 1].v, 1, MPI_INT, rank + 1, rank, MPI_COMM_WORLD);
        }
        if (rank != 0) {
            int that_v; // car[-1].v
            MPI_Recv(&that_v, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            car[0].d += that_v - car[0].v;
        }
    }

    // Collect results
    int *recvcounts = NULL;
    int *displs = NULL;
    Car *cars = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(*recvcounts));
        displs = malloc(size * sizeof(*displs));
        cars = malloc(n * sizeof(*cars));
    }
    MPI_Gather(&this_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Produce indexes for variable-gathering
    if (rank == 0) {
        displs[0] = 0;
        for (size_t i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }
    MPI_Gatherv(car, this_size, MPI_Car, cars, recvcounts, displs, MPI_Car, 0, MPI_COMM_WORLD);
    MPI_Type_free(&MPI_Car);
    const double endtime = MPI_Wtime();
    MPI_Finalize();

    if (rank == 0) {
        printf("Car %3d: Speed = %3d\n", 0, cars[0].v);
        for (int i = 1; i < n; i++) {
            printf("Car %3d: Speed = %3d, Distance = %3d\n", i, cars[i].v, cars[i].d);
        }
        free(recvcounts);
        free(displs);
        free(cars);
        const char *log_time_file = getenv("LOG_TIME_FILE");
        if (log_time_file != NULL) {
            FILE *fp = fopen(log_time_file, "a");
            fprintf(fp, "%lf\n", endtime - starttime);
            fclose(fp);
        }
    }
    return 0;
}
