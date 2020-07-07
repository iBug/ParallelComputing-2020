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

    if (rank == 0) {
        scanf(" %u %u", &n, &rounds);
        n += 1;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rounds, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Construct datatype
    MPI_Datatype MPI_Car;
    {
        int blocklengths[2] = {1, 1};
        MPI_Datatype types[2] = {MPI_INT, MPI_INT};
        MPI_Aint offsets[2] = {offsetof(Car, v), offsetof(Car, d)};
        MPI_Type_create_struct(2, blocklengths, offsets, types, &MPI_Car);
        MPI_Type_commit(&MPI_Car);
    }

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
            if (rand() <= p_cap && car[i].v > 0) {
                car[i].v--;
            }
        }

        // Distance change
        for (int j = 1; j < this_size; j++) {
            car[j].d += car[j - 1].v - car[j].v;
        }

        // Communicate
        if (rank != size - 1) {
            MPI_Send(&car[0].v, 1, MPI_INT, rank + 1, rank, MPI_COMM_WORLD);
        }
        if (rank != 0) {
            unsigned that_v; // car[-1].v
            MPI_Status status;
            MPI_Recv(&that_v, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, &status);
            car[0].d += that_v - car[0].v;
        }
    }

    // Collect results
    int *these_sizes = NULL;
    unsigned *indexes = NULL;
    Car *cars = NULL;
    if (rank == 0) {
        these_sizes = malloc(size * sizeof(*these_sizes));
        indexes = malloc(size * sizeof(*indexes));
        cars = malloc(n * sizeof(*cars));
    }
    MPI_Gather(&this_size, 1, MPI_INT, these_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Produce indexes for variable-gathering
    if (rank == 0) {
        indexes[0] = 0;
        for (size_t i = 1; i < size; i++) {
            indexes[i] = 1 + indexes[i - 1] + these_sizes[i - 1];
        }
    }
    MPI_Gatherv(car, this_size, MPI_Car, cars, these_sizes, indexes, MPI_Car, 0, MPI_COMM_WORLD);
    MPI_Type_free(&MPI_Car);
    MPI_Finalize();

    if (rank == 0) {
        printf("Car %3d: Speed = %3d\n", 0, cars[0].v);
        for (int i = 1; i < this_size; i++) {
            printf("Car %3d: Speed = %3d, Distance = %3d\n", i, cars[i].v, cars[i].d);
        }
        free(these_sizes);
        free(indexes);
        free(cars);
    }
    return 0;
}
