#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define V_MAX 100
#define P 0.2

typedef struct _Car {
    unsigned v, d;
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
    unsigned n, rounds;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    init_random();

    if (rank == 0) {
        scanf(" %u %u", &n, &rounds);
        n += 1;
    }
    MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rounds, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Construct datatype
    MPI_Datatype MPI_Car;
    {
        int blocklengths[2] = {1, 1};
        MPI_Datatype types[2] = {MPI_UNSIGNED, MPI_UNSIGNED};
        MPI_Aint offsets[2] = {offsetof(Car, v), offsetof(Car, d)};
        MPI_Type_create_struct(2, blocklengths, offsets, types, &MPI_Car);
        MPI_Type_commit(&MPI_Car);
    }

    size_t this_size = n * (rank + 1) / size - n * rank / size;
    Car *car = malloc(this_size * sizeof(Car));
    for (int i = 0; i < this_size; i++) {
        car[i].v = car[i].d = 0U;
    }
    if (rank == 0) {
        car[0].d = V_MAX + 1;
    }

    // Do simulation
    for (unsigned round = 0; round < rounds; round++) {
    }

    // Collect results

    if (rank == 0) {
        printf("%.12lf\n", total);
    }
    MPI_Finalize();
    return 0;
}
