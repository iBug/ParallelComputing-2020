#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define G 6.67e-11
#define MASS 1e4
#define FRAME 1e-3

typedef struct _Pos {
    double x, y;
} Pos;

inline double distance_2(Pos x, Pos y) {
    return (x.x - y.x) * (x.x - y.x) + (x.y - y.y) * (x.y - y.y);
}
inline double distance(Pos x, Pos y) {
    return sqrt(distance_2(x, y));
}

void compute_force(const Pos *pos, int n, int this, Pos *result) {
    result->x = result->y = 0.0;
    for (int i = 0; i < n; i++) {
        if (i == this)
            continue;
        double r2 = distance_2(pos[i], pos[this]);
        double f = G * MASS * MASS / r2;
        double r = sqrt(r2);
        double df = f / r;
        result->x += (pos[i].x - pos[this].x) * df;
        result->y += (pos[i].y - pos[this].y) * df;
    }
}

void compute_velocity(Pos *vel, const Pos *force) {
    vel->x += force->x / MASS * FRAME;
    vel->y += force->y / MASS * FRAME;
}

void compute_position(Pos *pos, const Pos *vel) {
    pos->x += vel->x * FRAME;
    pos->y += vel->y * FRAME;
}

int main(int argc, char **argv) {
    int rank, size; // MPI
    int n, side;
    double duration;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        scanf(" %u", &n);
        scanf(" %lf", &duration);

        // Make sure N is a square number
        side = (int)round(sqrt(n));
        if (side * side != n) {
            fprintf(stderr, "Error: Expected a square number\n");
            n = 0;
        } else if (duration < 0.0) {
            fprintf(stderr, "Error: Expected a non-negative duration\n");
            n = 0;
        }
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Bcast(&side, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&duration, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Prepare parameters
    int *recvcounts = malloc(n * sizeof(int)), *displs = malloc((n + 1) * sizeof(int));
    displs[0] = 0;
    for (int i = 0; i < n; i++) {
        displs[i + 1] = n * (i + 1) / size;
        recvcounts[i] = displs[i + 1] - displs[i];
    }
    const int this_start = displs[rank];
    const int this_size = recvcounts[rank];

    // Construct datatype
    MPI_Datatype MPI_Pos;
    {
        int blocklengths[2] = {1, 1};
        MPI_Aint offsets[2] = {offsetof(Pos, x), offsetof(Pos, y)};
        MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
        MPI_Type_create_struct(2, blocklengths, offsets, types, &MPI_Pos);
        MPI_Type_commit(&MPI_Pos);
    }

    // Initialize data
    Pos *pos = malloc(n * sizeof(Pos));
    Pos *vel = malloc(this_size * sizeof(Pos));
    for (int i = 0; i < this_size; i++) {
        vel[i].x = vel[i].y = 0.0;
    }
    for (int i = 0; i < n; i++) {
        pos[i].x = 1e-2 * (i % side - (side - 1.0) / 2.0);
        pos[i].y = 1e-2 * (i / side - (side - 1.0) / 2.0);
    }

    // Do simulation
    double runtime = 0.0;
    Pos force;
    while (runtime < duration) {
        for (int i = 0; i < this_size; i++) {
            compute_force(pos, n, this_start + i, &force);
            compute_velocity(&vel[i], &force);
        }
        for (int i = 0; i < this_size; i++) {
            compute_position(&pos[this_start + i], &vel[i]);
        }
        runtime += FRAME;

        // Communicate
        MPI_Allgatherv(MPI_IN_PLACE, this_size, MPI_Pos, pos, recvcounts, displs, MPI_Pos, MPI_COMM_WORLD);
    }

    // Results already collected from allgatherv, so cleanup only
    MPI_Type_free(&MPI_Pos);
    MPI_Finalize();

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            printf("Object %3d: (%.8lf, %.8lf)\n", i + 1, pos[i].x, pos[i].y);
        }
    }
    free(pos);
    free(vel);
    return 0;
}
