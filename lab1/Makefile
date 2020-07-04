CC := gcc
MPICC := mpicc
CFLAGS := -O2 -Wno-unused-result
LDFLAGS := -lm

OMPEXES := 1-openmp 2-openmp
MPIEXES := 1-mpi 2-mpi

.PHONY: all clean

all: $(OMPEXES) $(MPIEXES)

$(OMPEXES): %: %.c
	${CC} ${CFLAGS} -fopenmp -o $@ $^ ${LDFLAGS}

$(MPIEXES): %: %.c
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

clean:
	rm -f $(OMPEXES) $(MPIEXES)
