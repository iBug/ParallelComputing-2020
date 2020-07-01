CC := gcc
MPICC := mpicc
CFLAGS := -fopenmp -O2 -Wno-unused-result
LDFLAGS := -lm

OMPEXES := 1-openmp
MPIEXES := 1-mpi

.PHONY: all clean

all: $(OMPEXES) $(MPIEXES)

$(OMPEXES): %: %.c
	${CC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

$(MPIEXES): %: %.c
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

clean:
	rm -f $(OMPEXES) $(MPIEXES)
