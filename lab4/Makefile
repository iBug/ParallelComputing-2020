CC := gcc
MPICC := mpicc
CFLAGS := -O2 -Wall -Wno-unused-result
LDFLAGS :=
EXES := main qsort

.PHONY: all clean

all: $(EXES)

main: main.c
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

qsort: qsort.c
	${CC} ${CFLAGS} -fopenmp -o $@ $^ ${LDFLAGS}

clean:
	rm -f $(EXES) *.o
