CC := gcc
CFLAGS := -fopenmp -O2
LDFLAGS := -lm

EXES := 1-openmp

.PHONY: all clean

all: $(EXES)

$(EXES): %: %.c
	${CC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

clean:
	rm -f $(EXES)
