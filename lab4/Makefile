CC := gcc
MPICC := mpicc
CFLAGS := -O2 -Wall -Wno-unused-result -fsanitize=address -g
LDFLAGS := -lm
EXES := main

.PHONY: all clean

all: $(EXES)

$(EXES): %: %.c
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

clean:
	rm -f $(EXES)
