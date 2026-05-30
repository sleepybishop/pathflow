CFLAGS = -std=c99 -Ide -O0 -g -D_DEFAULT_SOURCE -Wall -Wextra
LDLIBS = -lm

all: libpathflow.a pathflow

pathflow.o: pathflow.c pathflow.h
	$(CC) $(CFLAGS) -c -o $@ $<

libpathflow.a: pathflow.o
	$(AR) rcs $@ $^

pathflow: cli.c libpathflow.a
	$(CC) $(CFLAGS) -o $@ cli.c libpathflow.a $(LDLIBS)

demo: demo.c libpathflow.a
	$(CC) $(CFLAGS) -o $@ demo.c libpathflow.a $(LDLIBS) -lncurses

check: all
	prove -I. t/

valgrind: clean all
	valgrind --error-exitcode=2 --leak-check=full ./pathflow

.PHONY: clean scan indent test check valgrind

clean:
	$(RM) *.o *.a pathflow demo

scan:
	scan-build $(MAKE) clean all

indent:
	clang-format -style="{BasedOnStyle: LLVM, IndentWidth: 4}" -i *.c *.h
