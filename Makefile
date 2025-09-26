CFLAGS = -std=c99 -Iamoeba -O0 -g -D_DEFAULT_SOURCE
LDLIBS = -lm 

all: pathflow 

pathflow: pathflow.c

.PHONY: clean scan indent

clean:
	$(RM) *.o *.a pathflow

scan:
	scan-build $(MAKE) clean all

indent:
	clang-format -style=LLVM -i *.c 


