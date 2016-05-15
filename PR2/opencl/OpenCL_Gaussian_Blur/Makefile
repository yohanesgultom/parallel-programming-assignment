CC=gcc
CFLAGS=-Iinclude
DEPS = args.h bitmap.h gaussian.h
OBJ = gaussian.c args.c bitmap.c gaussian.c main.c
LIBS=-lm -lOpenCL

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

opencl-tutorial: $(OBJ)
	gcc -o $@ $^ $(CFLAGS) $(LIBS)
