CC = gcc
NVCC = nvcc
CFLAGS = -O3 -Wall -g
LDFLAGS = -lm -L/usr/local/cuda-7.5/lib64 -lcudart

all: c63enc c63dec c63pred

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o c63.h common.o enc_me.o me_block.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
c63dec: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
c63pred: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
	$(CC) -DC63_PRED $(CFLAGS) $^ -o $@ $(LDFLAGS)

me_block.o:
	$(NVCC) -c me_block.cu
clean:
	rm -f *.o c63enc c63dec c63pred
