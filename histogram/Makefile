COMMON_INCLUDE_DIR = ../include
NVCC = nvcc
NVCC_FLAGS = -O2 -lineinfo -arch=native -I$(COMMON_INCLUDE_DIR)

TARGET = histogram
SRC = histogram.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
	rm -f *.o