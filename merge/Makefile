COMMON_INCLUDE_DIR = ../include
NVCC = nvcc
NVCC_FLAGS = -O2 -lineinfo -arch=native -I$(COMMON_INCLUDE_DIR)
NCU = ncu
NCU_FLAGS = -f --import-source yes --set full --kernel-id ::merge_kernel:2

TARGET = merge
SRC = merge.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

profile: $(TARGET)
	$(NCU) $(NCU_FLAGS) -o $< $<	

clean:
	rm -f $(TARGET)
	rm -rf $(TARGET).ncu-rep
	rm -f *.o