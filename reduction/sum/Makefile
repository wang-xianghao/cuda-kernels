COMMON_INCLUDE_DIR = ../../include
NVCC = nvcc
NVCC_FLAGS = -O2 -lineinfo -arch=native -I$(COMMON_INCLUDE_DIR)
NCU = ncu
NCU_FLAGS = -f --import-source yes --set full --kernel-id ::sum_kernel:2
NCU_FLAGS_WARP = -f --import-source yes --set full --kernel-id ::sum_kernel:2

TARGET = sum
TARGET_WARP = sum_warp
SRC = sum.cu
SRC_WARP = sum_warp.cu

all: $(TARGET) $(TARGET_WARP)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TARGET_WARP): $(SRC_WARP)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

profile: $(TARGET)
	$(NCU) $(NCU_FLAGS) -o $< $

profile_warp: $(TARGET_WARP)
	$(NCU) $(NCU_FLAGS_WARP) -o $< $

profile_all: profile profile_warp

clean:
	rm -f $(TARGET) $(TARGET_WARP)
	rm -rf $(TARGET).ncu-rep $(TARGET_WARP).ncu-rep
	rm -f *.o