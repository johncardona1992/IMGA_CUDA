build: IMGA.cu
	nvcc --std=c++11 -g -G IMGA.cu -o IMGA -run

clean:
	rm -f IMGA

profile:
	nsys profile --stats=true ./IMGA