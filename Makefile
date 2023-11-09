.PHONY: all test

all: test

test:
	nvcc -o knn knn.cpp yourSolution.cu -lcuda

clean:
	rm test