ARCH=\
  -gencode arch=compute_35,code=compute_35 \
  -gencode arch=compute_35,code=sm_35 \
  -gencode arch=compute_61,code=compute_61 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_80,code=compute_80 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=compute_86 \
  -gencode arch=compute_86,code=sm_86

OPTIONS=-std=c++14 -O2 -g -Xcompiler="-Werror" -lineinfo  --expt-extended-lambda --expt-relaxed-constexpr -use_fast_math -Xptxas="-v" -I src -I/usr/local/cuda/include/cub

all: \
	test

test: \
  test_is_sorted \
  test_edge_softmax \
  test_scan \
  test_reduce \
  test_simple \
  test_intersection \
  test_lookup_query \
  test_slice \
  test_concat \
  test_boysf \
  test_matmul \
  test_eri

test_is_sorted: test/test_is_sorted.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_edge_softmax: test/test_edge_softmax.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_scan: test/test_scan.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_reduce: test/test_reduce.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_simple: test/test_simple.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_intersection: test/test_intersection.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_lookup_query: test/test_lookup_query.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_slice: test/test_slice.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_concat: test/test_concat.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_boysf: test/test_boysf.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_matmul: test/test_matmul.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_eri: test/test_eri.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

clean:
	rm test_is_sorted
	rm test_edge_softmax
	rm test_scan
	rm test_reduce
	rm test_simple
	rm test_intersection
	rm test_lookup_query
	rm test_slice
	rm test_concat
	rm test_boysf
	rm test_matmul
	rm test_eri