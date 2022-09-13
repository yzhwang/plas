ARCH=\
  -gencode arch=compute_35,code=compute_35 \
  -gencode arch=compute_35,code=sm_35 \
  -gencode arch=compute_61,code=compute_61 \
  -gencode arch=compute_61,code=sm_61 \

OPTIONS=-std=c++11 -O2 -g -Xcompiler="-Werror" -lineinfo  --expt-extended-lambda --expt-relaxed-constexpr -use_fast_math -Xptxas="-v" -I src -I/usr/local/cuda/include/cub

all: \
	test

test: \
  test_is_sorted \
  test_edge_softmax \
  test_scan \
  test_simple \
  test_seg_vector_reduce \
  test_seg_vector_scatter \
  test_seg_vector_reduce_op \
  test_seg_vector_reduce_op_grad \
  test_intersection \
  test_lookup_query

test_is_sorted: test/test_is_sorted.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_edge_softmax: test/test_edge_softmax.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_scan: test/test_scan.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_simple: test/test_simple.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_seg_vector_reduce: test/test_seg_vector_reduce.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_seg_vector_scatter: test/test_seg_vector_scatter.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_seg_vector_reduce_op: test/test_seg_vector_reduce_op.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_seg_vector_reduce_op_grad: test/test_seg_vector_reduce_op_grad.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_intersection: test/test_intersection.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_lookup_query: test/test_lookup_query.cu *.h
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

clean:
	rm test_is_sorted
	rm test_edge_softmax
	rm test_scan
	rm test_simple
	rm test_seg_vector_reduce
	rm test_seg_vector_scatter
	rm test_seg_vector_reduce_op
	rm test_seg_vector_reduce_op_grad
	rm test_intersection
	rm test_lookup_query

