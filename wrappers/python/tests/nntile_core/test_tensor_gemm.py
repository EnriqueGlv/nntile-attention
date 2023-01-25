# All necesary imports
import nntile
import numpy as np
# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}
# Define mapping between tested function and numpy type
gemm = {np.float32: nntile.tensor.gemm_fp32,
        np.float64: nntile.tensor.gemm_fp64}

# Helper function returns bool value true if test passes
def helper(dtype):
    # Describe single-tile tensor, located at node 0
    shape = [2, 2]
    mpi_distr = [0]
    next_tag = 0
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    next_tag = A.next_tag;
    B = Tensor[dtype](traits, mpi_distr, next_tag)
    next_tag = B.next_tag;
    C = Tensor[dtype](traits, mpi_distr, next_tag)
    # Set initial values of tensors
    src_A = np.array(np.random.randn(*shape), dtype=dtype, order='F')
    src_B = np.array(np.random.randn(*shape), dtype=dtype, order='F')
    src_C = np.array(np.random.randn(*shape), dtype=dtype, order='F')
    dst_C = np.zeros_like(src_C)
    A.from_array(src_A)
    B.from_array(src_B)
    C.from_array(src_C)
    # Get results by means of nntile and convert to numpy
    alpha = 1
    beta = -1
    gemm[dtype](alpha, nntile.notrans, A, nntile.trans, B, beta, C, 1)
    C.to_array(dst_C)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    C.unregister()
    # Get result in numpy
    src_C = beta*src_C + alpha*(src_A@(src_B.T))
    # Check if results are almost equal
    return (dst_C == src_C).all()

# Test runner for different precisions
def test():
    for dtype in dtypes:
        assert helper(dtype)

# Repeat tests
def test_repeat():
    for dtype in dtypes:
        assert helper(dtype)

if __name__ == "__main__":
    test()
    test_repeat()

