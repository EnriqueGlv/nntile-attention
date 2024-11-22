import numpy as np

# Imports
import nntile
from nntile.tensor import linear_relu_async, gemm_async, relu_async

config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.profiling_init()
nntile.starpu.init()
nntile.starpu.restrict_cuda()
nntile.starpu.profiling_enable()

# setup model dimensions
n_cols = 2048
batch_size = 128
n_cols_tile = 2048
batch_size_tile = 128
gemm_ndim = 1
hidden_layer_dim = 128  # Rank of approximation
hidden_layer_dim_tile = 128
nclasses = 100

# Define tensors
x_traits = nntile.tensor.TensorTraits(
    [n_cols, batch_size], [n_cols_tile, batch_size_tile]
)
x_distr = [0] * x_traits.grid.nelems

# Define activation tensor
x = nntile.tensor.Tensor_fp32(x_traits, x_distr, 0)
x_grad = nntile.tensor.Tensor_fp32(x_traits, x_distr, 0)
x_grad_required = False
x_moments = nntile.tensor.TensorMoments(x, x_grad, x_grad_required)
input_value = np.random.rand(n_cols,batch_size)
x.value.from_array(input_value)
print("Tensors initialized !")

y = nntile.tensor.Tensor_fp32(x_traits, x_distr, 0)

# TODO: better test
z = linear_relu_async()

nntile.starpu.wait_for_all()