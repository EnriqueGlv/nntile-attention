import numpy as np

# Imports
import nntile
from nntile.tensor import linear_relu_async, gemm_async, relu_forward_async, to_numpy

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

### ez test
def initTensor():
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
    nntile.starpu.wait_for_all()
    print("Tensors initialized !")

    return x_moments

def fillTensor(print_msg="value"):
    x_mom = initTensor()
    value = np.random.rand(n_cols,batch_size) -0.5
    print(f"{print_msg}: ", value)
    x_mom.value.from_array(value)
    return x_mom

np.random.seed(0)
in_r, out_r = fillTensor('input'), initTensor()

# print("in:", to_numpy(in_r.value))
# print("out:", to_numpy(out_r.value))

relu_forward_async(in_r.value, out_r.value)

print("in:", to_numpy(in_r.value))
print("out:", to_numpy(out_r.value))

print("in==out:", np.array_equal(to_numpy(in_r.value), to_numpy(out_r.value)))

in_r.unregister()
out_r.unregister()

# np.random.seed(0)
# in_lr = initTensors()

# print(to_numpy(in_lr.value))

