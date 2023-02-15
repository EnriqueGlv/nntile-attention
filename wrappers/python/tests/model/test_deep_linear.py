# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/deep_linear.py
# Example of using Deep Linear network of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-15

# Imports
import nntile
import numpy as np
import time

# Set up StarPU+MPI and init codelets
config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
next_tag = 0

# Define matrix A as a Hilbert matrix
n_rows = 128
n_cols = 128
n_batches = 10
batch_size = 128
n_cols_tile = 128
n_rows_tile = 128
batch_size_tile = 128
gemm_ndim = 1
hidden_layer_dim = 128
hidden_layer_dim_tile = 128
nlayers = 2
A = np.zeros((n_rows, n_cols), order='F', dtype=np.float32)
for i in range(n_rows):
    for j in range(n_cols):
        A[i, j] = 1.0 / (i+j+1)

# Define batches of X and Y
batch_input = []
batch_output = []
x_traits_full = nntile.tensor.TensorTraits([n_cols, batch_size], [n_cols,
    batch_size])
x_traits = nntile.tensor.TensorTraits([n_cols, batch_size], [n_cols_tile,
    batch_size_tile])
x_distr = [0] * x_traits.grid.nelems
y_traits_full = nntile.tensor.TensorTraits([n_rows, batch_size], [n_rows,
    batch_size])
y_traits = nntile.tensor.TensorTraits([n_rows, batch_size], [n_rows_tile,
    batch_size_tile])
y_distr = [0] * y_traits.grid.nelems
np.random.seed(0)
for i in range(n_batches):
    X = np.random.randn(n_cols, batch_size)
    Y = A @ X
    x_full = nntile.tensor.Tensor_fp32(x_traits_full, [0], next_tag)
    next_tag = x_full.next_tag
    nntile.starpu.wait_for_all()
    x_full.from_array(X)
    nntile.starpu.wait_for_all()
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
    next_tag = x.next_tag
    nntile.nntile_core.tensor.scatter_fp32(x_full, x)
    x_full.unregister()
    del x_full
    batch_input.append(x)
    y_full = nntile.tensor.Tensor_fp32(y_traits_full, [0], next_tag)
    next_tag = y_full.next_tag
    nntile.starpu.wait_for_all()
    y_full.from_array(Y)
    nntile.starpu.wait_for_all()
    y = nntile.tensor.Tensor_fp32(y_traits, y_distr, next_tag)
    next_tag = y.next_tag
    nntile.nntile_core.tensor.scatter_fp32(y_full, y)
    y_full.unregister()
    del y_full
    batch_output.append(y)

# Define tensor X for input batches
x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
next_tag = x.next_tag
x_grad = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
next_tag = x_grad.next_tag
x_grad_required = False
x_moments = nntile.tensor.TensorMoments(x, x_grad, x_grad_required)

# Define deep linear network
m = nntile.model.DeepLinear(x_moments, 'R', gemm_ndim, hidden_layer_dim,
        hidden_layer_dim_tile, nlayers, next_tag)
next_tag = m.next_tag

# Randomly init weights of deep linear network
m.init_randn_async()

# Set up Frobenius loss function for the model
frob, next_tag = nntile.loss.Frob.generate_simple(m.activations[-1], next_tag)

# Set up training pipeline
n_epochs = 1
lr = -1e-9
pipeline = nntile.pipeline.Pipeline(batch_input, batch_output, m, None, frob,
        n_epochs, lr)

# Wait for all computations to finish
nntile.starpu.wait_for_all()

# Start timer and run training
time0 = -time.time()
pipeline.train_async()

# Wait for all computations to finish
nntile.starpu.wait_for_all()
time0 += time.time()
print("Done in {} seconds".format(time0))
np_val = np.array([1], order='F', dtype=np.float32)
np_val[0] = 0
frob.val.to_array(np_val)
nntile.starpu.wait_for_all()
print("Loss is {}".format(np_val[0]))
print("Norm is {}".format(np.linalg.norm(A, 'fro')))

## Check a single test
#X = np.eye(n_rows, dtype=np.float32, order='F')
#Y = A
#x_full = nntile.tensor.Tensor_fp32(x_traits_full, [0], next_tag)
#next_tag = x_full.next_tag
#nntile.starpu.wait_for_all()
#x_full.from_array(X)
#nntile.starpu.wait_for_all()
#x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
#next_tag = x.next_tag
#nntile.nntile_core.tensor.scatter_fp32(x_full, x)
#x_full.unregister()
#del x_full
#nntile.tensor.copy_async(x, m.activations[0].value)
#m.forward_async()
#nntile.starpu.wait_for_all()
#Y2 = np.zeros_like(Y, order='F', dtype=np.float32)
#m.activations[-1].value.to_array(Y2)

## Unregister all tensors related to model
#m.unregister()
#
## Unregister loss function
#frob.y.unregister()
#frob.val.unregister()
#frob.tmp.unregister()
#
## Unregister input/output batches
#for x in batch_input:
#    x.unregister()
#for x in batch_output:
#    x.unregister()
