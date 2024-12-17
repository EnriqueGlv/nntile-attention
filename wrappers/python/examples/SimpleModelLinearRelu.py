import sys
import numpy as np
import torch
import torch.nn as nn
import time

import nntile
from nntile.layer.act import Act
from nntile.layer.linear import Linear
from wrappers.python.nntile.layer.fused_linear import FusedLinear
from nntile.model.base_model import BaseModel
from nntile.tensor import TensorMoments, notrans, to_numpy


##################
# Setup SimpleNN #
##################

class SimpleNN(BaseModel):
    next_tag: int

    def __init__(self, x: TensorMoments, side: str, ndim: int,
            add_shape: int, add_basetile_shape: int,
            n_classes: int, next_tag: int, bias: bool = False):

        # Check parameter side
        if side != 'L' and side != 'R':
            raise ValueError("side must be either 'L' or 'R'")
        # Check parameter ndim
        if ndim <= 0:
            raise ValueError("ndim must be positive integer")
        
        # intial activations and list of layers
        activations = [x]
        layers = []

        # Linear layer
        new_layer, next_tag = Linear.generate_simple(x, side, notrans, ndim,
                [add_shape], [add_basetile_shape], next_tag, bias)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # ReLU layer
        new_layer, next_tag = Act.generate_simple(activations[-1], "relu",
                next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # Linear layer
        new_layer, next_tag = Linear.generate_simple(activations[-1],
                side, notrans, 1, [n_classes], [n_classes], next_tag, bias)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        self.next_tag = next_tag
        super().__init__(activations, layers)

    # Randomly init all linear layers
    def init_randn_async(self):
        for l in self.layers:
            # if type(l) is Linear:
            l.init_randn_async()

class FusedNN(BaseModel):
    next_tag: int

    def __init__(self, x: TensorMoments, side: str, ndim: int,
            add_shape: int, add_basetile_shape: int,
            n_classes: int, next_tag: int, bias: bool = False):

        # Check parameter side
        if side != 'L' and side != 'R':
            raise ValueError("side must be either 'L' or 'R'")
        # Check parameter ndim
        if ndim <= 0:
            raise ValueError("ndim must be positive integer")
        
        # intial activations and list of layers
        activations = [x]
        layers = []

        # LinearRelu layer
        new_layer, next_tag = FusedLinear.generate_simple(x, side, notrans, ndim,
                [add_shape], [add_basetile_shape], next_tag, bias)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # ReLU layer
        new_layer, next_tag = Act.generate_simple(activations[-1], "relu",
                next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # Linear layer
        new_layer, next_tag = Linear.generate_simple(activations[-1],
                side, notrans, 1, [n_classes], [n_classes], next_tag, bias)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        self.next_tag = next_tag
        super().__init__(activations, layers)

    # Randomly init all linear layers
    def init_randn_async(self):
        for l in self.layers:
            # if type(l) in [Linear, LinearRelu]:
            l.init_randn_async()


################
# Run SimpleNN #
################

# print(59, "\n", file=sys.stderr)
# Init starpu config and codelets
config = nntile.starpu.Config(-1, -1, -1)
# print(59, "\n", file=sys.stderr)
nntile.starpu.profiling_init()
nntile.starpu.init()
# print(63, file=sys.stderr)
nntile.starpu.restrict_cuda()
nntile.starpu.profiling_enable()

next_tag = 0
print("Starpu intialized !")

# setup model dimensions
n_cols = 2048
batch_size = 128
n_cols_tile = 2048
batch_size_tile = 128
gemm_ndim = 1
hidden_layer_dim = 128  # Rank of approximation
hidden_layer_dim_tile = 128
nclasses = 100

def initTensors():
    global next_tag

    # Define tensors
    x_traits = nntile.tensor.TensorTraits(
        [n_cols, batch_size], [n_cols_tile, batch_size_tile]
    )
    x_distr = [0] * x_traits.grid.nelems

    # Define activation tensor
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
    next_tag = x.next_tag
    x_grad = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
    next_tag = x_grad.next_tag
    x_grad_required = False
    x_moments = nntile.tensor.TensorMoments(x, x_grad, x_grad_required)
    nntile.starpu.wait_for_all()
    print("Tensors initialized !")

    return x_moments

def benchNN(nn_model):
    # setup input
    # copy_async(x_minibatch, self.model.activations[0].value)
    # simpNN.clear_parameters_grads()
    # simpNN.clear_activations_grads()
    input_value = 100 * np.random.rand(n_cols,batch_size)
    print("input: ", input_value)
    nn_model.activations[0].value.from_array(input_value)
    print("input val:", to_numpy(nn_model.activations[0].value))
    print("Input intialized !")

    # Init weights
    nn_model.init_randn_async()
    print(nn_model.parameters)
    for p in nn_model.parameters:
        print("param val:", to_numpy(p.value))
    print("Weights initialized !")


    # for _ in range(100):
    # Start timer and run forward pass
    time0 = -time.time()
    # nntile.starpu.profiling_enable()
    nn_model.forward_async()
    time0 += time.time()
    print("Finish adding tasks in {} seconds".format(time0))

    # Wait for all computations to finish
    time0 = -time.time()
    nntile.starpu.wait_for_all()
    time0 += time.time()
    print("Forward pass finished in {} seconds".format(time0))

    # time.sleep(1)
    # nntile.starpu.add_trace_event("end of forward")

    # time0 = -time.time()
    # simpNN.backward_async()
    # time0 += time.time()
    # print("Finish adding tasks in {} seconds".format(time0))

    # # Wait for all computations to finish
    # time0 = -time.time()
    # nntile.starpu.wait_for_all()
    # time0 += time.time()
    # print("Backward pass finished in {} seconds".format(time0))

### setup numpy

# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)


### SimpNN

np.random.seed(0)
simpnn_x_moments = initTensors()

# Create model
simpNN = SimpleNN(
    simpnn_x_moments,
    "R",
    gemm_ndim,
    hidden_layer_dim,
    hidden_layer_dim_tile,
    nclasses,
    next_tag
)
next_tag = simpNN.next_tag
print("simpNN intialized !")

print("--- simpleNN perf ---")

benchNN(simpNN)

print("simpNN output:")

# t0 = to_numpy(simpNN.activations[0].value)
# print("act[0]:", t0)
t1 = to_numpy(simpNN.activations[1].value)
print("act[1]:", t1)
t12 = to_numpy(simpNN.activations[2].value)
print("act[2]:", t12)
# t13 = to_numpy(simpNN.activations[3].value)
# print("act[3]:", t13)

### Fused NN

np.random.seed(0)
fusenn_x_moments = initTensors()
next_tag=0

# Create model
fuseNN = FusedNN(
    fusenn_x_moments,
    "R",
    gemm_ndim,
    hidden_layer_dim,
    hidden_layer_dim_tile,
    nclasses,
    next_tag
)
next_tag = fuseNN.next_tag
print("fusedNN intialized !")

print("--- fusedNN perf ---")

benchNN(fuseNN)

print("fusedNN output:")

t2 = to_numpy(fuseNN.activations[1].value)
print("act[1]:", t2)

# t22 = to_numpy(fuseNN.activations[1].value)
# print("act[2]:", t22)

# t2 = to_numpy(fuseNN.activations[1].value)
# print("act[1]:", t2)

print("--- Comparison ---")

print("output equals:")

print(t12 == t2)
print("np equal:", np.array_equal(t12, t2))
print("t1 equal 0:", np.array_equal(t1, np.zeros_like(t1)))
print("t12 equal 0:", np.array_equal(t12, np.zeros_like(t1)))
print("t2 equal 0:", np.array_equal(t2, np.zeros_like(t2)))
print("t1[-1] equal 0:", np.array_equal(to_numpy(simpNN.activations[-1].value), np.zeros_like(t2)))

print("t1[-1]", to_numpy(simpNN.activations[-1].value))
print("t2[-1]", to_numpy(fuseNN.activations[-1].value))

# unregister model and tensors
nntile.starpu.profiling_disable()
simpnn_x_moments.unregister()
fusenn_x_moments.unregister()
simpNN.unregister()
fuseNN.unregister()
