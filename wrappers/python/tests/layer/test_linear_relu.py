# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_linear.py
# Test for nntile.layer.linear
#
# @version 1.1.0

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import nntile
import nntile.utils.constructors as nntc
from nntile.layer import LinearRelu
from nntile.tensor import to_numpy, linear_relu_async, transpose_async

# Define mapping between numpy and nntile types
Tensor = {
    np.float32: nntile.tensor.Tensor_fp32,
    np.float64: nntile.tensor.Tensor_fp64,
}

config = nntile.starpu.Config(1, 1, 1)
nntile.starpu.init()
nntile.starpu.restrict_cuda()

@pytest.mark.parametrize('dtype', [np.float32])
@pytest.mark.parametrize('side', ['L', 'R'])
# @pytest.mark.parametrize('n_batch', [0]) # Batch gemm untested for now on because not used in GPT
@pytest.mark.parametrize('n_tiles', [1, 2])
def test_linrelu_relu(side: str, dtype: np.dtype, n_tiles: bool):
    # Describe single-tile tensor, located at node 0
    A_shape = [4, 6, 8]
    A_tiles = [int(s/n_tiles) for s in A_shape]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_tiles)
    mpi_distr = [0]*A_traits.grid.nelems
    next_tag = 0
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    layer, next_tag = LinearRelu.generate_simple(
        A_moments, side, nntile.tensor.notrans, 2, [7, 8], [7, 8], next_tag,
        bias=False, batch_ndim=0, act="relu")
    rand_W = rng.standard_normal(layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    nntile.tensor.clear_async(layer.w.grad)

    # Check result of forward pass layer.y.value
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    layer.forward_async()
    nntile.starpu.wait_for_all()

    match side:
        case 'L':
            np_Y = np.tensordot(np_A, np_W, 2)
        case 'R':
            np_Y = np.tensordot(np_W, np_A, 2)

    np_Y = F.relu(torch.from_numpy(np_Y)).numpy()

    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)

    # print("np:", np_Y[0])
    # print("nntile:", np_Y2[0])

    assert np.linalg.norm(np_Y - np_Y2) / np.linalg.norm(np_Y) <= 1e-5

    # Check results of backward pass layer.w.grad and layer.x.grad
    # layer.y.grad.from_array(np_Y)
    # layer.backward_async()

    # match side:
    #     case 'L':
    #         np_Z = np.einsum("ijk,ilm->jklm", np_A, np_Y2)
    #     case 'R':
    #         np_Z = np.einsum("ijk,lmk->ijlm", np_Y2, np_A)
    # np_Z2 = np.zeros_like(np_Z, order='F')
    # layer.w.grad.to_array(np_Z2)
    # assert np.linalg.norm(np_Z - np_Z2) / np.linalg.norm(np_Z) <= 1e-5

    # match side:
    #     case 'L':
    #         np_Z3 = np.einsum("ijk,lmjk->ilm", np_Y2, np_W)
    #     case 'R':
    #         np_Z3 = np.einsum("ijkl,ijm->klm", np_W, np_Y2)
    # np_Z4 = np.zeros_like(np_Z3, order='F')
    # layer.x.grad.to_array(np_Z4)
    # assert np.linalg.norm(np_Z3 - np_Z4) / np.linalg.norm(np_Z3) < 1e-5

    A_moments.unregister()
    layer.unregister()

@pytest.mark.parametrize('dtype', [np.float32])
@pytest.mark.parametrize('side', ['L', 'R'])
# @pytest.mark.parametrize('n_batch', [0]) # Batch gemm untested for now on because not used in GPT
@pytest.mark.parametrize('n_tiles', [1, 2])
def test_linrelu_gelu(side: str, dtype: np.dtype, n_tiles: bool):
    # Describe single-tile tensor, located at node 0
    A_shape = [4, 6, 8]
    A_tiles = [int(s/n_tiles) for s in A_shape]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_tiles)
    mpi_distr = [0]*A_traits.grid.nelems
    next_tag = 0
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    layer, next_tag = LinearRelu.generate_simple(
        A_moments, side, nntile.tensor.notrans, 2, [7, 8], [7, 8], next_tag,
        bias=False, batch_ndim=0, act="gelutanh")
    rand_W = rng.standard_normal(layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    nntile.tensor.clear_async(layer.w.grad)

    # Check result of forward pass layer.y.value
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    layer.forward_async()
    nntile.starpu.wait_for_all()

    match side:
        case 'L':
            np_Y = np.tensordot(np_A, np_W, 2)
        case 'R':
            np_Y = np.tensordot(np_W, np_A, 2)

    np_Y = F.gelu(torch.from_numpy(np_Y), approximate="tanh").numpy()

    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)

    assert np.linalg.norm(np_Y - np_Y2) / np.linalg.norm(np_Y) <= 1e-5

    A_moments.unregister()
    layer.unregister()


@pytest.mark.parametrize('dtype', [np.float32])
@pytest.mark.parametrize('n_tiles', [1, 2])
def test_linrelu_bias(dtype: np.dtype, n_tiles: bool):
    # Describe single-tile tensor, located at node 0
    A_shape = [4, 6, 8]
    A_tiles = [int(s/n_tiles) for s in A_shape]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_tiles)
    mpi_distr = [0]*A_traits.grid.nelems
    next_tag = 0
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    layer, next_tag = LinearRelu.generate_simple(
        A_moments, "R", nntile.tensor.notrans, 1, [7], [7], next_tag,
        bias=True, batch_ndim=0, act="relu")

    # init weights matrix
    rand_W = rng.standard_normal(layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    nntile.tensor.clear_async(layer.w.grad)

     # init bias matrix
    rand_B = rng.standard_normal(layer.b.value.shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    layer.b.value.from_array(np_B)
    nntile.tensor.clear_async(layer.b.grad)

    # Check result of forward pass layer.y.value
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    layer.forward_async()
    nntile.starpu.wait_for_all()

    np_Y = np.tensordot(np_W, np_A, 1)
    np_Y += np_B[:,np.newaxis,np.newaxis]
    np_Y = F.relu(torch.from_numpy(np_Y)).numpy()

    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)

    # print("np:", np_Y[0])
    # print("nntile:", np_Y2[0])
    # print("bias:", np_B)

    assert np.linalg.norm(np_Y - np_Y2) / np.linalg.norm(np_Y) <= 1e-5

    A_moments.unregister()
    layer.unregister()

@pytest.mark.parametrize('dtype', [np.float32])
@pytest.mark.parametrize('n_tiles', [1, 2])
def test_linrelu_gelutanh_bias(dtype: np.dtype, n_tiles: bool):
    # Describe single-tile tensor, located at node 0
    A_shape = [4, 6, 8]
    A_tiles = [int(s/n_tiles) for s in A_shape]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_tiles)
    mpi_distr = [0]*A_traits.grid.nelems
    next_tag = 0
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    layer, next_tag = LinearRelu.generate_simple(
        A_moments, "R", nntile.tensor.notrans, 1, [7], [7], next_tag,
        bias=True, batch_ndim=0, act="gelutanh")

    # init weights matrix
    rand_W = rng.standard_normal(layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    nntile.tensor.clear_async(layer.w.grad)

     # init bias matrix
    rand_B = rng.standard_normal(layer.b.value.shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    layer.b.value.from_array(np_B)
    nntile.tensor.clear_async(layer.b.grad)

    # Check result of forward pass layer.y.value
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    layer.forward_async()
    nntile.starpu.wait_for_all()

    np_Y = np.tensordot(np_W, np_A, 1)
    np_Y += np_B[:,np.newaxis,np.newaxis]
    np_Y = F.gelu(torch.from_numpy(np_Y), approximate="tanh").numpy()

    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)

    # print("np:", np_Y[0])
    # print("nntile:", np_Y2[0])
    # print("bias:", np_B)

    assert np.linalg.norm(np_Y - np_Y2) / np.linalg.norm(np_Y) <= 1e-5

    A_moments.unregister()
    layer.unregister()


@pytest.mark.parametrize('dtype', [np.float32])
@pytest.mark.parametrize('n_tiles', [1, 2])
def test_linrelu_reshape(dtype: np.dtype, n_tiles: bool):
    # Describe single-tile tensor, located at node 0
    A_shape = [4, 6, 8]
    A_tiles = [int(s/n_tiles) for s in A_shape]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_tiles)
    mpi_distr = [0]*A_traits.grid.nelems
    next_tag = 0
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define layer (only used for weights allocation)
    layer, next_tag = LinearRelu.generate_simple(
        A_moments, "R", nntile.tensor.notrans, 1, [7], [7], next_tag,
        bias=True, batch_ndim=0, act="gelutanh")

    # init weights matrix
    rand_W = rng.standard_normal([7,4])
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    # nntile.tensor.clear_async(layer.w.grad)

     # init bias matrix
    # rand_B = rng.standard_normal(layer.b.value.shape)
    # np_B = np.array(rand_B, dtype=dtype, order='F')
    # layer.b.value.from_array(np_B)
    # nntile.tensor.clear_async(layer.b.grad)
    
    # init output
    Y_shape = [6, 8, 7]
    Y_tiles = [int(s/n_tiles) for s in Y_shape[:-1]] + [Y_shape[-1]]
    Y_traits = nntile.tensor.TensorTraits(Y_shape, Y_tiles)
    mpi_distr = [0]*Y_traits.grid.nelems
    next_tag = 0
    # Tensor objects
    Y = Tensor[dtype](Y_traits, mpi_distr, next_tag)
    nntile.tensor.clear_async(Y)

    # Check result of forward pass layer.y.value
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    # layer.forward_async()
    linear_relu_async(1.0, nntile.tensor.notrans, layer.w.value, layer.trans_x,
                        layer.x.value, 0.0, layer.y.value, layer.ndim, layer.batch_ndim,
                        redux=layer.redux, act=layer.act, D=Y, reshape_ndim=1)
    
    # transpose_async(1.0,layer.y.value,Y,1)
    nntile.starpu.wait_for_all()


    np_Y = np.tensordot(np_W, np_A, 1)
    np_Y3 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y3)
    # np_Y += np_B[:,np.newaxis,np.newaxis]
    np_Y = F.gelu(torch.from_numpy(np_Y), approximate="tanh").numpy()
    np_Y = np.transpose(np_Y, axes=[1,2,0])

    np_Y2 = np.zeros_like(np_Y, order='F')
    Y.to_array(np_Y2)

    # print("np:", np_Y[0])
    # print("to_np:", to_numpy(Y))
    # print("nntile:", np_Y2[0])
    # print("post-gemm:", np_Y3[0])
    # print("bias:", np_B)

    assert np.linalg.norm(np_Y - np_Y2) / np.linalg.norm(np_Y) <= 1e-5

    A_moments.unregister()
    layer.unregister()
