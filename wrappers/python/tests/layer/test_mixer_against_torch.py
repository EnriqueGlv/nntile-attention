# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_mixer_against_torch.py
# Test for nntile.layer.mixer
#
# @version 1.0.0

# All necesary imports
import nntile
import numpy as np
import torch.nn.functional as F
import torch
from nntile.torch_models.mlp_mixer import Mixer as TorchMixerLayer


def image_patching(image, patch_size):
    c, h, w = image.shape
    if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError("patch size should be divisor of both image height and width")
    n_patches = int(h * w / (patch_size ** 2))
    n_channels = c * (patch_size ** 2)
    patched_batch = torch.empty((n_patches, n_channels), dtype=image.dtype)

    n_y = int(w / patch_size)

    for i in range(n_patches):
        x = i // n_y
        y = i % n_y

        for clr in range(c):
            vect_patch = image[clr, x * patch_size: (x+1) * patch_size , y * patch_size: (y+1) * patch_size].flatten()
            patched_batch[i, clr * (patch_size ** 2) : (clr+1) * (patch_size ** 2)] = vect_patch
    return patched_batch

# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}
# Get mixer layer
Mixer = nntile.layer.Mixer

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
    if dtype == np.float32:
        tol = 1e-5
    elif dtype == np.float64:
        tol = 1e-10
    
    minibatch_size = 8
    clr_space_size = 3
    h, w = 8, 8
    patch_size = 4

    n_patches = int(h * w / (patch_size ** 2))
    n_channels = clr_space_size * (patch_size ** 2)

    # Set initial values of tensors
    patched_batch = torch.empty((n_patches, minibatch_size, n_channels), dtype=torch.float32)
    rand_A = torch.rand(minibatch_size, clr_space_size, h, w)
    for k in range(minibatch_size):
        patched_batch[:, k, :] = image_patching(rand_A[k, :, :, :], patch_size)
    np_A = np.array(patched_batch, dtype=dtype, order='F')

    # Describe single-tile tensor, located at node 0
    A_shape = patched_batch.shape
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0

    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
   
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)

    # Define mixer layer
    mixer_layer, next_tag = Mixer.generate_simple(A_moments, next_tag)
    
    rand_W1 = np.random.randn(*mixer_layer.mlp_1.linear_1.w.value.shape)
    np_W1 = np.array(rand_W1, dtype=dtype, order='F')
    mixer_layer.mlp_1.linear_1.w.value.from_array(np_W1)

    rand_W2 = np.random.randn(*mixer_layer.mlp_1.linear_2.w.value.shape)
    np_W2 = np.array(rand_W2, dtype=dtype, order='F')
    mixer_layer.mlp_1.linear_2.w.value.from_array(np_W2)

    rand_W3 = np.random.randn(*mixer_layer.mlp_2.linear_1.w.value.shape)
    np_W3 = np.array(rand_W3, dtype=dtype, order='F')
    mixer_layer.mlp_2.linear_1.w.value.from_array(np_W3)

    rand_W4 = np.random.randn(*mixer_layer.mlp_2.linear_2.w.value.shape)
    np_W4 = np.array(rand_W4, dtype=dtype, order='F')
    mixer_layer.mlp_2.linear_2.w.value.from_array(np_W4)

    rand_gamma = np.random.randn(n_channels)
    np_gamma = np.array(rand_gamma, dtype=dtype, order='F')
    rand_beta = np.random.randn(n_channels)
    np_beta = np.array(rand_beta, dtype=dtype, order='F')

    mixer_layer.norm_1.gamma.value.from_array(np_gamma)
    mixer_layer.norm_1.beta.value.from_array(np_beta)

    mixer_layer.norm_2.gamma.value.from_array(np_gamma)
    mixer_layer.norm_2.beta.value.from_array(np_beta)
    A.from_array(np_A)
    
    mixer_layer.clear_gradients()
    mixer_layer.forward_async()
    nntile.starpu.wait_for_all()

    np_Y2 = np.zeros(mixer_layer.y.value.shape, dtype=dtype, order="F")
    mixer_layer.y.value.to_array(np_Y2)
    fro_loss, next_tag = nntile.loss.Frob.generate_simple(mixer_layer.y, next_tag)
    np_zero = np.zeros(mixer_layer.y.value.shape, dtype=dtype, order="F")
    fro_loss.y.from_array(np_zero)
    fro_loss.calc_async()

    mixer_layer.backward_async()
    nntile.starpu.wait_for_all()
    
    torch_mixer_layer = TorchMixerLayer(n_patches, n_channels)
    torch_mixer_layer.set_weight_parameters(np_W1, np_W2, np_W3, np_W4)
    torch_mixer_layer.set_normalization_parameters(np_gamma, np_beta, np_gamma, np_beta)
    torch_mixer_layer.zero_grad()
    torch_output = torch_mixer_layer.forward(torch.from_numpy(np_A))

    torch_loss = 0.5 * torch.sum(torch.square(torch_output))
    torch_loss.backward()

    np_Y = np.array(torch_output.detach().numpy(), order="F", dtype=dtype)

    if np.linalg.norm(np_Y-np_Y2)/np.linalg.norm(np_Y) > tol:
        A_moments.unregister()
        mixer_layer.unregister()
        fro_loss.unregister()
        return False 

    for p_nntile, p_torch in zip(mixer_layer.parameters, torch_mixer_layer.parameters()):
        p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", dtype=dtype)
        p_nntile.grad.to_array(p_nntile_grad_np)
        if p_torch.grad.shape[0] != p_nntile_grad_np.shape[0]:
            p_nntile_grad_np = np.transpose(p_nntile_grad_np)
        rel_error = torch.norm(p_torch.grad - torch.from_numpy(p_nntile_grad_np)) / torch.norm(p_torch.grad)
        # print("Relative error in gradient in layer {} = {}".format(i, rel_error.item()))
        if rel_error > tol:
            A_moments.unregister()
            mixer_layer.unregister()
            fro_loss.unregister()
            return False

    A_moments.unregister()
    mixer_layer.unregister()
    fro_loss.unregister()
    print("Test successful")
    assert True


# Test runner for different precisions
def test():
    for dtype in dtypes:
        helper(dtype)

# Repeat tests
def test_repeat():
    for dtype in dtypes:
        helper(dtype)

if __name__ == "__main__":
    test()
    test_repeat()
