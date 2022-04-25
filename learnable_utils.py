import numpy as np
from os.path import join, exists
from copy import deepcopy
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from file_utils import *
from jax import grad, jit, lax
import jax.numpy as jnp

def generate_atrous_kernel():
	kernel_weights = np.array([1.0, 2.0 / 3.0, 1.0 / 6.0])

	size = 5
	atrous_kernel = np.zeros((size, size))
	# sum = 0
	for i in range(size):
		for j in range(size):
			ii, jj = abs(i - int(size/2)), abs(j - int(size/2))
			atrous_kernel[i, j] = kernel_weights[ii] * kernel_weights[jj]
			# sum += atrous_kernel[i, j]

	return atrous_kernel


# if __name__ == "__main__":
# 	print(generate_atrous_kernel())