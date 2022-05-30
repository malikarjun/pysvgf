import numpy as np
from jax import grad, jit, lax, vmap
import jax.scipy as jsp
import jax.numpy as jnp


def test_reprojected_depth(z1, z2, dz):
	z_diff = abs(z1 - z2)
	return z_diff < 2.0 * (dz + 1e-3)

# vectorized
def test_reprojected_normal_vec(n1, n2):
	return jnp.sum(n1 * n2, axis=2) > 0.9

def test_reprojected_normal(n1, n2):
	return jnp.sum(n1 * n2) > 0.9

def ddy(buffer, x, y):
	return max(abs(buffer[y, x] - buffer[y-1, x]), abs(buffer[y, x] - buffer[y+1, x]))


def ddx(buffer, x, y):
	return max(abs(buffer[y, x] - buffer[y, x+1]), abs(buffer[y, x] - buffer[y, x+1]))


def generate_atrous_filter():
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

def generate_box_filter(radius=1):
	size = 2*radius + 1
	return np.ones((size, size))

def luminance(rgb):
	return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def gaussian_filter(img):
	x = jnp.linspace(-3, 3, 7)
	window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
	return jsp.signal.convolve(img, window, mode='same')

def luminance_vec(img):
	return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]


def luminance_vec_1d(img):
	return 0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2]


def luminance(rgb):
	return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


# only works for square images
# check out this answer for generic impl https://stackoverflow.com/a/44230705
'''
For n = 2, return out as the following array
out[:, :, 0]
|0|0|
|1|1|

out[:, :, 1]
|0|1|
|0|1|

out is further reshaped so that it can be passed to vmap
'''
def indices_array(n):
	r = np.arange(n)
	out = np.empty((n, n, 2), dtype=int)
	out[:, :, 0] = r[:, None]
	out[:, :, 1] = r
	return out.reshape(n*n, 2)


def generate_offsets(step, radius=2):
	rows = np.zeros((2 * radius + 1, 2 * radius + 1)).astype(int)
	cols = np.zeros((2 * radius + 1, 2 * radius + 1)).astype(int)

	for i in range(0, 2 * radius + 1):
		for j in range(0, 2 * radius + 1):
			rows[i, j] = i - radius
			cols[i, j] = j - radius

	rows = rows * step
	cols = cols * step
	return rows, cols

def generate_dist(step=1, radius=2):
	rows, cols = generate_offsets(step, radius)
	return jnp.linalg.norm(jnp.stack([rows, cols], axis=2), axis=2)


def data_prep(a, step=1, radius=2):
	orig_h, orig_w = a.shape[0], a.shape[1]

	offsets = generate_offsets(step, radius)

	boundary = radius * step

	if len(a.shape) == 3:
		a = jnp.pad(a, ((boundary, boundary), (boundary, boundary), (0, 0)))
	else:
		a = jnp.pad(a, ((boundary, boundary), (boundary, boundary)))

	def func_tile(idx):
		i, j = idx[0], idx[1]
		i += boundary
		j += boundary
		return a[offsets[0] + i, offsets[1] + j]

	# for now h = w, but might need to change this later
	idxs = indices_array(orig_h)
	return vmap(func_tile)(idxs)


# all arguments are of size (2*radius + 1, 2*radius + 1), where radius is 2
# all arguments prefixed with `phi` are scalars..
# this function will be vmapped over the entire 2D image space for high performance on GPU
def tile_atrous_decomposition(illum, variance, filter,
							  depth_center, depth_p, phi_depth,
							  normal_center, normal_p, phi_normal,
							  l_illum_center, l_illum_p, phi_l_illum, radius=2):
	weight_normal = jnp.power(jnp.sum(normal_center * normal_p, axis=2), phi_normal)
	weight_depth = jnp.where(phi_depth == 0, 0, jnp.abs(depth_center - depth_p) / phi_depth)
	weight_l_illum = jnp.abs(l_illum_center - l_illum_p) / phi_l_illum
	weight = jnp.exp(0.0 - jnp.maximum(weight_l_illum, 0.0) - jnp.maximum(weight_depth, 0.0)) * weight_normal
	weight *= filter
	weight = jnp.maximum(1e-6, weight)
	# TODO: why should we ignore the middle element? is it taken into account before this fn is called?
	weight = weight.at[radius, radius].set(0)

	if len(illum.shape) == 3:
		weight = jnp.expand_dims(weight, axis=2)

	filtered_illum = jnp.sum(illum * weight, axis=(0, 1)) / jnp.sum(weight)

	var_weight = jnp.square(weight)
	filtered_variance = jnp.sum(variance * var_weight) / jnp.sum(var_weight)

	return filtered_illum, filtered_variance


def learnable_vmap_atrous_decomposition(illum, variance, filter,
										depth_center, depth_p, phi_depth,
										normal_center, normal_p, phi_normal,
										l_illum_center, l_illum_p, phi_l_illum):
	filter = jnp.repeat(jnp.expand_dims(filter, axis=0), len(illum), axis=0)
	return vmap(tile_atrous_decomposition)(illum, variance, filter,
										   depth_center, depth_p, phi_depth,
										   normal_center, normal_p, phi_normal,
										   l_illum_center, l_illum_p, phi_l_illum)


def multiple_iter_atrous_decomposition(input_illum, input_var, input_depth, input_normal, input_depth_grad,
									   atrous_filter, g_phi_illum=4, g_phi_normal=128, g_phi_depth=1, radius=2,
									   compute_lum=True):
	ht, wt = input_illum.shape[0], input_illum.shape[1]

	def single_iter(i, data):
		input_illum, input_var, input_depth, input_normal, input_depth_grad, atrous_filter = data
		step_size = 1 << i

		input_var = gaussian_filter(input_var)
		if compute_lum:
			input_l_illum = luminance_vec(input_illum)
		else:
			# illum is already passed as lillum
			input_l_illum = input_illum

		illum = data_prep(input_illum, step=step_size, radius=radius)
		variance = data_prep(input_var, step=step_size, radius=radius)

		l_illum_p = data_prep(input_l_illum, step=step_size, radius=radius)
		depth_p = data_prep(input_depth, step=step_size, radius=radius)
		normal_p = data_prep(input_normal, step=step_size, radius=radius)

		l_illum_center = jnp.reshape(input_l_illum, newshape=(ht * wt))
		depth_center = jnp.reshape(input_depth, newshape=(ht * wt))
		normal_center = jnp.reshape(input_normal, newshape=(ht * wt, 3))

		phi_l_illum = g_phi_illum * jnp.sqrt(jnp.maximum(0.0, 1e-8 + input_var)).flatten()
		tmp2 = jnp.expand_dims(g_phi_depth * jnp.maximum(1e-8, input_depth_grad), axis=(2, 3))
		dist_vals = generate_dist(step=step_size, radius=radius)
		tmp11 = jnp.repeat(
			jnp.expand_dims(dist_vals, axis=0),
			wt,
			axis=0
		)
		tmp1 = jnp.repeat(
			jnp.expand_dims(tmp11, axis=0),
			ht,
			axis=0
		)
		phi_depth = jnp.reshape(tmp1 * tmp2, newshape=(ht * wt, 2*radius + 1, 2*radius+1))
		phi_normal = g_phi_normal * jnp.ones(ht * wt)

		output_illum, output_variance = learnable_vmap_atrous_decomposition(illum, variance, atrous_filter,
																				 depth_center, depth_p, phi_depth,
																				 normal_center, normal_p, phi_normal,
																				 l_illum_center, l_illum_p, phi_l_illum)
		output_illum = output_illum.reshape(input_illum.shape)
		output_var = output_variance.reshape(input_var.shape)

		data[0] = output_illum
		data[1] = output_var
		return data

	data = [input_illum, input_var, input_depth, input_normal, input_depth_grad, atrous_filter]

	for i in range(4):
		data = single_iter(i, data)
	filtered_data = data

	# The following is slightly faster on CPU but can't be used because it fails with step_size concretization error
	# for more details, just uncomment the following lines and run the code
	# filtered_data = lax.fori_loop(
	# 	0, 4, single_iter, data
	# )
	return filtered_data[0]


# if __name__ == "__main__":
# 	print(generate_atrous_kernel())