import time
from os.path import join, exists
import os
from copy import deepcopy

import numpy as np
from tqdm import tqdm
from jax import grad, jit, lax, vmap
import jax.scipy as jsp
import jax.numpy as jnp

from learnable_utils import *
from file_utils import *

input_path = "data"
output_path = "output_vec"
inter_path = "intermediate_results"

g_phi_illum = 4
g_phi_normal = 128
g_phi_depth = 3
global_alpha = 0.2
radius = 2


def debug(i):
	return i // 512 == 210 and i % 512 == 125


def luminance_vec(img):
	return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]


def luminance_vec_1d(img):
	return 0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2]


def luminance(rgb):
	return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


# lambda in paper
def relative_gradient(frame0, frame1):
	assert frame0.shape == frame1.shape
	# small epsilon to avoid NaNs
	eps = 1e-6
	frame0 = jnp.maximum(frame0, np.full(fill_value=eps, shape=frame0.shape))
	frame1 = jnp.maximum(frame1, np.full(fill_value=eps, shape=frame1.shape))

	rel_grad = jnp.minimum(jnp.abs((frame1 - frame0)) / jnp.maximum(frame0, frame1), jnp.ones(frame0.shape))
	return luminance_vec(rel_grad)

def test_reprojected_depth(z1, z2, dz):
	z_diff = abs(z1 - z2)
	return z_diff < 2.0 * (dz + 1e-3)

def test_reprojected_normal(n1, n2):
	return jnp.sum(n1 * n2, axis=2) > 0.9


def jnp_max(a, b):
	return jnp.maximum(jnp.array([a]), jnp.array([b]))[0]


def jnp_min(a, b):
	return jnp.minimum(jnp.array([a]), jnp.array([b]))[0]

def saturate(val):
	return jnp.maximum(jnp.array([0]), jnp.minimum(jnp.array([val]), jnp.array([1])))[0]


def frac(val):
	return np.ceil(val) - val


def inside(p, _h, _w):
	return np.all(np.greater_equal(p, np.array([0, 0]))) and np.all(np.less(p, np.array([_h, _w])))


def lerp(a, b, frac):
	return a * (1 - frac) + b * frac

def compute_weight(depth_center, depth_p, phi_depth, normal_center, normal_p, phi_normal, luminance_illum_center,
				   luminance_illum_p, phi_illum):
	weight_normal = jnp.power(saturate(normal_center.dot(normal_p)), phi_normal)
	weight_z = jnp.where(jnp.array([phi_depth]) == 0, jnp.array([0]),
						 jnp.array([jnp.abs(depth_center - depth_p) / phi_depth]))[0]
	# weight_z = 0.0 if phi_depth == 0 else abs(depth_center - depth_p) / phi_depth
	weight_l_illum = jnp.abs(luminance_illum_center - luminance_illum_p) / phi_illum
	weight_illum = jnp.exp(0.0 - jnp_max(weight_l_illum, 0.0) - jnp_max(weight_z, 0.0)) * weight_normal

	return weight_illum


def convert_to_np(lst):
	return [np.array(item) for item in lst]

def convert_to_jnp(lst):
	return [jnp.array(item) for item in lst]

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

# only works for square images
# check out this answer for generic impl https://stackoverflow.com/a/44230705
def indices_array(n):
	r = np.arange(n)
	out = np.empty((n, n, 2), dtype=int)
	out[:, :, 0] = r[:, None]
	out[:, :, 1] = r
	return out.reshape(n*n, 2)


def data_prep(a, step=1):
	orig_h, orig_w = a.shape[0], a.shape[1]

	offsets = generate_offsets(step, radius)

	boundary = radius * step
	boundary = radius * 2 ** 5
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


def gaussian_filter(img):
	x = jnp.linspace(-3, 3, 7)
	window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
	return jsp.signal.convolve(img, window, mode='same')





def compute_moments(frame_illum):
	moments = []
	for illum in frame_illum:
		moment_1 = 0.2126 * illum[:, :, 0] + 0.7152 * illum[:, :, 1] + 0.0722 * illum[:, :, 2]
		moment_2 = jnp.square(moment_1)
		moments.append(jnp.stack([moment_1, moment_2], axis=2))
	return moments

def compute_depth_gradient(frame_depth):
	depth_gradients = []
	for depth in frame_depth:
		depth_gradients.append(jnp.ones(depth.shape))
	return depth_gradients

def compute_adaptive_alpha(frame_illum, frame_depth, frame_normal, frame_depth_grad, frame=1):

	frame0 = frame_illum[frame-1]
	frame1 = frame_illum[frame]

	lamda = relative_gradient(frame0, frame1)
	alpha = (1 - lamda) * global_alpha + lamda

	prev_depth = frame_depth[frame - 1]
	prev_normal = frame_normal[frame - 1]
	curr_depth = frame_depth[frame]
	curr_normal = frame_normal[frame]
	curr_depth_grad = frame_depth_grad[frame]

	depth_mask = test_reprojected_depth(prev_depth, curr_depth, curr_depth_grad)
	normal_mask = test_reprojected_normal(prev_normal, curr_normal)

	alpha = jnp.where(depth_mask & normal_mask, alpha, jnp.ones(alpha.shape))
	disocclusion = jnp.where(depth_mask & normal_mask, jnp.zeros(alpha.shape, dtype=jnp.uint8),
							 jnp.ones(alpha.shape, dtype=jnp.uint8))

	return alpha, disocclusion


def temporal_integration(frame_illum, frame_moments, adapt_alpha, curr_frame=1):

	prev_illum, curr_illum = frame_illum[curr_frame-1], frame_illum[curr_frame]
	prev_moments, curr_moments = frame_moments[curr_frame-1], frame_moments[curr_frame]

	adapt_alpha = jnp.expand_dims(adapt_alpha, axis=2)
	integrated_illum = lerp(prev_illum, curr_illum, adapt_alpha)
	integrated_moments = lerp(prev_moments, curr_moments, adapt_alpha)
	integrated_variance = integrated_moments[:, :, 1] - jnp.square(integrated_moments[:, :, 0])

	return integrated_illum, integrated_moments, integrated_variance


# all arguments are of size (2*radius + 1, 2*radius + 1), where radius is 2
# moments is of size (2*radius + 1, 2*radius + 1, 2)
# all arguments prefixed with `phi` are scalars..
# this function will be vmapped over the entire 2D image space for high performance on GPU
def tile_spatial_variance_computation(moments,
									  depth_center, depth_p, phi_depth,
									  normal_center, normal_p, phi_normal,
									  l_illum_center, l_illum_p, phi_l_illum):
	weight_normal = jnp.power(jnp.sum(normal_center * normal_p, axis=2), phi_normal)
	weight_depth = jnp.where(phi_depth == 0, 0, jnp.abs(depth_center - depth_p) / phi_depth)
	weight_l_illum = jnp.abs(l_illum_center - l_illum_p) / phi_l_illum
	weight = jnp.exp(0.0 - jnp.maximum(weight_l_illum, 0.0) - jnp.maximum(weight_depth, 0.0)) * weight_normal
	weight = jnp.maximum(1e-6, weight)
	# weight = weight.at[radius, radius].set(0)

	var_weight = jnp.square(weight)
	filtered_moments = jnp.sum(moments * jnp.expand_dims(var_weight, axis=2), axis=(0, 1)) / jnp.sum(var_weight)
	filtered_variance = filtered_moments[1] - jnp.square(filtered_moments[0])

	return filtered_variance


def spatial_variance_computation(input_illum, input_var, input_depth, input_normal, input_depth_grad, input_moments,
								 disocclusion):
	ht, wt, c = input_illum.shape
	input_l_illum = luminance_vec(input_illum)

	input_moments = data_prep(input_moments)
	l_illum_p = data_prep(input_l_illum)
	depth_p = data_prep(input_depth)
	normal_p = data_prep(input_normal)

	l_illum_center = jnp.reshape(input_l_illum, newshape=(ht * wt))
	depth_center = jnp.reshape(input_depth, newshape=(ht * wt))
	normal_center = jnp.reshape(input_normal, newshape=(ht * wt, c))

	phi_l_illum = g_phi_illum * jnp.sqrt(jnp.maximum(0.0, 1e-8 + input_var)).flatten()
	tmp2 = jnp.expand_dims(g_phi_depth * jnp.maximum(1e-8, input_depth_grad), axis=(2, 3))
	dist_vals = generate_dist()
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
	phi_depth = jnp.reshape(tmp1 * tmp2, newshape=(ht * wt, 2 * radius + 1, 2 * radius + 1))
	phi_normal = g_phi_normal * jnp.ones(ht * wt)

	filtered_variance = vmap(tile_spatial_variance_computation)(input_moments,
																depth_center, depth_p, phi_depth,
																normal_center, normal_p, phi_normal,
																l_illum_center, l_illum_p, phi_l_illum)

	return jnp.where(jnp.reshape(disocclusion, newshape=(ht * wt)),
					 filtered_variance,
					 jnp.reshape(input_var, newshape=(ht * wt)))


# all arguments are of size (2*radius + 1, 2*radius + 1), where radius is 2
# all arguments prefixed with `phi` are scalars..
# this function will be vmapped over the entire 2D image space for high performance on GPU
def tile_atrous_decomposition(illum, variance, filter,
							  depth_center, depth_p, phi_depth,
							  normal_center, normal_p, phi_normal,
							  l_illum_center, l_illum_p, phi_l_illum):
	weight_normal = jnp.power(jnp.sum(normal_center * normal_p, axis=2), phi_normal)
	weight_depth = jnp.where(phi_depth == 0, 0, jnp.abs(depth_center - depth_p) / phi_depth)
	weight_l_illum = jnp.abs(l_illum_center - l_illum_p) / phi_l_illum
	weight = jnp.exp(0.0 - jnp.maximum(weight_l_illum, 0.0) - jnp.maximum(weight_depth, 0.0)) * weight_normal
	weight *= filter
	weight = jnp.maximum(1e-6, weight)
	weight = weight.at[radius, radius].set(0)

	filtered_illum = jnp.sum(illum * jnp.expand_dims(weight, axis=2), axis=(0, 1)) / jnp.sum(weight)

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
									   atrous_filter):
	ht, wt, c = input_illum.shape

	def single_iter(i, data):
		input_illum, input_var, input_depth, input_normal, input_depth_grad, atrous_filter = data
		step_size = 1 << i

		input_var = gaussian_filter(input_var)
		input_l_illum = luminance_vec(input_illum)

		illum = data_prep(input_illum, step_size)
		variance = data_prep(input_var, step=step_size)

		l_illum_p = data_prep(input_l_illum, step=step_size)
		depth_p = data_prep(input_depth, step=step_size)
		normal_p = data_prep(input_normal, step=step_size)

		l_illum_center = jnp.reshape(input_l_illum, newshape=(ht * wt))
		depth_center = jnp.reshape(input_depth, newshape=(ht * wt))
		normal_center = jnp.reshape(input_normal, newshape=(ht * wt, c))

		phi_l_illum = g_phi_illum * jnp.sqrt(jnp.maximum(0.0, 1e-8 + input_var)).flatten()
		tmp2 = jnp.expand_dims(g_phi_depth * jnp.maximum(1e-8, input_depth_grad), axis=(2, 3))
		dist_vals = generate_dist(step=step_size)
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


	# using fori_loop is very slow takes 4 minutes for 2 iterations. twice as slower compared to the regular python loop
	filtered_data = lax.fori_loop(
		0, 5, single_iter, data
	)
	return filtered_data[0]


def asvgf(frame_illum, frame_depth, frame_normal, filter):
	frame_moments = compute_moments(frame_illum)
	frame_depth_grad = compute_depth_gradient(frame_depth)

	curr_frame = 1

	adaptive_alpha, disocclusion = compute_adaptive_alpha(frame_illum, frame_depth, frame_normal, frame_depth_grad)

	integrated_illum, integrated_moments, integrated_variance = temporal_integration(frame_illum, frame_moments,
																					 adaptive_alpha)

	input_list = [
		frame_illum[curr_frame], integrated_variance, frame_depth[curr_frame], frame_normal[curr_frame],
		frame_depth_grad[curr_frame], frame_moments[curr_frame]
	]
	input_illum, input_var, input_depth, input_normal, input_depth_grad, input_moments = convert_to_jnp(input_list)

	# input_illum = jnp.array(frame_illum[curr_frame])
	# input_var = jnp.array(integrated_variance)
	# input_depth = jnp.array(frame_depth[curr_frame])
	# input_normal = jnp.array(frame_normal[curr_frame])
	# input_depth_grad = jnp.array(frame_depth_grad[curr_frame])
	# input_moments = jnp.array(frame_moments[curr_frame])
	input_var = spatial_variance_computation(input_illum, input_var, input_depth, input_normal, input_depth_grad,
													 input_moments, disocclusion)

	ht, wt, c = input_illum.shape
	input_var = jnp.reshape(input_var, newshape=(ht, wt))

	output_illum = multiple_iter_atrous_decomposition(integrated_illum, input_var, input_depth, input_normal,
													  input_depth_grad, filter)
	return output_illum



def _loss_fn_multiple_iter(filter, gt, aux_args):
	pred_img = multiple_iter_atrous_decomposition(*aux_args, filter)
	return jnp.mean(jnp.square(pred_img - gt))

def loss_fn_multiple_iter(filter, gt, aux_args):
	pred_img = asvgf(*aux_args, filter)
	return jnp.mean(jnp.square(pred_img - gt))


if __name__ == '__main__':
	os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
	USE_TEMPORAL_ACCU = True

	frame_illum = []
	frame_depth = []
	frame_normal = []
	frame_moments = []
	frame_depth_grad = []

	for i in range(2):
		frame_illum.append(read_exr_file(join(input_path, "frame{}.exr".format(i))))
		frame_depth.append(read_exr_file(join(input_path, "frame{}_depth.exr".format(i)))[:, :, 0])
		frame_normal.append(read_exr_file(join(input_path, "frame{}_normal.exr".format(i))))

	prev_frame = 0
	curr_frame = 1
	atrous_filter = jnp.array(generate_atrous_kernel())

	output_illum = asvgf(frame_illum, frame_depth, frame_normal, atrous_filter)
	write_exr_file(join(output_path, "final_color_all_jax.exr"), output_illum)

	print("starting gradient computation...")

	gt = read_exr_file(join(input_path, "frame1_gt.exr"))
	aux_args = [frame_illum, frame_depth, frame_normal]

	start_time = time.time()
	grad_loss = jit(grad(loss_fn_multiple_iter))
	gradient_filter = grad_loss(atrous_filter, gt, aux_args)
	print("time for gradient computation {} s".format(time.time() - start_time))
	print(gradient_filter.shape)
	print(gradient_filter)