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



def _relative_gradient(frame0, frame1):
	assert frame0.shape == frame1.shape
	# small epsilon to avoid NaNs
	eps = 1e-6
	frame0 = np.maximum(frame0, np.full(fill_value=eps, shape=frame0.shape))
	frame1 = np.maximum(frame1, np.full(fill_value=eps, shape=frame1.shape))

	rel_grad = np.minimum(np.abs((frame1 - frame0)) / np.maximum(frame0, frame1), np.ones(frame0.shape))
	return luminance_vec(rel_grad)

# lambda in paper
def relative_gradient(frame0, frame1):
	assert frame0.shape == frame1.shape
	# small epsilon to avoid NaNs
	eps = 1e-6
	frame0 = jnp.maximum(frame0, np.full(fill_value=eps, shape=frame0.shape))
	frame1 = jnp.maximum(frame1, np.full(fill_value=eps, shape=frame1.shape))

	rel_grad = jnp.minimum(jnp.abs((frame1 - frame0)) / jnp.maximum(frame0, frame1), jnp.ones(frame0.shape))
	return luminance_vec(rel_grad)


def _test_reprojected_depth(z1, z2, dz):
	z_diff = abs(z1 - z2)
	return z_diff < 2.0 * (dz + 1e-3)

def test_reprojected_depth(z1, z2, dz):
	z_diff = abs(z1 - z2)
	return z_diff < 2.0 * (dz + 1e-3)

def _test_reprojected_normal(n1, n2):
	return n1.dot(n2) > 0.9

def test_reprojected_normal(n1, n2):
	return jnp.sum(n1 * n2, axis=2) > 0.9


def _compute_moments(color):
	moments = np.zeros((color.shape[0], color.shape[1], 2))
	moments[:, :, 0] = 0.2126 * color[:, :, 0] + 0.7152 * color[:, :, 1] + 0.0722 * color[:, :, 2]
	moments[:, :, 1] = np.square(moments[:, :, 0])
	return moments


def _compute_depth_gradient(depth):
	depth = deepcopy(depth)
	h, w = depth.shape

	depth_grad = np.ones((h, w))

	return depth_grad


def jnp_max(a, b):
	return jnp.maximum(jnp.array([a]), jnp.array([b]))[0]


def jnp_min(a, b):
	return jnp.minimum(jnp.array([a]), jnp.array([b]))[0]

def _saturate(val):
	return max(0, min(val, 1))
	# return jnp.maximum(jnp.array([0]), jnp.minimum(jnp.array([val]), jnp.array([1])))[0]

def saturate(val):
	# return max(0, min(val, 1))
	return jnp.maximum(jnp.array([0]), jnp.minimum(jnp.array([val]), jnp.array([1])))[0]


def frac(val):
	return np.ceil(val) - val


def inside(p, _h, _w):
	return np.all(np.greater_equal(p, np.array([0, 0]))) and np.all(np.less(p, np.array([_h, _w])))


def lerp(a, b, frac):
	return a * (1 - frac) + b * frac


def _compute_adaptive_alpha(frame_depth, frame_normal, frame_depth_grad, frame=1):
	print("compute adaptive alpha")

	hh, ww = frame_depth[0].shape

	disocclusion = np.zeros((hh, ww))

	frame0 = read_exr_file(join(input_path, "frame0.exr"))
	frame1 = read_exr_file(join(input_path, "frame1.exr"))

	lamda = _relative_gradient(frame0, frame1)
	alpha = (1 - lamda) * global_alpha + lamda

	prev_depth = frame_depth[frame - 1]
	prev_normal = frame_normal[frame - 1]
	curr_depth = frame_depth[frame]
	curr_normal = frame_normal[frame]
	curr_depth_grad = frame_depth_grad[frame]

	for i in tqdm(range(hh)):
		for j in range(ww):
			if not _test_reprojected_depth(prev_depth[i, j], curr_depth[i, j], curr_depth_grad[i, j]) or \
					not _test_reprojected_normal(prev_normal[i, j], curr_normal[i, j]):
				alpha[i, j] = 1.0
				disocclusion[i, j] = 1

	return alpha, disocclusion


def _temporal_integration(g_prev_illum, g_prev_moments, g_illum, g_moments, g_adapt_alpha):
	g_adapt_alpha = deepcopy(g_adapt_alpha)
	hh, ww, cc = g_illum.shape
	integrated_illum = deepcopy(g_illum)
	integrated_moments = deepcopy(g_moments)
	integrated_variance = np.zeros((hh, ww))

	hh, ww, cc = g_prev_illum.shape
	print("running temporal integration...")
	for i in tqdm(range(hh)):
		for j in range(ww):
			prev_illum = g_prev_illum[i, j]
			prev_moments = g_prev_moments[i, j]

			integrated_illum[i, j] = lerp(prev_illum, g_illum[i, j], g_adapt_alpha[i, j])
			integrated_moments[i, j] = lerp(prev_moments, g_moments[i, j], g_adapt_alpha[i, j])
			integrated_variance[i, j] = integrated_moments[i, j, 1] - np.square(integrated_moments[i, j, 0])

	return integrated_illum, integrated_moments, integrated_variance


def compute_weight(depth_center, depth_p, phi_depth, normal_center, normal_p, phi_normal, luminance_illum_center,
				   luminance_illum_p, phi_illum):
	weight_normal = jnp.power(saturate(normal_center.dot(normal_p)), phi_normal)
	weight_z = jnp.where(jnp.array([phi_depth]) == 0, jnp.array([0]),
						 jnp.array([jnp.abs(depth_center - depth_p) / phi_depth]))[0]
	# weight_z = 0.0 if phi_depth == 0 else abs(depth_center - depth_p) / phi_depth
	weight_l_illum = jnp.abs(luminance_illum_center - luminance_illum_p) / phi_illum
	weight_illum = jnp.exp(0.0 - jnp_max(weight_l_illum, 0.0) - jnp_max(weight_z, 0.0)) * weight_normal

	return weight_illum

def _compute_weight(depth_center, depth_p, phi_depth, normal_center, normal_p, phi_normal, luminance_illum_center,
                   luminance_illum_p, phi_illum):
	weight_normal = pow(_saturate(normal_center.dot(normal_p)), phi_normal)
	weight_z = 0.0 if phi_depth == 0 else abs(depth_center - depth_p) / phi_depth
	weight_l_illum = abs(luminance_illum_center - luminance_illum_p) / phi_illum
	weight_illum = np.exp(0.0 - max(weight_l_illum, 0.0) - max(weight_z, 0.0)) * weight_normal

	return weight_illum


def convert_to_np(lst):
	return [np.array(item) for item in lst]

def _compute_variance_spatially(frame_illum, frame_depth, frame_normal, frame_moments, frame_depth_grad, disocclusion,
							   in_variance, frame=1):
	print("computing variance spatially...")
	hh, ww, _ = frame_illum[frame].shape
	g_illumination = frame_illum[frame]
	g_moments = frame_moments[frame]
	g_depth = frame_depth[frame]
	g_normal = frame_normal[frame]
	g_depth_grad = frame_depth_grad[frame]

	variance = deepcopy(in_variance)

	radius = 3

	for i in tqdm(range(hh)):
		for j in range(ww):

			if not disocclusion[i, j]:
				continue

			ipos = np.array([i, j])
			sum_w_illumination = 0.0
			sum_moments = np.array([0, 0]).astype(float)

			illumination_center = g_illumination[i, j]
			l_illumination_center = luminance(illumination_center)
			z_center = g_depth[i, j]
			n_center = g_normal[i, j]

			phi_depth = max(1e-8, g_depth_grad[i, j]) * g_phi_depth

			for yy in range(-radius, radius):
				for xx in range(-radius, radius):
					p = np.array([yy, xx]) + ipos
					inside = np.all(np.greater_equal(p, np.array([0, 0]))) and np.all(np.less(p, np.array([hh, ww])))

					if inside:
						y, x = p[0], p[1]
						illumination_p = g_illumination[y, x]
						moments_p = g_moments[y, x]
						l_illumination_p = luminance(illumination_p)
						z_p = g_depth[y, x]
						n_p = g_normal[y, x]

						w = _compute_weight(z_center, z_p, phi_depth * np.linalg.norm(np.array([yy, xx])),
										   n_center, n_p, g_phi_normal,
										   l_illumination_center, l_illumination_p, g_phi_illum)

						sum_w_illumination += w
						sum_moments += w * moments_p

			sum_w_illumination = max(sum_w_illumination, 1e-6)
			sum_moments /= sum_w_illumination

			variance[i, j] = sum_moments[1] - sum_moments[0] * sum_moments[0]
	return variance


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


def loss_fn(img, gt, variance, filter, aux_args):
	pred_img, pred_var = learnable_vmap_atrous_decomposition(img, variance, filter, *aux_args)
	return jnp.mean(jnp.square(pred_img - gt))


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


	# _integrated_illum, _integrated_moments, _ = _temporal_integration(np.array(prev_illum), np.array(prev_moments),
	# 												np.array(curr_illum), np.array(curr_moments), np.array(adapt_alpha))


	integrated_moments = lerp(prev_moments, curr_moments, adapt_alpha)
	# print("testing {}".format(np.allclose(np.array(integrated_illum), _integrated_illum, atol=1e-5)))
	# print("testing {}".format(np.allclose(np.array(integrated_moments), _integrated_moments, atol=1e-5)))

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

	# return filtered_variance


def dump_data(lst, dir):
	[np.save(join(dir, "{}".format(i)), data) for (i, data) in enumerate(lst)]

def asvgf(frame_illum, frame_depth, frame_normal):
	frame_moments = compute_moments(frame_illum)
	frame_depth_grad = compute_depth_gradient(frame_depth)

	# frame_moments = []
	# frame_depth_grad = []
	# for i in range(2):
	# 	frame_moments.append(_compute_moments(frame_illum[i]))
	# 	frame_depth_grad.append(_compute_depth_gradient(frame_depth[i]))


	prev_frame = 0
	curr_frame = 1


	# adaptive_alpha, disocclusion = _compute_adaptive_alpha(*convert_to_np([frame_depth, frame_normal, frame_depth_grad]))
	adaptive_alpha, disocclusion = compute_adaptive_alpha(frame_illum, frame_depth, frame_normal, frame_depth_grad)

	# print(np.allclose(np.array(adaptive_alpha), np.load("intermediate_results/adaptive_alpha.npy")))
	# print(np.allclose(np.array(disocclusion), np.load("intermediate_results/disocclusion.npy")))


	# integrated_illum, integrated_moments, integrated_variance = _temporal_integration(*convert_to_np([frame_illum[prev_frame],
	# 																				  frame_moments[prev_frame],
	# 																				  frame_illum[curr_frame],
	# 																				  frame_moments[curr_frame], adaptive_alpha]))

	integrated_illum, integrated_moments, integrated_variance = temporal_integration(frame_illum, frame_moments,
																					 adaptive_alpha)

	# print(np.allclose(np.array(integrated_illum), np.load("intermediate_results/integrated_illum.npy"), atol=1e-5))
	# print(np.allclose(np.array(integrated_moments), np.load("intermediate_results/integrated_moments.npy"), atol=1e-5))

	input_illum = jnp.array(frame_illum[curr_frame])
	input_var = jnp.array(integrated_variance)
	input_depth = jnp.array(frame_depth[curr_frame])
	input_normal = jnp.array(frame_normal[curr_frame])
	input_depth_grad = jnp.array(frame_depth_grad[curr_frame])
	input_moments = jnp.array(frame_moments[curr_frame])
	input_var = spatial_variance_computation(input_illum, input_var, input_depth, input_normal, input_depth_grad,
													 input_moments, disocclusion)

	# input_var = _compute_variance_spatially(*convert_to_np([frame_illum, frame_depth, frame_normal, frame_moments,
	# 														frame_depth_grad, disocclusion, input_var]))

	ht, wt, c = input_illum.shape
	input_var = jnp.reshape(input_var, newshape=(ht, wt))


	# they don't have to be perfectly equal because the padding done for the tiled computation will lead to
	# differences
	# jnp_variance = np.array(integrated_variance).reshape(input_var.shape)
	# orig_variance = np.load("intermediate_results/variance.npy")
	atrous_filter = jnp.array(generate_atrous_kernel())

	# write_exr_file("input_illum_all_jax.exr", np.array(input_illum))
	# dump_data([input_illum, input_var, input_depth, input_normal, input_depth_grad], "intermediate_results/all_jax")

	output_illum = multiple_iter_atrous_decomposition(integrated_illum, input_var, input_depth, input_normal,
													  input_depth_grad, atrous_filter)
	return output_illum



def loss_fn_multiple_iter(filter, gt, aux_args):
	pred_img = multiple_iter_atrous_decomposition(*aux_args, filter)
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
		frame_moments.append(_compute_moments(frame_illum[i]))
		frame_depth_grad.append(_compute_depth_gradient(frame_depth[i]))

	prev_frame = 0
	curr_frame = 1

	output_illum = asvgf(frame_illum, frame_depth, frame_normal)
	write_exr_file(join(output_path, "final_color_all_jax.exr"), output_illum)
	# exit(2)

	adaptive_alpha_path = join(inter_path, "adaptive_alpha.npy")
	disocclusion_path = join(inter_path, "disocclusion.npy")
	adaptive_alpha = get_file(adaptive_alpha_path)
	disocclusion = get_file(disocclusion_path)
	if adaptive_alpha is None or disocclusion is None:
		adaptive_alpha, disocclusion = _compute_adaptive_alpha(frame_depth, frame_normal, frame_depth_grad)
		np.save(adaptive_alpha_path, adaptive_alpha)
		np.save(disocclusion_path, disocclusion)

	integrated_illum_path = join(inter_path, "integrated_illum.npy")
	integrated_moments_path = join(inter_path, "integrated_moments.npy")
	integrated_variance_path = join(inter_path, "variance.npy")

	integrated_illum = get_file(integrated_illum_path)
	integrated_moments = get_file(integrated_moments_path)
	integrated_variance = get_file(integrated_variance_path)

	if integrated_illum is None or integrated_moments is None or integrated_variance is None:
		integrated_illum, integrated_moments, integrated_variance = _temporal_integration(
			g_prev_illum=frame_illum[prev_frame], g_prev_moments=frame_moments[prev_frame],
			g_illum=frame_illum[curr_frame], g_moments=frame_moments[curr_frame], g_adapt_alpha=adaptive_alpha)

		integrated_variance = _compute_variance_spatially(frame_illum, frame_depth, frame_normal, frame_moments,
														 frame_depth_grad, disocclusion, integrated_variance)

		np.save(integrated_illum_path, integrated_illum)
		np.save(integrated_moments_path, integrated_moments)
		np.save(integrated_variance_path, integrated_variance)

	frame_illum[curr_frame] = integrated_illum
	frame_moments[curr_frame] = integrated_moments

	input_illum = jnp.array(frame_illum[curr_frame])
	input_var = jnp.array(integrated_variance)
	input_depth = jnp.array(frame_depth[curr_frame])
	input_normal = jnp.array(frame_normal[curr_frame])
	input_depth_grad = jnp.array(frame_depth_grad[curr_frame])
	atrous_filter = jnp.array(generate_atrous_kernel())
	ht, wt, c = input_illum.shape

	# write_exr_file("input_illum_orig.exr", np.array(input_illum))
	# dump_data([input_illum, input_var, input_depth, input_normal, input_depth_grad], "intermediate_results/orig")

	output_illum = multiple_iter_atrous_decomposition(input_illum, input_var, input_depth, input_normal,
	                                                  input_depth_grad, atrous_filter)
	write_exr_file(join(output_path, "final_color_ad.exr"), output_illum)

	# TODO: the gradient computation is taking forever...
	# print("starting gradient computation...")
	#
	# gt = read_exr_file(join(input_path, "frame1_gt.exr"))
	# aux_args = [input_illum, input_var, input_depth, input_normal, input_depth_grad]
	#
	# start_time = time.time()
	# grad_loss = jit(grad(loss_fn_multiple_iter))
	# gradient_filter = grad_loss(atrous_filter, gt, aux_args)
	# print("time for gradient computation {} s".format(time.time() - start_time))
	# print(gradient_filter.shape)
	# print(gradient_filter)