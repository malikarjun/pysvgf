from svgf_utils import *


def generate_offsets_non_negative(radius=2):
	rows = np.zeros((2 * radius + 1, 2 * radius + 1)).astype(int)
	cols = np.zeros((2 * radius + 1, 2 * radius + 1)).astype(int)

	for i in range(0, 2 * radius + 1):
		for j in range(0, 2 * radius + 1):
			rows[i, j] = i
			cols[i, j] = j

	return rows, cols

def data_prep_non_overlapping(a, radius=1):
	orig_h, orig_w = a.shape[0], a.shape[1]

	offsets = generate_offsets_non_negative(radius)

	def func_tile(idx):
		i, j = idx[0], idx[1]
		return a[offsets[0] + i, offsets[1] + j]

	# for now h = w, but might need to change this later
	idxs = indices_array(n=orig_h, step=2*radius+1)
	return vmap(func_tile)(idxs)

# all arguments are of size (2*radius + 1, 2*radius + 1), where radius is 2
# all arguments prefixed with `phi` are scalars..
# this function will be vmapped over the entire 2D image space for high performance on GPU
def tile_atrous_grad_decomposition(temp_grad, lum, variance, filter,
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

	filtered_lum = jnp.sum(lum * weight, axis=(0, 1)) / jnp.sum(weight)

	temp_grad_weight = jnp.stack([weight, weight], axis=2)
	filtered_temp_grad = jnp.sum(temp_grad * temp_grad_weight, axis=(0, 1)) / jnp.sum(weight)

	var_weight = jnp.square(weight)
	filtered_variance = jnp.sum(variance * var_weight) / jnp.sum(var_weight)

	return filtered_temp_grad, filtered_lum, filtered_variance


def learnable_vmap_atrous_grad_decomposition(temp_grad, lum, variance, filter,
										depth_center, depth_p, phi_depth,
										normal_center, normal_p, phi_normal,
										l_illum_center, l_illum_p, phi_l_illum):
	filter = jnp.repeat(jnp.expand_dims(filter, axis=0), len(lum), axis=0)
	return vmap(tile_atrous_grad_decomposition)(temp_grad, lum, variance, filter,
										   depth_center, depth_p, phi_depth,
										   normal_center, normal_p, phi_normal,
										   l_illum_center, l_illum_p, phi_l_illum)


# input_temp_grad has two values numer and denom, they are both filtered separately
def multiple_iter_atrous_grad_decomposition(input_temp_grad, input_lum, input_var, input_depth, input_normal,
											input_depth_grad, atrous_filter, g_phi_illum=4, g_phi_normal=128,
											g_phi_depth=1, radius=1):
	ht, wt = input_lum.shape[0], input_lum.shape[1]

	def single_iter(i, data):
		input_temp_grad, input_lum, input_var, input_depth, input_normal, input_depth_grad, atrous_filter = data
		step_size = 1 << i

		# input_var = gaussian_filter(input_var)

		temp_grad = data_prep(input_temp_grad, step=step_size, radius=radius)
		lum = data_prep(input_lum, step=step_size, radius=radius)
		variance = data_prep(input_var, step=step_size, radius=radius)

		lum_p = data_prep(input_lum, step=step_size, radius=radius)
		depth_p = data_prep(input_depth, step=step_size, radius=radius)
		normal_p = data_prep(input_normal, step=step_size, radius=radius)

		lum_center = jnp.reshape(input_lum, newshape=(ht * wt))
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

		output_temp_grad, output_illum, output_variance = learnable_vmap_atrous_grad_decomposition(
			temp_grad, lum, variance, atrous_filter, depth_center, depth_p, phi_depth, normal_center, normal_p,
			phi_normal, lum_center, lum_p, phi_l_illum)

		output_temp_grad = output_temp_grad.reshape(input_temp_grad.shape)
		output_illum = output_illum.reshape(input_lum.shape)
		output_var = output_variance.reshape(input_var.shape)

		data[0] = output_temp_grad
		data[1] = output_illum
		data[2] = output_var
		return data

	data = [input_temp_grad, input_lum, input_var, input_depth, input_normal, input_depth_grad, atrous_filter]

	for i in range(5):
		data = single_iter(i, data)
	filtered_data = data

	return filtered_data[0]


if __name__=="__main__":
	a = np.zeros((6, 6, 3))
	# for i in range(6):
	# 	for j in range(6):
	# 		a[i, j] = 6*i + j

	val = data_prep_non_overlapping(jnp.array(a))
	pass
