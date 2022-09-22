from jax import jit, grad
from temporal_gradient import *


input_path = "data"
scene_path = "scenes/cbox"
output_path = "output"

global_alpha = 0.2


def debug(i):
	return i // 512 == 210 and i % 512 == 125


# lambda in paper
def relative_gradient(frame0, frame1):
	assert frame0.shape == frame1.shape
	# small epsilon to avoid NaNs
	eps = 1e-6
	frame0 = jnp.maximum(frame0, np.full(fill_value=eps, shape=frame0.shape))
	frame1 = jnp.maximum(frame1, np.full(fill_value=eps, shape=frame1.shape))

	rel_grad = jnp.minimum(jnp.abs((frame1 - frame0)) / jnp.maximum(frame0, frame1), jnp.ones(frame0.shape))
	return luminance_vec(rel_grad)


def compute_weight(depth_center, depth_p, phi_depth, normal_center, normal_p, phi_normal, luminance_illum_center,
				   luminance_illum_p, phi_illum):
	weight_normal = jnp.power(saturate(normal_center.dot(normal_p)), phi_normal)
	weight_z = jnp.where(jnp.array([phi_depth]) == 0, jnp.array([0]),
						 jnp.array([jnp.abs(depth_center - depth_p) / phi_depth]))[0]
	# weight_z = 0.0 if phi_depth == 0 else abs(depth_center - depth_p) / phi_depth
	weight_l_illum = jnp.abs(luminance_illum_center - luminance_illum_p) / phi_illum
	weight_illum = jnp.exp(0.0 - jnp_max(weight_l_illum, 0.0) - jnp_max(weight_z, 0.0)) * weight_normal

	return weight_illum


def compute_moments(frame_illum):
	moments = []
	for illum in frame_illum:
		moment_1 = 0.2126 * illum[:, :, 0] + 0.7152 * illum[:, :, 1] + 0.0722 * illum[:, :, 2]
		moment_2 = jnp.square(moment_1)
		moments.append(jnp.stack([moment_1, moment_2], axis=2))
	return moments

def compute_disocclusion(frame_depth, frame_normal, frame_depth_grad, frame=1):

	prev_depth = frame_depth[frame - 1]
	prev_normal = frame_normal[frame - 1]
	curr_depth = frame_depth[frame]
	curr_normal = frame_normal[frame]
	curr_depth_grad = frame_depth_grad[frame]

	depth_mask = test_reprojected_depth(prev_depth, curr_depth, curr_depth_grad)
	normal_mask = test_reprojected_normal_vec(prev_normal, curr_normal)

	disocclusion = jnp.where(depth_mask & normal_mask, jnp.zeros(depth_mask.shape, dtype=jnp.uint8),
							 jnp.ones(depth_mask.shape, dtype=jnp.uint8))

	return disocclusion


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
								 disocclusion,  g_phi_illum=4, g_phi_normal=128, g_phi_depth=1, radius=2):
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


# TODO : pass history_len 2D array to increment history length unless occlusion is encountered
def asvgf(frame_illum, frame_depth, frame_normal,  vbuffer_lst, viewproj_lst, model_mats_lst,
                                   jax_models_lst, random_is_idxs, filter):
	frame_moments = compute_moments(frame_illum)
	frame_depth_grad = compute_frame_depth_gradient(frame_depth)

	curr_frame = 1
	disocclusion = compute_disocclusion(frame_depth, frame_normal, frame_depth_grad, frame=curr_frame)

	adaptive_alpha = compute_adaptive_alpha(frame_illum, frame_depth, frame_normal, vbuffer_lst, viewproj_lst,
											model_mats_lst, jax_models_lst, random_is_idxs)

	# we are doing temporal integration for all the pixels, later we will do spatial filtering for pixels which
	# encounter disocclusion
	# TODO: moment and frame_illum global accumulators need to be passed to integrate over more than 2 frames
	# TODO: try tap filtering, if no tap is found then set C'_i = C_i (disocclusion case). Note this is different than
	#  what we are doing for the moments/variance, we spatially filter to find a variance instead of just setting it
	#  to the previous value
	integrated_illum, integrated_moments, integrated_variance = temporal_integration(frame_illum, frame_moments,
																					 adaptive_alpha)

	input_list = [
		frame_illum[curr_frame], integrated_variance, frame_depth[curr_frame], frame_normal[curr_frame],
		frame_depth_grad[curr_frame], frame_moments[curr_frame]
	]
	input_illum, input_var, input_depth, input_normal, input_depth_grad, input_moments = convert_to_jnp(input_list)

	# this function only updates the pixels where there is a disocclusion and retains the values computed using
	# temporal_integration function for the rest
	input_var = spatial_variance_computation(input_illum, input_var, input_depth, input_normal, input_depth_grad,
													 input_moments, disocclusion)

	ht, wt, c = input_illum.shape
	input_var = jnp.reshape(input_var, newshape=(ht, wt))

	output_illum = multiple_iter_atrous_decomposition(integrated_illum, input_var, input_depth, input_normal,
													  input_depth_grad, filter)

	# TODO : return accumulated color and moments as well, they will again be passed as input to the next invocation of
	#  this function. They work as global accumulator variables.
	return output_illum


def loss_fn_asvgf(filter, gt, aux_args):
	pred_img = asvgf(*aux_args, filter)
	return jnp.mean(jnp.square(pred_img - gt))


if __name__ == '__main__':
	os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
	USE_TEMPORAL_ACCU = True

	os.makedirs(output_path, exist_ok=True)

	frame_illum = []
	frame_depth = []
	frame_normal = []
	frame_moments = []
	frame_depth_grad = []

	vbuffer_lst = []
	viewproj_lst = []
	model_mats_lst = []
	model_fnames_lst = []
	models_lst = []

	# TODO: render 10 frames, first frame with light on and the rest with light off
	for i in range(2):
		frame_illum.append(read_exr_file(join(input_path, "frame{}.exr".format(i))))
		frame_depth.append(read_exr_file(join(input_path, "frame{}_depth.exr".format(i)), single_channel=True))
		frame_normal.append(read_exr_file(join(input_path, "frame{}_normal.exr".format(i))))

		vbuffer_lst.append(load_vbuffer(join(input_path, "frame{}_vbuffer.npy".format(i))))
		viewproj_lst.append(np.load(join(input_path, "frame{}_viewproj.npy".format(i))))

		model_mats_lst.append(np.load(join(input_path, "frame{}_model_mats.npy".format(i))))
		model_fnames = read_txt_file(join(input_path, "frame{}_model_fnames.txt".format(i)))
		model_fnames_lst.append(model_fnames)
		models_lst.append(load_models(scene_path, model_fnames))

	downsample = 3

	# random intra stratum indices : these are generated beforehand and provided as input because it is non-trivial to
	# handle this in jax.More info can be found here https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html
	random_is_idxs = jnp.array(random.choices(population=np.arange(downsample ** 2),
											  k=(frame_depth[0].shape[0] // downsample) * (
														  frame_depth[0].shape[1] // downsample)))

	jax_models_lst = []
	for f in range(2):
		models = models_lst[f]

		jax_models = []
		max_k = -1
		for i in range(len(models)):
			model = models[i]
			jax_model = []

			# iterate over two keys, vertices and faces
			for j in range(len(model.keys())):
				key = list(model.keys())[j]
				vert_faces = model[key]
				max_k = max(max_k, len(vert_faces))

				jax_vert_faces = []

				for k in range(len(vert_faces)):
					item = list(vert_faces[k])

					jax_item = []
					for l in range(len(item)):
						jax_item.append(item[l])

					jax_vert_faces.append(jax_item)

				jax_model.append(jax_vert_faces)

			jax_models.append(jax_model)

		for i in range(len(jax_models)):
			for j in range(len(jax_models[i])):
				for k in range(max_k - len(jax_models[i][j])):
					jax_models[i][j].append([0.0] * len(jax_models[i][j][0]))

		jax_models_lst.append(jnp.array(jax_models))

	prev_frame = 0
	curr_frame = 1
	atrous_filter = jnp.array(generate_atrous_filter())

	output_illum = asvgf(frame_illum, frame_depth, frame_normal, vbuffer_lst, viewproj_lst, model_mats_lst,
                                   jax_models_lst, random_is_idxs, atrous_filter)
	write_exr_file(join(output_path, "final_color.exr"), output_illum)

	print("starting gradient computation...")

	gt = read_exr_file(join(input_path, "frame1_gt.exr"))
	aux_args = [frame_illum, frame_depth, frame_normal, vbuffer_lst, viewproj_lst, model_mats_lst, jax_models_lst,
				random_is_idxs]

	start_time = time.time()
	grad_loss = jit(grad(loss_fn_asvgf))
	gradient_filter = grad_loss(atrous_filter, gt, aux_args)
	print("time for gradient computation {} s".format(time.time() - start_time))
	print(gradient_filter.shape)
	print(gradient_filter)