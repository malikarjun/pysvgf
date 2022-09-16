import time
from os.path import exists, join
import os
import numpy as np
from file_utils import *
from matrix_utils import *
from svgf_utils import *
import random
from tqdm import tqdm
from functools import partial
from jax.lax import scan

input_path = "data_fixed"
output_path = "output_ad"
inter_path = "intermediate_results/temp_grad"

def debug(y, x):
	return y == 270 and x == 51

def models_to_jnp_array(model):
	print(model)
	for i in range(len(model)):
		for j in range(len(model[i])):
			model[i][j] = list(model[i][j])

	return jnp.array(model)

def interpolate_vertex(model, prim_id, uv):
	face = model[1][prim_id].astype(int)
	verts = model[0]
	u, v = uv[0], uv[1]
	w = 1 - u - v
	return verts[face[0]] * w + verts[face[1]] * u + verts[face[2]] * v


def _interpolate_vertex(model, prim_id, uv):
	face = model.mesh_list[0].faces[prim_id]
	verts = np.array(model.vertices)
	u, v = uv[0], uv[1]
	w = 1 - u - v
	return verts[face[0]] * w + verts[face[1]] * u + verts[face[2]] * v


def get_intra_stratum_loc(downsample=3):
	loc = random.randint(0, downsample ** 2 - 1)
	return loc // downsample, loc % downsample


# lambda in the paper
@partial(jit, static_argnames=['h', 'w', 'downsample'])
def compute_temporal_gradient(illum_lst, depth_lst, normal_lst, vbuffer_lst, viewproj_lst, model_mats_lst,
							  models_lst, random_is_idxs, h, w, prev_frame=0, curr_frame=1, downsample=3):

	nn = (h//downsample) * (w//downsample)

	init_s = dict(prev_vbuffer = vbuffer_lst[prev_frame], prev_models = models_lst[prev_frame],
				  prev_illum = illum_lst[prev_frame], prev_depth = depth_lst[prev_frame],
				  prev_normal = normal_lst[prev_frame], 	curr_vbuffer = vbuffer_lst[curr_frame],
				  curr_viewproj = viewproj_lst[curr_frame], curr_model_mats = model_mats_lst[curr_frame],
				  curr_models = models_lst[curr_frame], curr_illum = illum_lst[curr_frame],
				  curr_depth = depth_lst[curr_frame], curr_normal = normal_lst[curr_frame],
				  temp_grad = jnp.zeros((h // downsample , w // downsample)),
				  visited = jnp.full(shape=(h // downsample, w // downsample), fill_value=False),
				  lillum = jnp.zeros((h // downsample, w // downsample)),
				  variance = jnp.zeros((h // downsample, w // downsample)),
				  depth = jnp.zeros((h // downsample, w // downsample)),
				  depth_gradient = jnp.ones((h // downsample, w // downsample)),
				  normal = jnp.zeros((h // downsample, w // downsample, 3)),
				  idx_array = jnp.array(indices_array(h, step=downsample)),
				  random_is_idxs = random_is_idxs,

				  ret_prev_depth = jnp.zeros((h // downsample, w // downsample)),
				  ret_curr_depth = jnp.zeros((h // downsample, w // downsample)),
				  ret_curr_depth_grad = jnp.zeros((h // downsample, w // downsample)),

				  ret_prev_normal = jnp.zeros((h // downsample, w // downsample, 3)),
				  ret_curr_normal = jnp.zeros((h // downsample, w // downsample, 3)),

				  curr_y=jnp.zeros(nn).astype(int),
				  curr_x=jnp.zeros(nn).astype(int),

				  )


	def step(s, i):
		y, x = s["idx_array"][i]

		# intra stratum pixel loc for prev frame
		loc = s["random_is_idxs"][i]
		prev_isy, prev_isx = loc // downsample, loc % downsample


		prev_x = x + prev_isx
		prev_y = y + prev_isy

		# the following need to be the same for two frames, that is how we establish correspondence
		shape_id = s["prev_vbuffer"][prev_y, prev_x, 0].astype(int)
		prim_id = s["prev_vbuffer"][prev_y, prev_x, 1].astype(int)
		uv = s["prev_vbuffer"][prev_y, prev_x, 2:]

		# load MVP matrix for the current frame
		curr_model_mat = s["curr_model_mats"][shape_id]

		prev_pos_lc = interpolate_vertex(s["prev_models"][shape_id], prim_id, uv)
		curr_mvp = s["curr_viewproj"] @ curr_model_mat

		# not exactly ndc, but the values are in the range [0, 1]^2
		curr_pos_ndc = xform_point(mat=curr_mvp, point=prev_pos_lc)

		curr_y, curr_x = (curr_pos_ndc[1] * h).astype(int), (curr_pos_ndc[0] * w).astype(int)
		s["curr_y"] = s["curr_y"].at[i].set(curr_y)
		s["curr_x"] = s["curr_x"].at[i].set(curr_x)

		# convert to stratum resolution
		curr_sx, curr_sy = curr_x // downsample, curr_y // downsample

		l_curr = luminance(s["curr_illum"][curr_y, curr_x])
		l_prev = luminance(s["prev_illum"][prev_y, prev_x])

		accept = True
		s["ret_curr_depth"] = s["ret_curr_depth"].at[curr_sy, curr_sx].set(s["curr_depth"][curr_y, curr_x])
		s["ret_prev_depth"] = s["ret_prev_depth"].at[curr_sy, curr_sx].set(s["prev_depth"][prev_y, prev_x])

		_curr_depth_gradient = jnp.maximum(ddx(s["curr_depth"], curr_x, curr_y), ddy(s["curr_depth"], curr_x, curr_y))
		s["ret_curr_depth_grad"] = s["ret_curr_depth_grad"].at[curr_sy, curr_sx].set(_curr_depth_gradient)


		s["ret_curr_normal"] =  s["ret_curr_normal"].at[curr_sy, curr_sx].set(s["curr_normal"][curr_y, curr_x])
		s["ret_prev_normal"] = s["ret_prev_normal"].at[curr_sy, curr_sx].set(s["prev_normal"][prev_y, prev_x])

		denom = jnp.maximum(jnp.maximum(l_curr, l_prev), 1e-8)
		s["data"] = s["data"].at[curr_sy, curr_sx].set(jnp.abs(l_curr - l_prev) / denom)

		return s, None

	s, _ = scan(step, init_s, np.arange(nn))

	depth_mask = test_reprojected_depth(s["ret_prev_depth"], s["ret_curr_depth"], s["ret_curr_depth_grad"])
	normal_mask = test_reprojected_normal_vec(s["ret_prev_normal"], s["ret_curr_normal"])

	s["data"] = jnp.where(depth_mask & normal_mask, s["data"], jnp.zeros(s["data"].shape))


	def filter_step(s, i):

		curr_y, curr_x = s["curr_y"][i], s["curr_x"][i]
		curr_sy, curr_sx = curr_y // downsample, curr_x // downsample

		mesh_id = s["curr_vbuffer"][curr_y, curr_x, 0]

		s["moments"] = jnp.zeros(2)
		s["sum_wt"] = 0

		def filter_inner_step(s, idx):
			yy, xx = idx
			p = jnp.array([curr_sy, curr_sx]) * downsample + jnp.array([yy, xx]).astype(int)

			mesh_id_p = s["curr_vbuffer"][p[0], p[1], 0]
			rgb = s["curr_illum"][p[0], p[1]]
			l = luminance(rgb)

			wt = jnp.where(mesh_id_p == mesh_id, 1.0, 0.0)
			s["moments"] += jnp.array([l, l ** 2]) * wt
			s["sum_wt"] += wt

			return s, None


		s, _ = scan(filter_inner_step, s, indices_array(downsample))

		s["moments"] /= s["sum_wt"]
		s["variance"] = s["variance"].at[curr_sy, curr_sx].set(jnp.maximum(0, s["moments"][1] - s["moments"][0] * s["moments"][0]))
		s["lillum"] = s["lillum"].at[curr_sy, curr_sx].set(s["moments"][0])
		s["depth"] = s["depth"].at[curr_sy, curr_sx].set(s["curr_depth"][curr_y, curr_x])
		s["normal"] = s["normal"].at[curr_sy, curr_sx].set(s["curr_normal"][curr_y, curr_x])
		s["depth_gradient"] = s["depth_gradient"].at[curr_sy, curr_sx].set(s["ret_curr_depth_grad"][curr_sy, curr_sx])

		del s["moments"]
		del s["sum_wt"]

		return s, None


	s, _ = scan(filter_step, s, np.arange(nn))

	return s["data"], s["lillum"], s["variance"], s["depth"], s["normal"], s["depth_gradient"]


def max_pool(x):
  idxs = jnp.arange(x.shape[0])

  def g(a, b):
    av, ai = a
    bv, bi = b
    which = av >= bv
    return jnp.where(which, av, bv), jnp.where(which, ai, bi)

  _, idxs = lax.reduce_window((x, idxs), (-np.inf, -1), g,
                    window_dimensions=(2,), window_strides=(2,),
                    padding=((0, 0),))
  return x[idxs]

# reconstruct temporal gradient using max pooling in the stratum resolution
def reconstruct_temp_gradient(temp_grad, downsample=3, grad_filter_radius=2):
	down_h, down_w = temp_grad.shape[0], temp_grad.shape[1]
	orig_h, orig_w = down_h * downsample, down_w * downsample

	recon_temp_grad = jnp.zeros((orig_h, orig_w))
	temp_grad = jnp.pad(temp_grad, grad_filter_radius, constant_values=-1)


	init_s = dict(temp_grad=temp_grad, recon_temp_grad=recon_temp_grad)


	# TODO: the images after using jax are slightly different. this image recon_filtered_temp_grad_new.exr
	def step(s, idx):
		y, x = idx
		# stratum indices
		sy, sx = y // downsample, x // downsample
		sy += grad_filter_radius
		sx += grad_filter_radius

		def inner_step(s, inner_idx):
			yy, xx = inner_idx

			# local stratum indices used within this loop nest
			lsy, lsx = sy + yy, sx + xx

			s["recon_temp_grad"] = s["recon_temp_grad"].at[y, x].set(jnp.maximum(s["recon_temp_grad"][y, x], s["data"][lsy, lsx]))
			return s, None

		local_idx_array = indices_array(grad_filter_radius * 2 + 1, start=-grad_filter_radius)
		s, _ = scan(inner_step, s, local_idx_array)

		return s, None

	idx_array = indices_array(orig_h)
	s, _ = scan(step, init_s, idx_array)



	return s["recon_temp_grad"]



if __name__=="__main__":
	os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
	random.seed(int(time.time()))

	illum_lst = []
	depth_lst = []
	normal_lst = []
	vbuffer_lst = []
	viewproj_lst = []
	model_mats_lst = []
	model_fnames_lst = []
	models_lst = []

	for frame in range(2):
		illum_lst.append(read_exr_file(join(input_path, "frame{}.exr".format(frame)), jax_array=True))
		depth_lst.append(read_exr_file(join(input_path, "frame{}_depth.exr".format(frame)), jax_array=True)[:, :, 0])
		normal_lst.append(read_exr_file(join(input_path, "frame{}_normal.exr".format(frame)), jax_array=True))
		vbuffer_lst.append(load_vbuffer(join(input_path, "frame{}_vbuffer.npy".format(frame))))
		viewproj_lst.append(np.load(join(input_path, "frame{}_viewproj.npy".format(frame))))

		model_mats_lst.append(np.load(join(input_path, "frame{}_model_mats.npy".format(frame))))
		model_fnames = read_txt_file(join(input_path, "frame{}_model_fnames.txt".format(frame)))
		model_fnames_lst.append(model_fnames)
		models_lst.append(load_models(model_fnames))



	downsample = 3
	hh, ww = depth_lst[0].shape
	# random intra stratum indices : these are generated beforehand and provided as input because it is non-trivial to
	# handle this in jax. More info can be found here https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html
	random_is_idxs = jnp.array(random.choices(population=np.arange(downsample ** 2),
									k= (depth_lst[0].shape[0]//downsample) * (depth_lst[0].shape[1] // downsample) ))

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



	temp_grad, lillum, variance, \
	depth, normal, depth_gradient = compute_temporal_gradient(illum_lst, depth_lst, normal_lst, vbuffer_lst, viewproj_lst,
											   model_mats_lst, jax_models_lst, random_is_idxs, hh, ww,
											   downsample=downsample)

	temp_grad, lillum, variance, depth, normal, depth_gradient = convert_to_np([temp_grad, lillum, variance,
																					depth, normal, depth_gradient])

	write_exr_file(join(output_path, "temp_grad_new.exr"), temp_grad)
	write_exr_file(join(inter_path, "lillum_new.exr"), lillum)
	write_exr_file(join(inter_path, "variance_new.exr"), variance)
	write_exr_file(join(inter_path, "depth_new.exr"), depth)
	write_exr_file(join(inter_path, "depth_gradient_new.exr"), depth_gradient)
	write_exr_file(join(inter_path, "normal_new.exr"), normal)

	radius = 1
	filtered_temp_grad = multiple_iter_atrous_decomposition(temp_grad, variance, depth, normal, depth_gradient,
															generate_box_filter(radius=radius), g_phi_normal=0,
															radius=radius,
															compute_lum=False)

	write_exr_file(join(output_path, "filtered_temp_grad_new.exr"), filtered_temp_grad)

	recon_filtered_temp_grad = reconstruct_temp_gradient(np.asarray(filtered_temp_grad))
	write_exr_file(join(input_path, "recon_filtered_temp_grad_new.exr"), recon_filtered_temp_grad)
