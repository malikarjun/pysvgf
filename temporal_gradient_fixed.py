import time
from os.path import exists, join
import os
import numpy as np
from file_utils import *
from matrix_utils import *
from svgf_utils import *
from asvgf_utils import *
import random
from tqdm import tqdm

input_path = "data_fixed"
output_path = "output_fixed"

def interpolate_vertex(model, prim_id, uv):
	face = model['faces'][prim_id]
	verts = np.array(model['vertices'])
	u, v = uv[0], uv[1]
	w = 1 - u - v
	return verts[face[0]] * w + verts[face[1]] * u + verts[face[2]] * v

# randomly sample a location in 3x3 stratum
def get_intra_stratum_loc(downsample=3):
	loc = random.randint(0, downsample ** 2 - 1)
	return loc // downsample, loc % downsample


# forward projection along with reprojection tests for depth, luminance, normal
def forward_projection(illum_lst, depth_lst, normal_lst, vbuffer_lst, viewproj_lst, model_mats_lst,
							  model_fnames_lst, models_lst, prev_frame=0, curr_frame=1, downsample=3):
	prev_vbuffer = vbuffer_lst[prev_frame]
	prev_models = models_lst[prev_frame]
	prev_illum = illum_lst[prev_frame]
	prev_depth = depth_lst[prev_frame]
	prev_normal = normal_lst[prev_frame]

	curr_vbuffer = vbuffer_lst[curr_frame]
	curr_viewproj = viewproj_lst[curr_frame]
	curr_model_mats = model_mats_lst[curr_frame]
	curr_models = models_lst[curr_frame]
	curr_illum = illum_lst[curr_frame]
	curr_depth = depth_lst[curr_frame]
	curr_normal = normal_lst[curr_frame]

	h, w = prev_vbuffer.shape[0], prev_vbuffer.shape[1]

	# just use luminance for shading sample value
	prev_shading_samples = np.zeros(shape=(h // downsample, w // downsample))
	curr_shading_samples = np.zeros(shape=(h // downsample, w // downsample))

	variance = np.zeros(shape=(h // downsample, w // downsample))
	depth = np.zeros(shape=(h // downsample, w // downsample))
	normal = np.zeros(shape=(h // downsample, w // downsample, 3))
	lum = np.zeros(shape=(h // downsample, w // downsample))

	for y in tqdm(range(0, h, downsample)):
		for x in range(0, w, downsample):
			prev_isy, prev_isx = get_intra_stratum_loc(downsample=downsample)

			prev_x = x + prev_isx
			prev_y = y + prev_isy

			# the following need to be the same for two frames, that is how we establish correspondence
			shape_id = int(prev_vbuffer[prev_y, prev_x, 0])
			prim_id = int(prev_vbuffer[prev_y, prev_x, 1])
			uv = prev_vbuffer[prev_y, prev_x, 2:]

			# load MVP matrix for the current frame
			curr_model_mat = curr_model_mats[shape_id]

			prev_pos_lc = interpolate_vertex(prev_models[shape_id], prim_id, uv)
			curr_mvp = curr_viewproj @ curr_model_mat

			curr_pos_ndc = xform_point(mat=curr_mvp, point=prev_pos_lc)

			curr_y, curr_x = int(curr_pos_ndc[1] * h), int(curr_pos_ndc[0] * w)

			# convert to stratum resolution
			curr_sx, curr_sy = curr_x // downsample, curr_y // downsample

			accept = True
			_curr_depth = curr_depth[curr_y, curr_x]
			_prev_depth = prev_depth[prev_y, prev_x]
			_curr_depth_gradient = max(ddx_(curr_depth, curr_x, curr_y), ddy_(curr_depth, curr_x, curr_y))

			accept = accept and test_reprojected_depth(_prev_depth, _curr_depth, _curr_depth_gradient)

			_curr_normal = curr_normal[curr_y, curr_x]
			_prev_normal = prev_normal[prev_y, prev_x]
			accept = accept and test_reprojected_normal(_prev_normal, _curr_normal)

			if not accept:
				continue

			prev_shading_samples[curr_sy, curr_sx] = luminance(prev_illum[prev_y, prev_x])
			curr_shading_samples[curr_sy, curr_sx] = luminance(curr_illum[curr_y, curr_x])


			intra_stratum_lum = luminance_vec(curr_illum[y:y+downsample, x:x+downsample])
			lum[curr_sy, curr_sx] = np.mean(intra_stratum_lum)
			variance[curr_sy, curr_sx] = np.mean(np.square(intra_stratum_lum)) - np.square(np.mean(intra_stratum_lum))
			depth[curr_sy, curr_sx] = curr_depth[curr_y, curr_x]
			normal[curr_sy, curr_sx] = curr_normal[curr_y, curr_x]


	numer = np.abs(curr_shading_samples - prev_shading_samples)
	denom = np.maximum(curr_shading_samples, prev_shading_samples)


	return numer, denom, lum, variance, depth, normal


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
		illum_lst.append(read_exr_file(join(input_path, "frame{}.exr".format(frame))))
		depth_lst.append(read_exr_file(join(input_path, "frame{}_depth.exr".format(frame)))[:, :, 0])
		normal_lst.append(read_exr_file(join(input_path, "frame{}_normal.exr".format(frame))))
		vbuffer_lst.append(load_vbuffer(join(input_path, "frame{}_vbuffer.npy".format(frame))))
		viewproj_lst.append(np.load(join(input_path, "frame{}_viewproj.npy".format(frame))))

		model_mats_lst.append(np.load(join(input_path, "frame{}_model_mats.npy".format(frame))))
		model_fnames = read_txt_file(join(input_path, "frame{}_model_fnames.txt".format(frame)))
		model_fnames_lst.append(model_fnames)
		models_lst.append(load_models(model_fnames))


	numer, denom, lum, variance, depth, normal = forward_projection(illum_lst, depth_lst, normal_lst, vbuffer_lst, viewproj_lst,
														model_mats_lst, model_fnames_lst, models_lst)

	write_exr_file(join(output_path, "numer.exr"), numer)
	write_exr_file(join(output_path, "denom.exr"), denom)
	write_exr_file(join(output_path, "lum.exr"), lum)
	write_exr_file(join(output_path, "variance.exr"), variance)
	write_exr_file(join(output_path, "depth.exr"), depth)
	write_exr_file(join(output_path, "normal.exr"), normal)

	# depth_grad = compute_depth_gradient(depth)
	#
	#
	# temp_grad = jnp.stack([numer, denom], axis=2)
	#
	# box_filter = jnp.array(generate_box_filter())
	# filtered_temp_grad = multiple_iter_atrous_grad_decomposition(
	# 	temp_grad, lum, variance, depth, normal, depth_grad, box_filter)
	#
	#
	# write_exr_file(join(output_path, "filtered_temp_grad.exr"), filtered_temp_grad)
