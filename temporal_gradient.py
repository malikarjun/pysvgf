import time
from os.path import exists, join
import os
import numpy as np
from file_utils import *
from matrix_utils import *
from svgf_utils import *
import random
from tqdm import tqdm

input_path = "data_fixed"
output_path = "output_ad"
inter_path = "intermediate_results/temp_grad"

def debug(y, x):
	return y == 270 and x == 51

def interpolate_vertex(model, prim_id, uv):
	face = model.mesh_list[0].faces[prim_id]
	verts = np.array(model.vertices)
	u, v = uv[0], uv[1]
	w = 1 - u - v
	return verts[face[0]] * w + verts[face[1]] * u + verts[face[2]] * v


def get_intra_stratum_loc(downsample=3):
	loc = random.randint(0, downsample ** 2 - 1)
	return loc // downsample, loc % downsample


# lambda in the paper
def compute_temporal_gradient(prev_frame=0, curr_frame=1, downsample=3):

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

	temp_grad = np.zeros( (h//downsample, w//downsample) )
	visited = np.full(shape=(h//downsample, w//downsample), fill_value=False)

	lillum = np.zeros((h//downsample, w//downsample))
	variance = np.zeros((h // downsample, w // downsample))

	depth = np.zeros((h//downsample, w//downsample))
	depth_gradient = np.ones((h//downsample, w//downsample))
	normal = np.zeros((h//downsample, w//downsample, 3))

	for y in tqdm(range(0, h, 3)):
		for x in range(0, w, 3):

			# if not debug(y, x):
			# 	continue

			# intra stratum pixel loc for prev frame
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

			# not exactly ndc, but the values are in the range [0, 1]^2
			curr_pos_ndc = xform_point(mat=curr_mvp, point=prev_pos_lc)

			curr_y, curr_x = int(curr_pos_ndc[1] * h), int(curr_pos_ndc[0] * w)


			# convert to stratum resolution
			curr_sx, curr_sy = curr_x // downsample, curr_y // downsample

			l_curr = luminance(curr_illum[curr_y, curr_x])
			l_prev = luminance(prev_illum[prev_y, prev_x])


			accept = True
			_curr_depth = curr_depth[curr_y, curr_x]
			_prev_depth = prev_depth[prev_y, prev_x]
			_curr_depth_gradient = max(ddx(curr_depth, curr_x, curr_y), ddy(curr_depth, curr_x, curr_y))

			accept = accept and test_reprojected_depth(_prev_depth, _curr_depth, _curr_depth_gradient)

			_curr_normal = curr_normal[curr_y, curr_x]
			_prev_normal = prev_normal[prev_y, prev_x]
			accept = accept and test_reprojected_normal(_prev_normal, _curr_normal)

			if accept:

				if visited[curr_sy, curr_sx]:
					continue

				visited[curr_sy, curr_sx] = True

				denom = max(max(l_curr, l_prev), 1e-8)
				# TODO: Christoph is doing some clamping between 0 and 200. Is that really needed?
				temp_grad[curr_sy, curr_sx] = np.abs(l_curr - l_prev) / denom


			# copied from Christoph's codebase
			mesh_id = curr_vbuffer[curr_y, curr_x, 0]
			moments = np.array([l_curr, l_curr ** 2])
			z_curr = curr_depth[curr_y, curr_x]
			normal_curr = curr_normal[curr_y, curr_x]
			sum_wt = 1
			for yy in range(downsample):
				for xx in range(downsample):
					p = np.array([curr_sy, curr_sx]) * downsample + np.array([yy, xx])
					p = p.astype(int)

					if not np.allclose(p, np.array([curr_y, curr_x])):
						mesh_id_p = curr_vbuffer[p[0], p[1], 0]
						rgb = curr_illum[p[0], p[1]]
						l = luminance(rgb)

						wt = 1.0 if mesh_id_p == mesh_id else 0.0

						moments += np.array([l, l ** 2]) * wt
						sum_wt += wt

			moments /= sum_wt
			variance[curr_sy, curr_sx] = max(0, moments[1] - moments[0] * moments[0])
			lillum[curr_sy, curr_sx] = moments[0]
			depth[curr_sy, curr_sx] = z_curr
			depth_gradient[curr_sy, curr_sx] = _curr_depth_gradient
			normal[curr_sy, curr_sx] = normal_curr

	return temp_grad, lillum, variance, depth, normal, depth_gradient

# reconstruct temporal gradient using max pooling in the stratum resolution
def reconstruct_temp_gradient(temp_grad, downsample=3, grad_filter_radius=2):
	down_h, down_w = temp_grad.shape[0], temp_grad.shape[1]
	orig_h, orig_w = down_h * downsample, down_w * downsample
	recon_temp_grad = np.zeros((orig_h, orig_w))

	for y in tqdm(range(orig_h)):
		for x in range(orig_w):
			# stratum indices
			sy, sx = y // downsample, x // downsample


			for yy in range(-grad_filter_radius, grad_filter_radius):
				for xx in range(-grad_filter_radius, grad_filter_radius):

					# local stratum indices used within this loop nest
					lsy, lsx = sy + yy, sx + xx

					if 0 <= lsy < down_h and 0 <= lsx < down_w:
						recon_temp_grad[y, x] = max(recon_temp_grad[y, x], temp_grad[lsy, lsx])

	return recon_temp_grad



if __name__=="__main__":
	os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
	random.seed(int(time.time()))

	if not exists(join(output_path, "temp_grad.exr")):
		temp_grad, lillum, variance, depth, normal, depth_gradient = compute_temporal_gradient()
		write_exr_file(join(output_path, "temp_grad.exr"), temp_grad)
		write_exr_file(join(inter_path, "lillum.exr"), lillum)
		write_exr_file(join(inter_path, "variance.exr"), variance)
		write_exr_file(join(inter_path, "depth.exr"), depth)
		write_exr_file(join(inter_path, "depth_gradient.exr"), depth_gradient)
		write_exr_file(join(inter_path, "normal.exr"), normal)
	else:
		temp_grad = read_exr_file(join(output_path, "temp_grad.exr"), single_channel=True)
		lillum = read_exr_file(join(inter_path, "lillum.exr"), single_channel=True)
		variance = read_exr_file(join(inter_path, "variance.exr"), single_channel=True)
		depth = read_exr_file(join(inter_path, "depth.exr"), single_channel=True)
		depth_gradient = read_exr_file(join(inter_path, "depth_gradient.exr"), single_channel=True)
		normal = read_exr_file(join(inter_path, "normal.exr"))

	radius = 1
	filtered_temp_grad = multiple_iter_atrous_decomposition(temp_grad, variance, depth, normal, depth_gradient,
															generate_box_filter(radius=radius), radius=radius,
															compute_lum=False)

	write_exr_file(join(output_path, "filtered_temp_grad.exr"), filtered_temp_grad)

	recon_filtered_temp_grad = reconstruct_temp_gradient(np.asarray(filtered_temp_grad))
	write_exr_file(join(output_path, "recon_filtered_temp_grad.exr"), recon_filtered_temp_grad)
