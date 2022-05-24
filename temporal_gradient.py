from os.path import exists, join
import os
import numpy as np
from file_utils import *
from matrix_utils import *
from learnable_utils import *
import random

input_path = "data_fixed"
output_path = "output_ad"

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
	vbuffer_lst = []
	viewproj_lst = []
	model_mats_lst = []
	model_fnames_lst = []
	models_lst = []

	for frame in range(2):
		illum_lst.append(read_exr_file(join(input_path, "frame{}.exr".format(frame))))
		vbuffer_lst.append(np.load(join(input_path, "frame{}_vbuffer.npy".format(frame))))
		viewproj_lst.append(np.load(join(input_path, "frame{}_viewproj.npy".format(frame))))

		model_mats_lst.append(np.load(join(input_path, "frame{}_model_mats.npy".format(frame))))
		model_fnames = read_txt_file(join(input_path, "frame{}_model_fnames.txt".format(frame)))
		model_fnames_lst.append(model_fnames)
		models_lst.append(load_models(model_fnames))

	prev_vbuffer = vbuffer_lst[prev_frame]
	prev_models = models_lst[prev_frame]
	prev_illum = illum_lst[prev_frame]

	curr_viewproj = viewproj_lst[curr_frame]
	curr_model_mats = model_mats_lst[curr_frame]
	curr_models = models_lst[curr_frame]
	curr_illum = illum_lst[curr_frame]

	h, w = prev_vbuffer.shape[0], prev_vbuffer.shape[1]

	world_loc = np.zeros((h, w, 3))
	depth_buffer = np.zeros((h, w))

	temp_grad = np.zeros( (h//downsample, w//downsample) )
	visited = np.full(shape=(h//downsample, w//downsample), fill_value=False)

	for y in range(0, h, 3):
		for x in range(0, w, 3):

			# intra stratum pixel loc for prev frame
			prev_isy, prev_isx = get_intra_stratum_loc(downsample=downsample)

			prev_x = x + prev_isx
			prev_y = y + prev_isy

			# the following need to be the same for two frames, that is how we establish correspondence
			shape_id = int(prev_vbuffer[prev_x, prev_y, 0])
			prim_id = int(prev_vbuffer[prev_x, prev_y, 1])
			uv = prev_vbuffer[prev_x, prev_y, 2:]


			# load MVP matrix for the current frame
			curr_model_mat = curr_model_mats[shape_id]


			if curr_models[shape_id] is None:
				# TODO: how do we handle spherical objects? Ans. we remove them from the scene :(
				continue

			prev_pos_lc = interpolate_vertex(prev_models[shape_id], prim_id, uv)
			curr_mvp = curr_viewproj @ curr_model_mat

			# curr_pos_wc = xform_point(mat=curr_model_mat, point=prev_pos_lc)

			# not exactly ndc, but the values are in the range [0, 1]^2
			curr_pos_ndc = xform_point(mat=curr_mvp, point=prev_pos_lc)

			curr_x, curr_y = int(curr_pos_ndc[0] * h), int(curr_pos_ndc[1] * w)

			# convert to stratum resolution
			curr_sx, curr_sy = curr_x//downsample, curr_y//downsample
			if visited[curr_sx, curr_sy]:
				continue

			visited[curr_sx, curr_sy] = True

			curr_lillum = luminance(curr_illum[curr_x, curr_y])
			prev_lillum = luminance(prev_illum[prev_x, prev_y])
			denom = max(max(curr_lillum, prev_lillum), 1e-8)
			temp_grad[curr_sx, curr_sy] = np.abs(curr_lillum - prev_lillum) / denom

			# depth_buffer[py, px] = pos_wc[2]
			# world_loc[y, x] = pos_wc

	# return world_loc, depth_buffer
	return temp_grad




if __name__=="__main__":
	os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

	frame = 1
	# world_loc, depth_buffer = generate_world_position(frame=frame)
	# write_exr_file(join(output_path, "frame{}_world_loc.exr".format(frame)), world_loc)
	# write_exr_file(join(output_path, "frame{}_depth_buffer.exr".format(frame)), depth_buffer)

	temp_grad = compute_temporal_gradient()
	write_exr_file(join(output_path, "temp_grad.exr"), temp_grad)
