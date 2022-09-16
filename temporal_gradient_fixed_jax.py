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
from jax.lax import scan

input_path = "data_fixed"
output_path = "output_fixed"
global_alpha = 0.2


def interpolate_vertex(model, prim_id, uv):
    face = model[1][prim_id].astype(int)
    verts = model[0]
    u, v = uv[0], uv[1]
    w = 1 - u - v
    return verts[face[0]] * w + verts[face[1]] * u + verts[face[2]] * v

# randomly sample a location in 3x3 stratum
def get_intra_stratum_loc(downsample=3):
    loc = random.randint(0, downsample ** 2 - 1)
    return loc // downsample, loc % downsample


# forward projection along with reprojection tests for depth, luminance, normal
def forward_projection(illum_lst, depth_lst, normal_lst, vbuffer_lst, viewproj_lst, model_mats_lst,
                       models_lst, random_is_idxs, h, w, prev_frame=0, curr_frame=1, downsample=3):

    nn = (h//downsample) * (w//downsample)

    init_s = dict(prev_vbuffer = vbuffer_lst[prev_frame], prev_models = models_lst[prev_frame],
                  prev_illum = illum_lst[prev_frame], prev_depth = depth_lst[prev_frame],
                  prev_normal = normal_lst[prev_frame], 	curr_vbuffer = vbuffer_lst[curr_frame],
                  curr_viewproj = viewproj_lst[curr_frame], curr_model_mats = model_mats_lst[curr_frame],
                  curr_models = models_lst[curr_frame], curr_illum = illum_lst[curr_frame],
                  curr_depth = depth_lst[curr_frame], curr_normal = normal_lst[curr_frame],


                  idx_array = jnp.array(indices_array(h, step=downsample)),
                  random_is_idxs = random_is_idxs,
                  curr_y=jnp.zeros(nn).astype(int),
                  curr_x=jnp.zeros(nn).astype(int),

                  ret_prev_depth=jnp.zeros((h // downsample, w // downsample)),
                  ret_curr_depth=jnp.zeros((h // downsample, w // downsample)),
                  ret_curr_depth_grad=jnp.zeros((h // downsample, w // downsample)),
                  ret_prev_normal=jnp.zeros((h // downsample, w // downsample, 3)),
                  ret_curr_normal=jnp.zeros((h // downsample, w // downsample, 3)),

                  prev_shading_samples=np.zeros(shape=(h // downsample, w // downsample)),
                  curr_shading_samples = np.zeros(shape=(h // downsample, w // downsample)),
                  depth = np.zeros(shape=(h // downsample, w // downsample)),
                  normal = np.zeros(shape=(h // downsample, w // downsample, 3)),
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

        curr_pos_ndc = xform_point(mat=curr_mvp, point=prev_pos_lc)

        curr_y, curr_x = (curr_pos_ndc[1] * h).astype(int), (curr_pos_ndc[0] * w).astype(int)
        s["curr_y"] = s["curr_y"].at[i].set(curr_y)
        s["curr_x"] = s["curr_x"].at[i].set(curr_x)

        # convert to stratum resolution
        curr_sx, curr_sy = curr_x // downsample, curr_y // downsample


        s["ret_curr_depth"] = s["ret_curr_depth"].at[curr_sy, curr_sx].set(s["curr_depth"][curr_y, curr_x])
        s["ret_prev_depth"] = s["ret_prev_depth"].at[curr_sy, curr_sx].set(s["prev_depth"][prev_y, prev_x])

        _curr_depth_gradient = jnp.maximum(ddx(s["curr_depth"], curr_x, curr_y), ddy(s["curr_depth"], curr_x, curr_y))
        s["ret_curr_depth_grad"] = s["ret_curr_depth_grad"].at[curr_sy, curr_sx].set(_curr_depth_gradient)

        s["ret_curr_normal"] = s["ret_curr_normal"].at[curr_sy, curr_sx].set(s["curr_normal"][curr_y, curr_x])
        s["ret_prev_normal"] = s["ret_prev_normal"].at[curr_sy, curr_sx].set(s["prev_normal"][prev_y, prev_x])

        s["prev_shading_samples"] = s["prev_shading_samples"].at[curr_sy, curr_sx].set(
            luminance(s["prev_illum"][prev_y, prev_x]))
        s["curr_shading_samples"] = s["curr_shading_samples"].at[curr_sy, curr_sx].set(
            luminance(s["curr_illum"][curr_y, curr_x]))

        s["depth"] = s["depth"].at[curr_sy, curr_sx].set(s["curr_depth"][curr_y, curr_x])
        s["normal"] = s["normal"].at[curr_sy, curr_sx].set(s["curr_normal"][curr_y, curr_x])

        return s, None

    s, _ = scan(step, init_s, np.arange(nn))


    depth_mask = test_reprojected_depth(s["ret_prev_depth"], s["ret_curr_depth"], s["ret_curr_depth_grad"])
    normal_mask = test_reprojected_normal_vec(s["ret_prev_normal"], s["ret_curr_normal"])

    s["prev_shading_samples"] = jnp.where(depth_mask & normal_mask, s["prev_shading_samples"],
                                          jnp.zeros(s["prev_shading_samples"].shape))
    s["curr_shading_samples"] = jnp.where(depth_mask & normal_mask, s["curr_shading_samples"],
                                          jnp.zeros(s["curr_shading_samples"].shape))


    numer = jnp.abs(s["curr_shading_samples"] - s["prev_shading_samples"])
    denom = jnp.maximum(s["curr_shading_samples"], s["prev_shading_samples"])

    # prev_vbuffer = vbuffer_lst[prev_frame]
    # prev_models = models_lst[prev_frame]
    # prev_illum = illum_lst[prev_frame]
    # prev_depth = depth_lst[prev_frame]
    # prev_normal = normal_lst[prev_frame]
    #
    # curr_vbuffer = vbuffer_lst[curr_frame]
    # curr_viewproj = viewproj_lst[curr_frame]
    # curr_model_mats = model_mats_lst[curr_frame]
    # curr_models = models_lst[curr_frame]
    # curr_illum = illum_lst[curr_frame]
    # curr_depth = depth_lst[curr_frame]
    # curr_normal = normal_lst[curr_frame]
    #
    # h, w = prev_vbuffer.shape[0], prev_vbuffer.shape[1]
    #
    # # just use luminance for shading sample value
    # prev_shading_samples = np.zeros(shape=(h // downsample, w // downsample))
    # curr_shading_samples = np.zeros(shape=(h // downsample, w // downsample))
    #
    # variance = np.zeros(shape=(h // downsample, w // downsample))
    # depth = np.zeros(shape=(h // downsample, w // downsample))
    # normal = np.zeros(shape=(h // downsample, w // downsample, 3))
    # lum = np.zeros(shape=(h // downsample, w // downsample))
    #
    # for y in tqdm(range(0, h, downsample)):
    # 	for x in range(0, w, downsample):
    # 		prev_isy, prev_isx = get_intra_stratum_loc(downsample=downsample)
    #
    # 		prev_x = x + prev_isx
    # 		prev_y = y + prev_isy
    #
    # 		# the following need to be the same for two frames, that is how we establish correspondence
    # 		shape_id = int(prev_vbuffer[prev_y, prev_x, 0])
    # 		prim_id = int(prev_vbuffer[prev_y, prev_x, 1])
    # 		uv = prev_vbuffer[prev_y, prev_x, 2:]
    #
    # 		# load MVP matrix for the current frame
    # 		curr_model_mat = curr_model_mats[shape_id]
    #
    # 		prev_pos_lc = interpolate_vertex(prev_models[shape_id], prim_id, uv)
    # 		curr_mvp = curr_viewproj @ curr_model_mat
    #
    # 		curr_pos_ndc = xform_point(mat=curr_mvp, point=prev_pos_lc)
    #
    # 		curr_y, curr_x = int(curr_pos_ndc[1] * h), int(curr_pos_ndc[0] * w)
    #
    # 		# convert to stratum resolution
    # 		curr_sx, curr_sy = curr_x // downsample, curr_y // downsample
    #
    # 		accept = True
    # 		_curr_depth = curr_depth[curr_y, curr_x]
    # 		_prev_depth = prev_depth[prev_y, prev_x]
    # 		_curr_depth_gradient = max(ddx_(curr_depth, curr_x, curr_y), ddy_(curr_depth, curr_x, curr_y))
    #
    # 		accept = accept and test_reprojected_depth(_prev_depth, _curr_depth, _curr_depth_gradient)
    #
    # 		_curr_normal = curr_normal[curr_y, curr_x]
    # 		_prev_normal = prev_normal[prev_y, prev_x]
    # 		accept = accept and test_reprojected_normal(_prev_normal, _curr_normal)
    #
    # 		if not accept:
    # 			continue
    #
    # 		prev_shading_samples[curr_sy, curr_sx] = luminance(prev_illum[prev_y, prev_x])
    # 		curr_shading_samples[curr_sy, curr_sx] = luminance(curr_illum[curr_y, curr_x])
    #
    # 		depth[curr_sy, curr_sx] = curr_depth[curr_y, curr_x]
    # 		normal[curr_sy, curr_sx] = curr_normal[curr_y, curr_x]
    #
    #
    # numer = np.abs(curr_shading_samples - prev_shading_samples)
    # denom = np.maximum(curr_shading_samples, prev_shading_samples)

    numer = jnp.reshape(numer, newshape=(h // downsample, w // downsample))
    denom = jnp.reshape(denom, newshape=(h // downsample, w // downsample))
    depth = jnp.reshape(s["depth"], newshape=(h // downsample, w // downsample))
    normal = jnp.reshape(s["normal"], newshape=(h // downsample, w // downsample, 3))

    return numer, denom, depth, normal


# compute average of luminance inside each of 3x3 strata and use it initial luminance
# compute init variance by using the first and second moments of all values inside 3x3 strata
def init_stratum_data(illum_data):

    def tile_init_stratum_data(illum):
        intra_stratum_lum = luminance_vec(illum)
        return jnp.mean(intra_stratum_lum), jnp.mean(jnp.square(intra_stratum_lum)) - jnp.square(jnp.mean(intra_stratum_lum))

    return vmap(tile_init_stratum_data)(illum_data)


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
def reconstruct_max_pool(data, downsample=3, filter_radius=2):
    down_h, down_w = data.shape[0], data.shape[1]
    orig_h, orig_w = down_h * downsample, down_w * downsample

    recon_data = jnp.zeros((orig_h, orig_w))
    data = jnp.pad(data, filter_radius, constant_values=-1)

    init_s = dict(data=data, recon_data=recon_data)

    def step(s, idx):
        y, x = idx
        # stratum indices
        sy, sx = y // downsample, x // downsample
        sy += filter_radius
        sx += filter_radius

        def inner_step(s, inner_idx):
            yy, xx = inner_idx

            # local stratum indices used within this loop nest
            lsy, lsx = sy + yy, sx + xx

            s["recon_data"] = s["recon_data"].at[y, x].set(jnp.maximum(s["recon_data"][y, x], s["data"][lsy, lsx]))
            return s, None

        local_idx_array = indices_array(filter_radius * 2 + 1, start=-filter_radius)
        s, _ = scan(inner_step, s, local_idx_array)

        return s, None

    idx_array = indices_array(orig_h)
    s, _ = scan(step, init_s, idx_array)


    return s["recon_data"]

def compute_adaptive_alpha(illum_lst, depth_lst, normal_lst, vbuffer_lst, viewproj_lst, model_mats_lst, models_lst,
                              downsample=3):

    hh, ww = depth_lst[0].shape
    curr_illum = jnp.array(illum_lst[1])
    illum_data = data_prep_non_overlapping(curr_illum, radius=1)
    lum, variance = init_stratum_data(illum_data)
    lum = jnp.reshape(lum, newshape=(hh // downsample, ww // downsample))
    variance = jnp.reshape(variance, newshape=(hh // downsample, ww // downsample))

    # random intra stratum indices : these are generated beforehand and provided as input because it is non-trivial to
    # handle this in jax.More info can be found here https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html
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

    numer, denom, depth, normal = forward_projection(illum_lst, depth_lst, normal_lst, vbuffer_lst, viewproj_lst,
                                                     model_mats_lst, jax_models_lst, random_is_idxs, hh, ww,
                                                     downsample=downsample)

    depth_grad = compute_depth_gradient(depth)
    temp_grad = jnp.stack([numer, denom], axis=2)

    box_filter = jnp.array(generate_box_filter())
    filtered_temp_grad = multiple_iter_atrous_grad_decomposition(
        temp_grad, lum, variance, depth, normal, depth_grad, box_filter)

    lamda = jnp.minimum(1, filtered_temp_grad[:, :, 0] / filtered_temp_grad[:, :, 1])

    adaptive_alpha = (1 - lamda) * global_alpha + lamda
    recon_adaptive_alpha = reconstruct_max_pool(adaptive_alpha)

    return recon_adaptive_alpha




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


    aa = compute_adaptive_alpha(illum_lst, depth_lst, normal_lst, vbuffer_lst, viewproj_lst, model_mats_lst,
                                   models_lst)
    write_exr_file(join(output_path, "aa.exr"), aa)

    #
    # downsample = 3
    # hh, ww = depth_lst[0].shape
    #
    # curr_illum = jnp.array(illum_lst[1])
    # illum_data = data_prep_non_overlapping(curr_illum, radius=1)
    # lum, variance = init_stratum_data(illum_data)
    # lum = jnp.reshape(lum, newshape=(hh // downsample, ww // downsample))
    # variance = jnp.reshape(variance, newshape=(hh // downsample, ww // downsample))
    #
    # # random intra stratum indices : these are generated beforehand and provided as input because it is non-trivial to
    # # handle this in jax.More info can be found here https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html
    # random_is_idxs = jnp.array(random.choices(population=np.arange(downsample ** 2),
    #                                 k= (depth_lst[0].shape[0]//downsample) * (depth_lst[0].shape[1] // downsample) ))
    #
    # jax_models_lst = []
    # for f in range(2):
    #     models = models_lst[f]
    #
    #     jax_models = []
    #     max_k = -1
    #     for i in range(len(models)):
    #         model = models[i]
    #         jax_model = []
    #
    #         # iterate over two keys, vertices and faces
    #         for j in range(len(model.keys())):
    #             key = list(model.keys())[j]
    #             vert_faces = model[key]
    #             max_k = max(max_k, len(vert_faces))
    #
    #             jax_vert_faces = []
    #
    #             for k in range(len(vert_faces)):
    #                 item = list(vert_faces[k])
    #
    #                 jax_item = []
    #                 for l in range(len(item)):
    #                     jax_item.append(item[l])
    #
    #                 jax_vert_faces.append(jax_item)
    #
    #             jax_model.append(jax_vert_faces)
    #
    #         jax_models.append(jax_model)
    #
    #
    #
    #     for i in range(len(jax_models)):
    #         for j in range(len(jax_models[i])):
    #             for k in range(max_k - len(jax_models[i][j])):
    #                 jax_models[i][j].append([0.0] * len(jax_models[i][j][0]))
    #
    #     jax_models_lst.append(jnp.array(jax_models))
    #
    #
    # numer, denom, depth, normal = forward_projection(illum_lst, depth_lst, normal_lst, vbuffer_lst, viewproj_lst,
    #                                                  model_mats_lst, jax_models_lst, random_is_idxs, hh, ww,
    #                                                  downsample=downsample)
    #
    # # write_exr_file(join(output_path, "numer.exr"), numer)
    # # write_exr_file(join(output_path, "denom.exr"), denom)
    # # write_exr_file(join(output_path, "lum.exr"), lum)
    # # write_exr_file(join(output_path, "variance.exr"), variance)
    # # write_exr_file(join(output_path, "depth.exr"), depth)
    # # write_exr_file(join(output_path, "normal.exr"), normal)
    #
    # depth_grad = compute_depth_gradient(depth)
    # temp_grad = jnp.stack([numer, denom], axis=2)
    #
    # box_filter = jnp.array(generate_box_filter())
    # filtered_temp_grad = multiple_iter_atrous_grad_decomposition(
    #     temp_grad, lum, variance, depth, normal, depth_grad, box_filter)
    #
    #
    # # write_exr_file(join(output_path, "filtered_temp_grad.exr"), filtered_temp_grad)
    #
    # lamda = jnp.minimum(1, filtered_temp_grad[:, :, 0] / filtered_temp_grad[:, :, 1])
    # # write_exr_file(join(output_path, "lamda.exr"), lamda)
    #
    # adaptive_alpha = compute_adaptive_alpha(lamda)
    # # write_exr_file(join(output_path, "adaptive_alpha.exr"), adaptive_alpha)
    #
    # recon_adaptive_alpha = reconstruct_max_pool(adaptive_alpha)
    # # write_exr_file(join(output_path, "recon_adaptive_alpha.exr"), recon_adaptive_alpha)
    #




