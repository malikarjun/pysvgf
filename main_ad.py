from os.path import join, exists
from copy import deepcopy
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from jax import grad, jit, lax
import jax.numpy as jnp

from learnable_utils import *

frame_base_path = "/Users/mallikarjunswamy/imp/acads/courses/winter-2022/CSE_272/lajolla_public/cmake-build-debug"
inter_path = "intermediate_results"

g_phi_illum=4
g_phi_normal=128
g_phi_depth=3
global_alpha=0.2

def debug(i, j):
    return i == 410 and j == 344

def luminance_vec(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def luminance(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

# lambda in paper
def relative_gradient(frame0, frame1):
    assert frame0.shape == frame1.shape
    # small epsilon to avoid NaNs
    eps = 1e-6
    frame0 = np.maximum(frame0, np.full(fill_value=eps, shape=frame0.shape))
    frame1 = np.maximum(frame1, np.full(fill_value=eps, shape=frame1.shape))

    rel_grad = np.minimum(np.abs((frame1 - frame0))/np.maximum(frame0, frame1), np.ones(frame0.shape))
    return luminance_vec(rel_grad[:, :, 0], rel_grad[:, :, 1], rel_grad[:, :, 2])

def test_reprojected_depth(z1, z2, dz):
    z_diff = abs(z1 - z2)
    return z_diff < 2.0 * (dz + 1e-3)

def test_reprojected_normal(n1, n2):
    return n1.dot(n2) > 0.9

def compute_adaptive_alpha(frame_depth, frame_normal, frame_depth_grad, frame=1):
    print("compute adaptive alpha")

    hh, ww = frame_depth[0].shape

    disocclusion = np.zeros((hh, ww))

    frame0 = read_exr_file(join(frame_base_path, "frame0.exr"))
    frame1 = read_exr_file(join(frame_base_path, "frame1.exr"))

    lamda = relative_gradient(frame0, frame1)
    alpha = (1 - lamda) * global_alpha + lamda

    prev_depth = frame_depth[frame-1]
    prev_normal = frame_normal[frame-1]
    curr_depth = frame_depth[frame]
    curr_normal = frame_normal[frame]
    curr_depth_grad = frame_depth_grad[frame]

    for i in tqdm(range(hh)):
        for j in range(ww):
            if not test_reprojected_depth(prev_depth[i, j], curr_depth[i, j], curr_depth_grad[i, j]) or \
                    not test_reprojected_normal(prev_normal[i, j], curr_normal[i, j]):
                alpha[i, j] = 1.0
                disocclusion[i, j] = 1

    return alpha, disocclusion


def compute_moments(color):
    moments = np.zeros((color.shape[0], color.shape[1], 2))
    moments[:, :, 0] = 0.2126 * color[:, :, 0] + 0.7152 * color[:, :, 1] + 0.0722 * color[:, :, 2]
    moments[:, :, 1] = np.square(moments[:, :, 0])
    return moments

# TODO: this is wrong. @trevor mentioned the right way to compute this using ray differentials
def compute_depth_gradient(depth):
    depth = deepcopy(depth)
    h, w = depth.shape

    depth_grad = np.ones((h, w))

    return depth_grad


def jnp_max(a, b):
    return jnp.maximum(jnp.array([a]), jnp.array([b]))[0]

def jnp_min(a, b):
    return jnp.minimum(jnp.array([a]), jnp.array([b]))[0]

def saturate(val):
    # return max(0, min(val, 1))
    return jnp.maximum(jnp.array([0]), jnp.minimum(jnp.array([val]), jnp.array([1])))[0]

def frac(val):
    return np.ceil(val) - val

def inside(p, _h, _w):
    return np.all(np.greater_equal(p, np.array([0, 0]))) and np.all(np.less(p, np.array([_h, _w])))

def lerp(a, b, frac):
    return a * (1 - frac) + b * frac

def temporal_integration(g_prev_illum, g_prev_moments, g_illum, g_moments, g_adapt_alpha):
    g_adapt_alpha = deepcopy(g_adapt_alpha)
    hh, ww, cc = g_illum.shape
    integrated_illum = deepcopy(g_illum)
    integrated_moments = deepcopy(g_moments)
    integrated_variance = np.zeros((hh, ww))

    hh, ww, cc = g_prev_illum.shape
    print("running temporal integration...")
    for i in tqdm(range(hh)):
        for j in range(ww):

            prev_illum = g_prev_illum[i , j]
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

    # return weight_illum
    return weight_illum



def compute_variance_spatially(frame_illum, frame_depth, frame_normal, frame_moments, frame_depth_grad, disocclusion,
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

            # TODO: revisit this
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

                        # TODO: how do we compute depth gradients
                        w = compute_weight(z_center, z_p, phi_depth * np.linalg.norm(np.array([yy, xx])),
                                           n_center, n_p, g_phi_normal,
                                           l_illumination_center, l_illumination_p, g_phi_illum)

                        sum_w_illumination += w
                        sum_moments += w * moments_p

            sum_w_illumination = max(sum_w_illumination, 1e-6)
            sum_moments /= sum_w_illumination

            variance[i, j] = sum_moments[1] - sum_moments[0] * sum_moments[0]
    return np.repeat(variance[:, :, np.newaxis], 3, axis=2)


_sum_illum = "sum_illum"
_sum_weight = "sum_weight"
_center_pos = "center_pos"
_illumination_center = "illumination_center"
_l_illumination_center = "l_illumination_center"
_z_center = "z_center"
_n_center = "n_center"
_var_center = "var_center"
_phi_depth = "phi_depth"
_phi_l_illumination = "phi_l_illumination"

def learnable_atrous_decomposition(illum, filter, variance, depth, normal, depth_grad, g_step_size):
    h, w, c = illum.shape

    filtered_img = jnp.zeros(illum.shape)
    radius = 2

    def func_micro(i, data_dict):
        _ii, _jj = i // (2 * radius + 1), i % (2 * radius + 1)
        ii, jj = i // (2*radius + 1) - radius, i % (2*radius + 1) - radius
        center_pos = data_dict[_center_pos]
        pos = center_pos + g_step_size * jnp.array([ii, jj])

        yy, xx = pos[0], pos[1]
        illumination_p = illum[yy, xx]
        # variance_p = variance[yy, xx]
        l_illumination_p = luminance(illumination_p)
        z_p = depth[yy, xx]
        n_p = normal[yy, xx]

        weight = compute_weight(
            data_dict[_z_center], z_p, data_dict[_phi_depth] * jnp.linalg.norm(jnp.array([ii, jj])),
            data_dict[_n_center], n_p, g_phi_normal,
            data_dict[_l_illumination_center], l_illumination_p, data_dict[_phi_l_illumination]
        ) * filter[_ii, _jj]

        vals = lax.cond(
            i != ((2 * radius + 1) ** 2) // 2,
            lambda : (illum[center_pos[0] + ii, center_pos[1] + jj], weight) ,
            lambda : (jnp.zeros(3), 0.0)
        )


        data_dict[_sum_illum] += vals[0] * vals[1]
        data_dict[_sum_weight] += vals[1]

        return data_dict

    def func_macro(i, filtered_img):
        ii, jj = i // w, i % w
        center_pos = jnp.array([ii, jj])

        data_dict = {
            _sum_illum: jnp.array([0.0, 0.0, 0.0]),
            _sum_weight: jnp.float32(0.0),
            _center_pos: center_pos,
            _illumination_center: illum[ii, jj],
            _l_illumination_center: luminance(illum[ii, jj]),
            _z_center: depth[ii, jj],
            _n_center : normal[ii, jj],
            _var_center : variance[ii, jj],
            _phi_depth : jnp_max(1e-8, depth_grad[ii, jj]) * g_phi_depth,
            _phi_l_illumination : g_phi_illum * jnp.sqrt(jnp_max(0.0, 1e-8 + variance[ii, jj]))
        }
        data_dict = lax.fori_loop(0, (2*radius + 1) ** 2, func_micro, data_dict)

        return filtered_img.at[ii, jj].set(data_dict[_sum_illum] / data_dict[_sum_weight])
        # return filtered_img.at[ii, jj].set(data_dict[_sum_illum])
        # return filtered_img.at[ii, jj].set(data_dict[_sum_weight])

    filtered_img = lax.fori_loop(radius * radius, (h - radius) * (w - radius), func_macro, filtered_img)

    return filtered_img

def compute_atrous_decomposition(illum, in_variance, depth, normal, depth_grad, g_step_size):
    print("computing atrous decomposition...")
    hh, ww, cc = illum.shape
    g_illumination = illum
    g_variance = deepcopy(in_variance)
    g_variance = gaussian_filter(g_variance, sigma=3, truncate=3)


    g_depth = depth
    g_normal = normal
    g_depth_grad = depth_grad
    kernel_weights = np.array([1.0, 2.0 / 3.0, 1.0 / 6.0])
    # TODO: the paper has different weights
    # kernel_weights = np.array([3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0])


    filtered_color = np.zeros((hh, ww, cc))
    filtered_variance = np.zeros((hh, ww))

    radius = 2

    for i in tqdm(range(hh)):
        for j in range(ww):

            # if not debug(i, j):
            #     continue

            ipos = np.array([i, j])
            # sum_w_illumination = kernel_weights[0]
            sum_w_illumination = 0.0
            # FIXME: best so far
            sum_illumination = np.array([0, 0, 0]).astype(float)
            # sum_illumination = g_illumination[i, j] * sum_w_illumination
            sum_variance = 0.0

            illumination_center = g_illumination[i, j]
            l_illumination_center = luminance(illumination_center)
            z_center = g_depth[i, j]
            n_center = g_normal[i, j]
            # this has been gaussian blurred
            var_center = g_variance[i, j]

            # TODO: revisit g_depth_grad
            phi_depth = max(1e-8, g_depth_grad[i, j]) * g_phi_depth

            # FIXME : best so far
            phi_l_illumination = g_phi_illum * np.sqrt(max(0.0, 1e-8 + var_center))
            # phi_l_illumination = g_phi_illum
            # phi_l_illumination = g_phi_illum * np.sqrt(max(0.0, 1.0 + g_variance[i, j]))

            for yy in range(-radius, radius):
                for xx in range(-radius, radius):
                    p = np.array([yy, xx]) * g_step_size + ipos
                    inside = np.all(np.greater_equal(p, np.array([0, 0]))) and np.all(np.less(p, np.array([hh, ww])))
                    kernel = kernel_weights[abs(xx)] * kernel_weights[abs(yy)]

                    if inside and (xx != 0 or yy != 0):
                        y, x = p[0], p[1]
                        illumination_p = g_illumination[y, x]
                        variance_p = g_variance[y, x]
                        l_illumination_p = luminance(illumination_p)
                        z_p = g_depth[y, x]
                        n_p = g_normal[y, x]

                        # TODO: how do we compute depth gradients
                        w = compute_weight(z_center, z_p, phi_depth * np.linalg.norm(np.array([yy, xx])),
                                           n_center, n_p, g_phi_normal,
                                           l_illumination_center, l_illumination_p, phi_l_illumination)

                        w_illumination = w * kernel
                        sum_w_illumination += w_illumination
                        sum_illumination += w_illumination * illumination_p
                        sum_variance += np.square(w_illumination) * variance_p

            sum_w_illumination = max(sum_w_illumination, 1e-6)

            sum_illumination /= sum_w_illumination
            sum_variance /= np.square(sum_w_illumination)

            filtered_color[i, j, :] = sum_illumination
            # print(sum_illumination)
            # exit(0)

            filtered_variance[i, j] = sum_variance
    return filtered_color, filtered_variance

if __name__ == '__main__':
    USE_TEMPORAL_ACCU = True

    frame_illum = []
    frame_depth = []
    frame_normal = []
    frame_moments = []
    frame_depth_grad = []

    for i in range(2):
        frame_illum.append(read_exr_file(join(frame_base_path, "frame{}.exr".format(i))))
        frame_depth.append(read_exr_file(join(frame_base_path, "frame{}_depth.exr".format(i)))[:, :, 0])
        frame_normal.append(read_exr_file(join(frame_base_path, "frame{}_normal.exr".format(i))))
        frame_moments.append(compute_moments(frame_illum[i]))
        frame_depth_grad.append(compute_depth_gradient(frame_depth[i]))

    prev_frame = 0
    curr_frame = 1

    adaptive_alpha_path = join(inter_path, "adaptive_alpha.npy")
    disocclusion_path = join(inter_path, "disocclusion.npy")
    adaptive_alpha = get_file(adaptive_alpha_path)
    disocclusion = get_file(disocclusion_path)
    if adaptive_alpha is None or disocclusion is None:
        adaptive_alpha, disocclusion = compute_adaptive_alpha(frame_depth, frame_normal, frame_depth_grad)
        np.save(adaptive_alpha_path, adaptive_alpha)
        np.save(disocclusion_path, disocclusion)

    integrated_illum_path = join(inter_path, "integrated_illum.npy")
    integrated_moments_path = join(inter_path, "integrated_moments.npy")
    integrated_variance_path = join(inter_path, "variance.npy")

    integrated_illum = get_file(integrated_illum_path)
    integrated_moments = get_file(integrated_moments_path)
    integrated_variance = get_file(integrated_variance_path)

    if integrated_illum is None or integrated_moments is None or integrated_variance is None:
        integrated_illum, integrated_moments, integrated_variance = temporal_integration(
            g_prev_illum=frame_illum[prev_frame], g_prev_moments=frame_moments[prev_frame],
            g_illum=frame_illum[curr_frame], g_moments=frame_moments[curr_frame], g_adapt_alpha=adaptive_alpha)

        integrated_variance = compute_variance_spatially(frame_illum, frame_depth, frame_normal, frame_moments,
                                                         frame_depth_grad, disocclusion, integrated_variance)

        np.save(integrated_illum_path, integrated_illum)
        np.save(integrated_moments_path, integrated_moments)
        np.save(integrated_variance_path, integrated_variance)

    frame_illum[curr_frame] = integrated_illum
    frame_moments[curr_frame] = integrated_moments

    input_illum = deepcopy(frame_illum[curr_frame])
    input_var = deepcopy(integrated_variance)

    for i in range(1):
        step_size = 1 << i

        # TODO: deepcopy before gaussian?
        input_var = gaussian_filter(input_var, sigma=3, truncate=3)

        input_illum = jnp.array(input_illum)
        input_var = jnp.array(input_var)
        input_depth = jnp.array(frame_depth[curr_frame])
        input_normal = jnp.array(frame_normal[curr_frame])
        input_depth_grad = jnp.array(frame_depth_grad[curr_frame])


        # TODO : replace this with the actual filter computed using 1d array
        # atrous_filter = jnp.ones((5, 5))
        atrous_filter = jnp.array(generate_atrous_kernel())
        learnable_atrous_decomposition = jit(learnable_atrous_decomposition
                                             # ,static_argnames=["depth", "normal", "depth_grad"]
                                             )

        output_illum = learnable_atrous_decomposition(input_illum, atrous_filter, input_var, input_depth,
                                                                input_normal, input_depth_grad,
                                                                g_step_size=step_size)
        write_exr_file("iter{}_color_ad.exr".format(i+1), output_illum)
        # write_exr_file("iter{}_variance.exr".format(i+1), output_var)
        # input_illum = deepcopy(output_illum)
        # input_var = deepcopy(output_var)
