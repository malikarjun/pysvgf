from os.path import join, exists
import os
from copy import deepcopy
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from file_utils import *

frame_base_path = "data"
output_path = "output"
inter_path = "intermediate_results"

g_phi_illum=4
g_phi_normal=128
g_phi_depth=3
global_alpha=0.2

def debug(i, j):
    return i == 210 and j == 125

# def debug(i, j):
#     return abs(i - 220) <= 1 and abs(j - 145) <= 1

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
    depth = np.pad(depth, (1,1), mode='edge')

    indices = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

    depth_grad = np.ones((h, w))
    # for i in tqdm(range(1, h + 1)):
    #     for j in range(1, w + 1):
    #         depth_center = depth[i, j]
    #         for idx in indices:
    #             depth_grad[i-1, j-1] += abs(depth_center - depth[i + idx[0], j + idx[1]])
    #         depth_grad[i-1, j-1] /= 8

    return depth_grad


def saturate(val):
    return max(0, min(val, 1))

def frac(val):
    return np.ceil(val) - val

def inside(p, _h, _w):
    return np.all(np.greater_equal(p, np.array([0, 0]))) and np.all(np.less(p, np.array([_h, _w])))

def lerp(a, b, frac):
    return a * (1 - frac) + b * frac

# TODO: we are not demodulating the albedo for A-SVGF, illum and color refer to the same thing
# TODO: no reprojection is happening now, fix this for dynamic camera and scenes
def temporal_integration(g_prev_illum, g_prev_moments, g_illum, g_moments, g_adapt_alpha):
    g_adapt_alpha = deepcopy(g_adapt_alpha)
    hh, ww, cc = g_illum.shape
    integrated_illum = deepcopy(g_illum)
    integrated_moments = deepcopy(g_moments)
    integrated_variance = np.zeros((hh, ww))

    offset = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    hh, ww, cc = g_prev_illum.shape
    print("running temporal integration...")
    for i in tqdm(range(hh)):
        for j in range(ww):

            # TODO : bilinear interpolation was causing issues at the boundary of spherical light

            # prev_illum = np.array([0, 0, 0]).astype(float)
            # prev_moments = np.array([0, 0]).astype(float)
            # # example (0, 0) -> (0.5, 0.5)
            # pos_prev = np.array([i, j]) + 0.5
            #
            # sumw = 0.0
            # y = frac(pos_prev[0])
            # x = frac(pos_prev[1])
            #
            # # bilinear weights
            # w = np.array([(1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y])
            #
            # # perform bilinear interpolation
            # for sample_idx in range(4):
            #     loc = pos_prev.astype(int) + offset[sample_idx]
            #     if inside(loc, hh, ww):
            #         prev_illum += w[sample_idx] * g_prev_illum[loc[0], loc[1]]
            #         prev_moments += w[sample_idx] * g_prev_moments[loc[0], loc[1]]
            #         sumw += w[sample_idx]
            #
            # prev_illum /= sumw
            # prev_moments /= sumw

            prev_illum = g_prev_illum[i , j]
            prev_moments = g_prev_moments[i, j]

            integrated_illum[i, j] = lerp(prev_illum, g_illum[i, j], g_adapt_alpha[i, j])
            integrated_moments[i, j] = lerp(prev_moments, g_moments[i, j], g_adapt_alpha[i, j])
            integrated_variance[i, j] = integrated_moments[i, j, 1] - np.square(integrated_moments[i, j, 0])

    return integrated_illum, integrated_moments, integrated_variance



def compute_weight(depth_center, depth_p, phi_depth, normal_center, normal_p, phi_normal, luminance_illum_center,
                   luminance_illum_p, phi_illum):
    weight_normal = pow(saturate(normal_center.dot(normal_p)), phi_normal)
    weight_z = 0.0 if phi_depth == 0 else abs(depth_center - depth_p) / phi_depth
    weight_l_illum = abs(luminance_illum_center - luminance_illum_p) / phi_illum
    weight_illum = np.exp(0.0 - max(weight_l_illum, 0.0) - max(weight_z, 0.0)) * weight_normal

    return weight_illum
    # return weight_illum



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
    # return np.repeat(variance[:, :, np.newaxis], 3, axis=2)
    return variance



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

            for yy in range(-radius, radius+1):
                for xx in range(-radius, radius+1):
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
            # filtered_color[i, j, :] = sum_w_illumination
            # print(sum_w_illumination)
            # print(sum_illumination)
            # exit(0)

            filtered_variance[i, j] = sum_variance
    return filtered_color, filtered_variance

if __name__ == '__main__':
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

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

    for i in range(5):
        step_size = 1 << i
        output_illum, output_var = compute_atrous_decomposition(input_illum, input_var, frame_depth[curr_frame],
                                                                frame_normal[curr_frame], frame_depth_grad[curr_frame],
                                                                g_step_size=step_size)
        write_exr_file(join(output_path, "iter{}_color.exr".format(i+1)), output_illum)
        write_exr_file(join(output_path, "iter{}_variance.exr").format(i+1), output_var)
        input_illum = deepcopy(output_illum)
        input_var = deepcopy(output_var)