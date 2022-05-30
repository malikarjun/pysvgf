import cv2
import OpenEXR, array
from os.path import exists, join, basename
import numpy as np
import pywavefront
import shutil
from glob import glob

scene_path = "/Users/mallikarjunswamy/imp/acads/courses/winter-2022/CSE_272/lajolla_public/scenes/cbox"

def load_vbuffer(filename):
    val = np.load(filename)
    return np.stack([val[:, :, 0].T, val[:, :, 1].T, val[:, :, 2].T, val[:, :, 3].T], axis=2)

def load_models(model_fnames):
    models = []
    for fname in model_fnames:
        if fname == "":
            models.append(None)
        else:
            scene = pywavefront.Wavefront(join(scene_path, fname), collect_faces=True)
            models.append(scene)

    return models

def read_txt_file(filename):
    lines = None
    with open(filename) as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def get_file(filename):
    if exists(filename):
        return np.load(filename)
    return None

def read_exr_file(filepath, single_channel=False):
    img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(img.shape) == 3:
        img = img[:, :, ::-1]

    if single_channel:
        img = img[:, :, 0]
    return img

def write_exr_file(filepath, data):

    if len(data.shape) == 3:
        h, w, _ = data.shape
        exr = OpenEXR.OutputFile(filepath, OpenEXR.Header(h, w))
        r = array.array('f', data[:, :, 0].flatten().tolist())
        g = array.array('f', data[:, :, 1].flatten().tolist())
        b = array.array('f', data[:, :, 2].flatten().tolist())
        exr.writePixels({'R': r, 'G': g, 'B': b})
    else:
        h, w = data.shape
        exr = OpenEXR.OutputFile(filepath, OpenEXR.Header(h, w))
        r = array.array('f', data[:, :].flatten().tolist())
        exr.writePixels({'R': r, 'G': r, 'B': r})
