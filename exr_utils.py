import cv2
import OpenEXR, array

def read_exr_file(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(img.shape) == 3:
        img = img[:, :, ::-1]
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
