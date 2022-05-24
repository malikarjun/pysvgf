import numpy as np

def xform_point(mat, point):
	point = np.append(point, 1.0)
	xformed_point = mat @ point
	xformed_point = xformed_point/xformed_point[3]
	return xformed_point[:3]