import numpy as np
import jax.numpy as jnp

def xform_point(mat, point):
	point = jnp.append(point, 1.0)
	xformed_point = mat @ point
	xformed_point = xformed_point/xformed_point[3]
	return xformed_point[:3]