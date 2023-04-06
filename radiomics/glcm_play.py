#3d glcm
#https://github.com/Prof-Iz/GLCM-from-3D-NumPy-Input

# gray-level invariant Haralick 
# https://github.com/patrik-brynolfsson/invariant-haralick-features/blob/master/GLCMFeaturesInvariant.m
# pxel coocurence loss https://openreview.net/attachment?id=rylrI1HtPr&name=original_pdf

def glcm_3d(input: np.ndarray, delta: tuple[int] = (1, 1, 1), d: int = 1):
    """_summary_

    Args:
        input (np.ndarray): input array. 3D. dtype int
        delta (tuple[int], optional): Direction vector from pixel. Defaults to (1, 1, 1).
        d (int, optional): Distance to check for neighbouring channel. Defaults to 1.

    Raises:
        Exception: if input is not of type dint or is not 3D

    Returns:
        _type_: GLCM Matrix
    """

    if 'int' not in input.dtype.__str__():
        raise Exception("Input should be of dtype Int")

    if len(input.shape) != 3:
        raise Exception("Input should be 3 dimensional")

    offset = (delta[0] * d, delta[1] * d, delta[2] * d)  # offset from each pixel

    x_max, y_max, z_max = input.shape  # boundary conditions during enumeration

    levels = input.max() + 1 # 0:1:n assume contn range of pixel values

    results = np.zeros((levels, levels))  # initialise results error


    for i, v in np.ndenumerate(input):
        x_offset = i[0] + offset[0]
        y_offset = i[1] + offset[1]
        z_offset = i[2] + offset[2]

        if (x_offset >= x_max) or (y_offset >= y_max) or (z_offset >= z_max):
            # if offset out of boundary skip
            continue

        value_at_offset = input[x_offset, y_offset, z_offset]

        results[v, value_at_offset] += 1

    return results / levels**2