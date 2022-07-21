import numpy as np
import cv2
from mono_depth.Process_PointCloud import load_PCD_in_PLY

def scale_to_255(a, min, max, dtype = np.uint8):
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def pointCloud_2_BirdView(points):
    res = 0.01
    side_range = (-10, 10)  # left-most to right-most
    fwd_range = (0, 10)  # back-most to forward-most
    height_range = (-10, 10)

    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # convert to pixel position values - based on resolution
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)

    # shift pixels to have minimum to be 0,0
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    pixel_values = np.clip(a=z_points, a_min=height_range[0], a_max=height_range[1])

    pixel_values = scale_to_255(pixel_values, min=height_range[0], max=height_range[1])

    x_max = 1 + int((side_range[1] -  side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] -  fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    im[y_img, x_img] = pixel_values
    im += 10

    im = cv2.resize(im, (1280, 720), interpolation=cv2.INTER_NEAREST)
    cv2.namedWindow("Bird eye view", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Bird eye view", im)
    cv2.waitKey(20000)
    return

if __name__ == '__main__':
    pcd_load = load_PCD_in_PLY("corner.pn-pointCloud.ply")
    xyz_load = np.asarray(pcd_load.points)
    pointCloud_2_BirdView(xyz_load)