#!/usr/camera/env python3
import cv2
import numpy as np
from PIL import Image
import imageio
import struct
import os
import open3d as o3d

def get_pointcloud_old(color_image,depth_image,camera_intrinsics):
    """ creates 3D point cloud of rgb images by taking depth information
        input : color image: numpy array[h,w,c], dtype= uint8
                depth image: numpy array[h,w] values of all channels will be same
        output : camera_points, color_points - both of shape(no. of pixels, 3)
    """

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,image_width-1,image_width),
                                  np.linspace(0,image_height-1,image_height))
    camera_points_x = np.multiply(pixel_x-camera_intrinsics[0,2],depth_image/camera_intrinsics[0,0])
    camera_points_y = np.multiply(pixel_y-camera_intrinsics[1,2],depth_image/camera_intrinsics[1,1])
    camera_points_z = depth_image
    camera_points = np.array([camera_points_x,camera_points_y,camera_points_z]).transpose(1,2,0).reshape(-1,3)

    color_points = color_image.reshape(-1,3)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
    camera_points = camera_points[valid_depth_ind,:]
    color_points = color_points[valid_depth_ind,:]

    return camera_points,color_points


def get_pointcloud(color_image, depth_image, camera_intrinsics):

    """
        input : color image: numpy array[h,w,c]
                depth image: numpy array[h,w]
        output : camera_points, color_points - both of shape(no. of pixels, 3)
    """

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]

    xx, yy = np.meshgrid(np.arange(image_width), np.arange(image_height))

    pixels = np.stack((xx*depth_image, yy*depth_image, depth_image), axis=2).reshape(-1,3)

    camera_points = np.dot(np.linalg.inv(camera_intrinsics), pixels.T).T
    color_points = color_image.reshape(-1,3)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
    camera_points = camera_points[valid_depth_ind,:]
    color_points = color_points[valid_depth_ind,:]

    return camera_points, color_points

def write_pointcloud(filename, xyz_points, rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'w')
    fid.write('ply\n')
    fid.write('format ascii 1.0\n')
    fid.write('element vertex %d\n'%xyz_points.shape[0])
    fid.write('property float32 x\n')
    fid.write('property float32 y\n')
    fid.write('property float32 z\n')
    fid.write('property uchar red\n')
    fid.write('property uchar green\n')
    fid.write('property uchar blue\n')
    fid.write('end_header\n')

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(" ".join([str(x) for x in [xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0],rgb_points[i,1],
                                     rgb_points[i,2]]])+'\n')

    fid.close()



def normalize_pointCloud(points):
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # substract offset from each direction
    x_points -= np.mean(x_points)
    y_points -= np.mean(y_points)
    z_points -= np.mean(z_points)

    max_value = np.max((abs(x_points).max(), abs(y_points).max(), abs(z_points).max()))

    # Normalize each direction to between 0 and 255
    #x_points = x_points / (max_value / 255.0)
    #y_points = y_points / (max_value / 255.0)
    #z_points = z_points / (max_value / 255.0)

    normalized_points = np.stack((x_points, y_points, z_points), axis=1)
    return normalized_points

def project_pointCloud_2_XZ_plane(points):
    points[:,1] = 0
    return points


def wirte_PCD_with_O3d(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud("o3d/sync.ply", pcd)
    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("o3d/sync.ply")
    o3d.visualization.draw_geometries([pcd_load])


def affine_transform(xyz, camera_extrinsics):
    homo_xyz = np.vstack((xyz.T, np.ones(xyz.shape[0])))
    transformed_xyz = np.dot(camera_extrinsics, homo_xyz)
    transformed_xyz = np.delete(transformed_xyz, 3 , axis = 0).T
    return transformed_xyz

def getCameraParameters(filename):
    chess_w = 7
    chess_h = 7

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess_w*chess_h,3), np.float32)
    objp[:,:2] = 30*np.mgrid[0:chess_w, 0:chess_h].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners, (9,9), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (chess_w, chess_h), corners2[::-1], ret)
        cv2.imshow('img',img)
        cv2.waitKey(200)

    # camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    # project chessboard to a virtual camera at right above the img
    imgpoints2, _ = cv2.projectPoints(objp, np.asarray([0, 0, 0], dtype=np.float32), np.asarray([0, 0, 700], dtype=np.float32), mtx, dist)

    tmp1 = imgpoints2
    tmp2 =  np.reshape(corners2[::-1], [-1,2])
    trans2,_ = cv2.findHomography(tmp2, tmp1)

    dat = cv2.warpPerspective(img, trans2, (1280, 720))
    cv2.imshow("perspect transformed", dat)
    cv2.waitKey(0)

    print("Number of valid views: " + str(len(rvecs)))

    print("Camera Intrinsics \n" + str(mtx))
    cv2.destroyAllWindows()
    print("kkk")
    return mtx, rvecs, tvecs


if __name__ == '__main__':
    rgb_filename = 'calib_images/2021_10_13/calib2.jpg'
    depth_filename = '../Results/flat_depth.npy'
    output_directory = './'
    w = 1280
    h = 720

    color_data = imageio.imread(rgb_filename)
    color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
    color_data = np.uint8(np.clip((cv2.add(1 * color_data, 30)), 0, 255))

    # color_data = np.asarray(im_color, dtype = "uint8")

    if os.path.splitext(os.path.basename(depth_filename))[1] == '.npy':
        depth_data = np.load(depth_filename)
    else:
        im_depth = imageio.imread(depth_filename)
        depth_data = np.asarray(im_depth)


    """
     camera_intrinsics  = [[fx 0 cx],
                           [0 fy cy],
                           [0 0 1]]
     p[639.574 363.718]  f[639.511 639.511]
    """

    #depth_data = depth_data[0,0,:]
    # 相机内参
    cam_fx = 639.511
    cam_fy = 639.511
    cam_cx = 639.574
    cam_cy = 363.718
    camera_intrinsics  = np.asarray([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])

    filename = os.path.basename(rgb_filename)[:9] + '-pointCloud.ply'
    output_filename = os.path.join(output_directory, filename)

    color_data = cv2.resize(color_data, (w, h), interpolation=cv2.INTER_NEAREST)
    depth_data = cv2.resize(depth_data, (w, h), interpolation=cv2.INTER_NEAREST)

    print("Creating the point Cloud file at : ", output_filename )
    camera_points, color_points = get_pointcloud(color_data, depth_data, camera_intrinsics)

    _, rvecs, tvecs = getCameraParameters(rgb_filename)
    rot_mat, _ = cv2.Rodrigues(rvecs[0])
    camera_extrinsics = np.vstack((np.hstack((rot_mat, tvecs[0])), [0,0,0,1]))
    transformed_xyz = affine_transform(camera_points, camera_extrinsics)

    #wirte_PCD_with_O3d(transformed_xyz)
    #wirte_PCD_with_O3d(camera_points)

    # camera_points = normalize_pointCloud(camera_points)
    # pointCloud_2_BirdView(camera_points)
    # camera_points = project_pointCloud_2_XZ_plane(camera_points)
    # write_pointcloud(output_filename, camera_points, color_points)
    #write_pointcloud(output_filename, transformed_xyz, color_points)