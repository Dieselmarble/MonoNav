import open3d as o3d
import numpy as np


def load_PCD_in_PLY(filename):
    pcd_load = o3d.io.read_point_cloud(filename)
    xyz_load = np.asarray(pcd_load.points)
    #o3d.visualization.draw_geometries([pcd_load])
    return pcd_load

def remove_outliers_in_PCD(pcd):
    """
    nb_neighbors, which specifies how many neighbors are taken into account
    std_ratio, tandard deviation of the average distances across the point cloud.
    """

    print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50,
                                                        std_ratio=1.5)
    display_inlier_outlier(pcd, ind)


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

if __name__ == '__main__':
    pcd = load_PCD_in_PLY("corner.pn-pointCloud.ply")
    #voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.05)
    # o3d.visualization.draw_geometries([voxel_down_pcd])
    remove_outliers_in_PCD(pcd)