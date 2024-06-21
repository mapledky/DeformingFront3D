import argparse
import os
import numpy as np
import open3d as o3d
import h5py
import json
import math
"""
python code/BlenderProc/examples/datasets/front_3d/util/depth2point.py dataset/3D-Deforming-FRONT/rawdata/pro20
python code/BlenderProc/examples/datasets/front_3d/util/depth2point.py code/BlenderProc/examples/datasets/front_3d/output
"""

def getDepth(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        depth_data = f['depth'][()]
        return depth_data

def compute_ground(json1,json2):
    cam_info1 = None
    cam_info2 = None
    with open(json1, 'r', encoding='utf-8') as f:
        cam_info1 = json.load(f)
    with open(json2, 'r', encoding='utf-8') as f:
        cam_info2 = json.load(f)
    matrix1 = np.array(cam_info1.get('matrix_world'))
    matrix2 = np.array(cam_info2.get('matrix_world'))
    res_rot = matrix1[:3, :3] @ matrix2[:3, :3].T
    res_trans = matrix1[:3, 3][None, :] - (res_rot @ matrix2[:3, 3][None, :].T).T

    gt_transform = np.concatenate((res_rot, res_trans.T), axis=1)
    gt_transform = np.concatenate((gt_transform, np.array([[0, 0, 0, 1]])), axis=0)
    return gt_transform

def numpy_depth_to_pointcloud(depth_array, json_file):
    # 将NumPy数组转换为Open3D图像
    depth = np.array(depth_array)
    cam_info = None
    with open(json_file, 'r', encoding='utf-8') as f:
        cam_info = json.load(f)
    angle_x = cam_info.get('angle_x')
    cam_clip = cam_info.get('clip')
    matrix = np.array(cam_info.get('matrix_world'))

    factor = 2.0 * math.tan(angle_x / 2.0)

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    # Valid depths are defined by the camera clipping planes
    valid = (depth > cam_clip[0]) & (depth < cam_clip[1])
    # Negate Z (the camera Z is at the opposite)
    z = -np.where(valid, depth, np.nan)
    # z = -np.where(valid, depth, 10)
    # Mirror X
    # Center c and r relatively to the image size cols and rows
    ratio = max(rows, cols)
    x = -np.where(valid, factor * z * (c - (cols / 2)) / ratio, 0)
    y = np.where(valid, factor * z * (r - (rows / 2)) / ratio, 0)

    pc = np.dstack((x, y, z))
    pc = pc.reshape(-1, 3)
    valid = valid.reshape(-1)
    pc = pc[valid]

    pc = (matrix[:3, :3] @ pc.T).T + matrix[:3, 3][None, :]

    return pc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir",default="dataset/3D-Deforming-FRONT/rawdata/pro20", help="Path to where the data should be saved")
    args = parser.parse_args()

    # 遍历指定目录下的所有子目录
    for root, dirs, files in os.walk(args.output_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            print(subdir_path)
            file_src_depth = os.path.join(subdir_path, "0.hdf5")
            file_ref_depth = os.path.join(subdir_path, "1.hdf5")
            file_src_w = os.path.join(subdir_path, "src_open3d_w")
            file_ref_w = os.path.join(subdir_path, "ref_open3d_w")
            file_relative_trans = os.path.join(subdir_path, "relative_transform.npy")
            file_world_out = os.path.join(subdir_path, "world_out.pcd")
            file_src_json = os.path.join(subdir_path, "src_param.json")
            file_ref_json = os.path.join(subdir_path, "ref_param.json")
            depth_src = getDepth(file_src_depth)
            depth_ref = getDepth(file_ref_depth)


            src_world = numpy_depth_to_pointcloud(depth_src,file_src_json)
            ref_world = numpy_depth_to_pointcloud(depth_ref, file_ref_json)
            gt_transform = compute_ground(file_src_json, file_ref_json)

            o3d_pc1 = o3d.geometry.PointCloud()
            o3d_pc1.points = o3d.utility.Vector3dVector(src_world)
            o3d_pc2 = o3d.geometry.PointCloud()
            o3d_pc2.points = o3d.utility.Vector3dVector(ref_world)
            
            o3d_pc1 = o3d_pc1.voxel_down_sample(voxel_size=0.03)
            o3d_pc2 = o3d_pc2.voxel_down_sample(voxel_size=0.03)

            o3d_pc1.paint_uniform_color([1, 0, 0])  # red
            o3d_pc2.paint_uniform_color([0, 0, 1])  # blue
            
            combined_pcd = o3d_pc1 + o3d_pc2
            
            o3d.io.write_point_cloud(file_world_out, combined_pcd)