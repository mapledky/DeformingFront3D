import numpy as np
from mathutils import Matrix, Vector, Euler
import blenderproc as bproc
import math
from sklearn.neighbors import NearestNeighbors, KDTree
import json
import random
from scipy.spatial.transform import Rotation


def compute_rt(cam_info1, cam_info2):
    matrix1 = np.array(cam_info1.get('matrix_world'))
    matrix2 = np.array(cam_info2.get('matrix_world'))
    res_rot = matrix1[:3, :3] @ matrix2[:3, :3].T
    res_trans = matrix1[:3, 3][None, :] - (res_rot @ matrix2[:3, 3][None, :].T).T

    gt_transform = np.concatenate((res_rot, res_trans.T), axis=1)
    gt_transform = np.concatenate((gt_transform, np.array([[0, 0, 0, 1]])), axis=0)
    return gt_transform


def transpoint(pc, gt_trans):
    res_rot = gt_trans[:3, :3]
    res_trans = gt_trans[:3, 3]
    pc = (res_rot @ pc.T).T + res_trans
    return pc


def depth2pointcloud(config, depth, cam_info, indices=[]):

    if isinstance(depth, list):
        depth = np.array(depth)
    elif not isinstance(depth, np.ndarray):
        raise TypeError("Depth must be a list or numpy array")

    depth = depth.squeeze()
    angle_x = cam_info.get('angle_x')
    matrix = np.array(cam_info.get('matrix_world'))

    factor = 2.0 * math.tan(angle_x / 2.0)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    # Valid depths are defined by the camera clipping planes
    valid = (depth > config.render_d2p_min_clip) & (depth < config.render_d2p_max_clip)
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

    #track_valid_indices
    valid_indices = np.arange(depth.size).reshape(-1)[valid]
    #[1,2,3,....] depthmap index after depth crop
    pc = (matrix[:3, :3] @ pc.T).T + matrix[:3, 3][None, :]
    voxel_indices = np.round(pc / config.render_d2p_voxel_size).astype(int)
    voxel_pointcloud_dict = {}
    voxel_indices_dict = {}#voxel之后的坐标对应原来的depth图中的二维坐标
    for i, idx in enumerate(voxel_indices):
        #idx:depth 值 /voxel_size
        key = tuple(idx)
        if key not in voxel_pointcloud_dict:
            voxel_pointcloud_dict[key] = pc[i]
            voxel_indices_dict[key] = valid_indices[i]

    voxel_pointcloud = np.array(list(voxel_pointcloud_dict.values()))

    #体素化下采样的n*3的每一个n对应原depth图的二维坐标，应该和indices是交集
    new_indices = np.array(list(voxel_indices_dict.values()))


    if len(indices) != 0:
        indices = np.array(indices)
        mask = np.isin(indices, valid_indices)
        new_indices = new_indices.tolist()
        indices = indices[mask]#背景的点在深度裁剪之后也存在的点,也是对原depth图的索引
        mapped_indices = []
        new_indices_set = set(new_indices)  # 将 new_indices 转换为集合
        new_indices_dict = {value: idx for idx, value in enumerate(new_indices)}  # 创建一个字典，键是 new_indices 的值，值是其索引

        for idx in indices:
            if idx in new_indices_set:
                mapped_indices.append(new_indices_dict[idx])
    else:
        mapped_indices = []
    # Remove flat areas from voxel_pointcloud
    updated_voxel_pointcloud, updated_indices = remove_flat_areas(voxel_pointcloud, mapped_indices, area_size=config.render_d2p_flat_area)
    return updated_voxel_pointcloud, updated_indices

def augment_point_cloud(points, aug_noise=0.01):
    points += (np.random.rand(points.shape[0], 3) - 0.5) * aug_noise
    return points

def save_point_cloud_to_pcd(point_cloud, file_path):
    pcd_file_path = file_path + ".pcd"
    with open(pcd_file_path, 'w') as f:
        # 写入 PCD 头部信息
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write("WIDTH %d\n" % point_cloud.shape[0])
        f.write("HEIGHT 1\n")
        f.write("POINTS %d\n" % point_cloud.shape[0])
        f.write("DATA ascii\n")
        
        # 写入点云数据
        for point in point_cloud:
            f.write("%f %f %f\n" % (point[0], point[1], point[2]))

def save_point_cloud_to_numpy_and_pcd(point_cloud, file_path_base):
    """
    Save point cloud data to both numpy and PCD files.

    :param point_cloud: The point cloud data (Nx3 numpy array).
    :param file_path_base: The base file path to save the files (without extension).
    """
    # 保存为 numpy 文件
    npy_file_path = file_path_base + ".npy"
    np.save(npy_file_path, point_cloud)
    
    # # 保存为 PCD 文件
    # pcd_file_path = file_path_base + ".pcd"
    # save_point_cloud_to_pcd(point_cloud, pcd_file_path)



def is_flat_area(points, threshold=0.01):
    """
    判断点云是否近似平坦。
    通过计算点云的协方差矩阵的特征值，判断点云是否平坦。
    """
    if len(points) < 3:
        return False
    
    # 计算协方差矩阵
    cov_matrix = np.cov(points, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    
    # 判断最小特征值是否小于阈值
    return eigenvalues[0] < threshold


def remove_flat_areas(voxel_pointcloud, mapped_indices, area_size=2.4):
    if len(mapped_indices) == 0:
        return voxel_pointcloud, mapped_indices
    """
    处理平坦区域，删除一半的点云。
    """
    voxel_pointcloud = np.array(voxel_pointcloud)
    mapped_indices = np.array(mapped_indices)
    
    # 使用 KDTree 来加速邻域搜索
    tree = KDTree(voxel_pointcloud)
    
    processed_indices = set()
    points_to_remove = set()
    
    for idx in mapped_indices:
        if idx in processed_indices:
            continue
        
        # 找到指定半径内的所有点
        point = voxel_pointcloud[idx]
        indices = tree.query_radius([point], r=area_size / 2)[0]
        # 只保留 mapped_indices 中的点
        indices = [i for i in indices if i in mapped_indices]
        # 检查这些点是否是平坦区域
        neighborhood_points = voxel_pointcloud[indices]
        if is_flat_area(neighborhood_points):
            # 删除一半的点云
            half_indices = indices[:len(indices) - (len(indices) // 3)]
            points_to_remove.update(half_indices)
        
        processed_indices.update(indices)
    
    
    # 删除指定的点云
    mask = np.ones(len(voxel_pointcloud), dtype=bool)
    mask[list(points_to_remove)] = False
    new_voxel_pointcloud = voxel_pointcloud[mask]
    
    old_to_new_index = np.cumsum(mask) - 1
    remaining_mapped_indices = [idx for idx in mapped_indices if mask[idx]]
    new_mapped_indices = [old_to_new_index[idx] for idx in remaining_mapped_indices]
    
    print('new_voxel ',len(new_voxel_pointcloud), 'new indices ', len(new_mapped_indices))
    return new_voxel_pointcloud, new_mapped_indices