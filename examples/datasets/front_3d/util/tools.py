import numpy as np
from mathutils import Matrix, Vector, Euler
import blenderproc as bproc
import math
import json

def look_at(cam_location, point):
    direction = point - cam_location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    return rot_quat.to_matrix()

def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed", "shelf", "cabinet", "desk", "pier", "stool", "lighting"]:
        if category_name in name.lower():
            return True
    return False


def compare_depth_maps(data1, data2, threshold=0.01):
    depth1 = np.array(data1.get('depth'))
    depth2 = np.array(data2.get('depth'))
    
    depth_diff = np.abs(depth2 - depth1)

    significant_change = depth_diff > threshold
    significant_change_count = np.sum(significant_change)

    total_pixels = depth1.size
    proportion = significant_change_count / total_pixels
    
    return proportion

def compute_back_indice(data, data_back, threshold=0.01):
    data = np.array(data.get('depth'))
    data_back = np.array(data_back.get('depth'))
    data = data.squeeze()
    data_back = data_back.squeeze()
    difference = np.abs(data - data_back)
    indices = np.where(difference < threshold)
    one_dim_indices = np.ravel_multi_index(indices, dims=data.shape)

    return one_dim_indices


def compute_frame_offset_similarity(offset, pre_offset):
    """
    Compute the average offset between two animation frames.

    :param offset: Current frame offsets (N x 3).
    :param pre_offset: Previous frame offsets (N x 3).
    :return: The average offset distance.
    """
    # Ensure both offsets have the same shape
    offset = np.array(offset)
    pre_offset = np.array(pre_offset)
    assert offset.shape == pre_offset.shape, "Offset arrays must have the same shape"
    
    # Calculate the difference between current and previous offsets
    offset_diff = offset - pre_offset
    # Compute the Euclidean distance for each point
    distances = np.linalg.norm(offset_diff, axis=1)
    
    # Compute the average distance
    average_offset_distance = np.mean(distances)
    
    return average_offset_distance


def getcameralocation(center_location, camera_location, pitch_range=(-7.5, 7.5), yaw_range=(-5.5, 5.5), location_offset_range=0.35):
    random_offset = np.random.uniform(-location_offset_range, location_offset_range, size=3)
    new_camera_location = camera_location + Vector(random_offset)
    if new_camera_location[2] >1.8:
        new_camera_location[2] = 1.8
    elif new_camera_location[2] <1.1:
        new_camera_location[2] = 1.1
    if center_location[2] > 1.6:
        center_location[2] = 1.6

    cam_rot = look_at(new_camera_location, center_location)
    
    # # add ramdom pitch and yaw
    # pitch = np.random.uniform(np.radians(pitch_range[0]), np.radians(pitch_range[1]))
    # yaw = np.random.uniform(np.radians(yaw_range[0]), np.radians(yaw_range[1]))

    # pitch_matrix = Matrix.Rotation(pitch, 3, 'X')
    # yaw_matrix = Matrix.Rotation(yaw, 3, 'Z')

    # cam_rot = yaw_matrix @ pitch_matrix @ cam_rot

    cam2world_matrix = bproc.math.build_transformation_mat(new_camera_location, cam_rot)
    
    return cam2world_matrix

def compute_overlap(pc1, pc2, voxel_size=0.0375):
    def voxelize(pc, voxel_size):
        voxel_grid = np.floor(pc / voxel_size).astype(np.int32)
        return voxel_grid
    
    voxel_pc1 = voxelize(pc1, voxel_size)
    voxel_pc2 = voxelize(pc2, voxel_size)

    set_pc1 = set(map(tuple, voxel_pc1))
    set_pc2 = set(map(tuple, voxel_pc2))
    
    intersection = set_pc1 & set_pc2
    #union = set_pc1 | set_pc2
    
    overlap_pc1 = len(intersection) / len(set_pc1)
    overlap_pc2 = len(intersection) / len(set_pc2)
    return overlap_pc1, overlap_pc2

