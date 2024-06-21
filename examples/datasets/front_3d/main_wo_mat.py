import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy
import json
import datetime
from mathutils import Matrix, Vector, Euler
from blenderproc.python.types.EntityUtility import Entity
import sys
sys.path.append('code/BlenderProc/examples/datasets/front_3d')

from util.hdftorgb import save_normals_as_image

"""
blenderproc run code/BlenderProc/examples/datasets/front_3d/main_wo_mat.py dataset/3D-FRONT/0a8d471a-2587-458a-9214-586e003e9cf9.json dataset/3D-FUTURE-model code/BlenderProc/examples/datasets/front_3d/output/test
""" 


def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed"]:
        if category_name in name.lower():
            return True
    return False

def degrees_to_radians(degrees):
    """将度转换为弧度"""
    return degrees * np.pi / 180

def getcameralocation(matrix, location_offset_range=0.15):
    camera_location = Matrix(matrix).to_translation()
    cam_rot = np.array(matrix)[:3,:3]
    random_offset = np.random.uniform(-location_offset_range, location_offset_range, size=3)
    new_camera_location = camera_location + Vector(random_offset)

    # 随机生成pitch和yaw角度（在-5到5度之间），并转换为弧度
    pitch_angle = degrees_to_radians(np.random.uniform(-15, 15))
    yaw_angle = degrees_to_radians(np.random.uniform(-10, 10))

    # 构造pitch旋转矩阵
    R_pitch = np.array([[np.cos(pitch_angle), 0, np.sin(pitch_angle)],
                        [0, 1, 0],
                        [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]])

    # 构造yaw旋转矩阵
    R_yaw = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle), 0],
                    [np.sin(yaw_angle), np.cos(yaw_angle), 0],
                    [0, 0, 1]])

    # 通过矩阵乘法得到最终的旋转矩阵
    R = R_yaw @ R_pitch @ cam_rot

    cam2world_matrix = bproc.math.build_transformation_mat(new_camera_location, R)
    
    return cam2world_matrix


parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("output_dir", help="Path to where the data should be saved")
parser.add_argument("--flow_skip", type=int, default=1, help="Frame skip interval for animation")


args = parser.parse_args()
args.output_dir =  args.output_dir

if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

# load the front 3D objects without textures
loaded_objects = bproc.loader.load_front3d_wo_mat(
    json_path=args.front,
    future_model_path=args.future_folder,
    label_mapping=mapping
)

# Init sampler for sampling locations inside the loaded front3D house
point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

poses = 0
tries = 0


# filter some objects from the loaded objects, which are later used in calculating an interesting score
special_objects = [obj.get_cp("category_id") for obj in loaded_objects if check_name(obj.get_name())]

proximity_checks = {"min": 3.0, "avg": {"min": 3.0, "max": 20}, "no_background": True}
pre_matrix = None
while tries < 10000 and poses < 1:
    # Sample point inside house
    height = np.random.uniform(1.4, 1.8)
    location = point_sampler.sample(height)
    # Sample rotation (fix around X and Y axis)
    rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

    # Check that obstacles are at least 1 meter away from the camera and have an average distance between 2.5 and 3.5
    # meters and make sure that no background is visible, finally make sure the view is interesting enough
    if bproc.camera.scene_coverage_score(cam2world_matrix, special_objects, special_objects_weight=10.0) > 0.8 \
            and bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
        bproc.camera.add_camera_pose(cam2world_matrix)
        pre_matrix = cam2world_matrix
        poses += 1
    tries += 1

new_matrix = getcameralocation(pre_matrix)

cam = bpy.data.objects["Camera"]
cam.data.clip_start = 0.25
cam.data.clip_end = 5.0
# Also render normals
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(128)

my_cam_infos_pre = {
            "angle_x": cam.data.angle_x,
            "clip": [cam.data.clip_start, cam.data.clip_end],
            "matrix_world": np.array(pre_matrix).tolist()
        }

bproc.camera.add_camera_pose(new_matrix)

my_cam_infos_now = {
            "angle_x": cam.data.angle_x,
            "clip": [cam.data.clip_start, cam.data.clip_end],
            "matrix_world": np.array(new_matrix).tolist()
        }


data = bproc.renderer.render()
# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)

# Iterate over each .hdf5 file in the data folder
for hdf5_file in os.listdir(args.output_dir):
    if hdf5_file.endswith(".hdf5"):
        hdf5_file_path = os.path.join(args.output_dir, hdf5_file)
        image_name = os.path.splitext(hdf5_file)[0] + '_normals.png'  # Construct image name from HDF5 file name
        save_normals_as_image(hdf5_file_path, args.output_dir, image_name)

src_file = os.path.join(args.output_dir, "src_param.json")
ref_file = os.path.join(args.output_dir, "ref_param.json")

with open(src_file, 'w', encoding='utf-8') as f:
    json.dump(my_cam_infos_pre, f, ensure_ascii=False, indent=4)
with open(ref_file, 'w', encoding='utf-8') as f:
    json.dump(my_cam_infos_now, f, ensure_ascii=False, indent=4)
