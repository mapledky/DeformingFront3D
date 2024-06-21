import blenderproc as bproc
import os
import json
import numpy as np
import random
import argparse

"""
blenderproc run code/BlenderProc/examples/datasets/front_3d_object_sampling/main_shapenet.py dataset/3D-FRONT/0a482eb4-e8fa-4b44-90d0-3623e0a60c71.json dataset/3D-FUTURE-model code/BlenderProc/examples/datasets/front_3d/config/shape.json dataset/ShapeNetCore.v2

"""

parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument('shape_json', help="Path to the ShapeNet models JSON file")
parser.add_argument('shapenet_base_path', help="Base path to the ShapeNet dataset")
parser.add_argument('output_dir', nargs='?', default="code/BlenderProc/examples/datasets/front_3d_object_sampling/output", help="Path to where the final files will be saved")
args = parser.parse_args()

if not os.path.exists(args.front) or not os.path.exists(args.future_folder) or not os.path.exists(args.shape_json) or not os.path.exists(args.shapenet_base_path):
    raise OSError("One of the necessary files or folders does not exist!")

bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)


import os
import h5py
import numpy as np
import argparse
from PIL import Image


def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed", "shelf", "cabinet", "desk", "pier", "stool", "lighting"]:
        if category_name in name.lower():
            return True
    return False


def save_normals_as_image(hdf5_file_path: str, output_folder: str, image_name: str):
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as f:
        # Read the normals data
        normals_data = f['normals'][()]
    
    # Convert the range of the normals from [-1, 1] to [0, 255]
    normals_data = ((normals_data + 1) / 2 * 255).astype(np.uint8)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the normals data as an image
    image_path = os.path.join(output_folder, image_name)
    Image.fromarray(normals_data).save(image_path)
    print(f"Normals image saved to: {image_path}")

def scale_object_to_size(obj, max_size):
    # 计算对象的边界框尺寸
    bounding_box = obj.get_bound_box()
    min_corner = np.min(bounding_box, axis=0)
    max_corner = np.max(bounding_box, axis=0)
    dimensions = max_corner - min_corner
    max_dimension = max(dimensions)

    # 计算缩放因子
    scale_factor = max_size / max_dimension if max_dimension > max_size else 1.0

    # 应用缩放
    obj.set_scale([scale_factor, scale_factor, scale_factor])


# Load and parse the JSON file
with open(args.shape_json, 'r') as f:
    shape_data = json.load(f)

# Extract all synsetId
synset_ids = []
for item in shape_data:
    synset_ids.append(item['synsetId'])

#Collect all ShapeNet model paths
shapenet_paths = []
for synset_id in synset_ids:
    synset_dir = os.path.join(args.shapenet_base_path, synset_id)
    if os.path.exists(synset_dir):
        used_source_ids = os.listdir(synset_dir)
        for used_source_id in used_source_ids:
            shapenet_paths.append((synset_id, used_source_id))

print("shapenet_paths_objects " + str(len(shapenet_paths)))
# Randomly select 50 model IDs
random.shuffle(shapenet_paths)
selected_shapenet_models = shapenet_paths[:10]


# Set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200, transmission_bounces=200, transparent_max_bounces=200)

# Load the front 3D objects
room_objs = bproc.loader.load_front3d_wo_mat(
    json_path=args.front,
    future_model_path=args.future_folder,
    label_mapping=mapping
)

# Define the camera intrinsics
bproc.camera.set_resolution(512, 512)

# Select the objects, where other objects should be sampled on the floor
sample_surface_objects = [obj for obj in room_objs if "table" in obj.get_name().lower() or "desk" in obj.get_name().lower() or "bed" in obj.get_name().lower() or "sofa" in obj.get_name().lower()] 
random.shuffle(sample_surface_objects)
selected_surfaces = sample_surface_objects[:int(len(sample_surface_objects))]

print("************************* sample surface  " + str(len(selected_surfaces)))

for obj in selected_surfaces:
    print("_____________________ sample surface  " + str(obj.get_name()))
    with bproc.utility.UndoAfterExecution():
        # Select the surfaces, where the objects should be sampled on
        surface_obj = bproc.object.slice_faces_with_normals(obj)
        if surface_obj is None:
            continue

        def sample_pose(obj: bproc.types.MeshObject):
            # Sample the sphere's location above the surface
            obj.set_location(bproc.sampler.upper_region(
                objects_to_sample_on=[surface_obj],
                min_height=1,
                max_height=4,
                use_ray_trace_check=False
            ))
            # Randomize the rotation of the sampled object
            obj.set_rotation_euler(bproc.sampler.uniformSO3())

        dropped_object_list = []
        for synset_id, used_source_id in selected_shapenet_models:
            # Load the ShapeNet object, which should be sampled on the surface
            shapenet_obj = bproc.loader.load_shapenet(args.shapenet_base_path, used_synset_id=synset_id, used_source_id=used_source_id)
            scale_object_to_size(shapenet_obj, 0.3)
            dropped_objects = bproc.object.sample_poses_on_surface([shapenet_obj], surface_obj, sample_pose,
                                                                   min_distance=0.1, max_distance=10,
                                                                   check_all_bb_corners_over_surface=False)
            if not dropped_objects:
                print(f"Dropping of the ShapeNet object {synset_id}/{used_source_id} failed")
                continue

            dropped_object_list.extend(dropped_objects)

        if not dropped_object_list:
            print("No objects were successfully dropped on the surface")
            continue

        # Enable physics for the dropped objects (active) and the surface (passive)
        for dropped_object in dropped_object_list:
            dropped_object.enable_rigidbody(True)
        surface_obj.enable_rigidbody(False)

# Run the physics simulation
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=10, check_object_interval=1)


point_sampler = bproc.sampler.Front3DPointInRoomSampler(room_objs)


bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in room_objs if isinstance(o, bproc.types.MeshObject)])

poses = 0

# Filter some objects from the loaded objects, which are later used in calculating an interesting score
special_objects = [obj.get_cp("category_id") for obj in room_objs if check_name(obj.get_name())]

proximity_checks = {"min": 3.0, "avg": {"min": 3.0, "max": 10}, "no_background": True}
    # Init sampler for sampling locations inside the loaded front3D house

# Sample 10 camera poses
camera_poses = []
for i in range(10000):
    if i % 1000 == 0 and i != 0:
        print('check 1000')
        proximity_checks = {"min": 3.4 - 0.2 * i / 1000, "avg": {"min": 3.0, "max": 10}, "no_background": True}
    # Sample point inside house
    height = np.random.uniform(1.0, 1.5)
    location = point_sampler.sample(height)
    # Sample rotation (fix around X and Y axis)
    rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

    # Check that obstacles are at least 1 meter away from the camera and have an average distance between 2.5 and 3.5
    # meters and make sure that no background is visible, finally make sure the view is interesting enough
    if bproc.camera.scene_coverage_score(cam2world_matrix, special_objects, special_objects_weight=10.0) > 0.25 \
            and bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
        camera_poses.append(cam2world_matrix)
        poses += 1
        print('found ' + str(poses) + ' interesting points')
    if poses > 5:
        break

# Select the best camera pose based on some criteria, here we use the scene coverage score
best_pose = max(camera_poses, key=lambda pose: bproc.camera.scene_coverage_score(pose, special_objects, special_objects_weight=10.0))

bproc.camera.add_camera_pose(best_pose, 0)
bproc.camera.set_resolution(1000, 1000)

bproc.renderer.enable_normals_output()
data = bproc.renderer.render()

# Write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)

        # Save normals as images
for hdf5_file in os.listdir(args.output_dir):
    if hdf5_file.endswith(".hdf5"):
        hdf5_file_path = os.path.join(args.output_dir, hdf5_file)
        image_name = f'frame_{str(hdf5_file)}' + '.png'
        save_normals_as_image(hdf5_file_path, args.output_dir, image_name)