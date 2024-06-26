import blenderproc as bproc
import os
import json
import numpy as np
import random
import argparse
import bpy
from mathutils import Matrix, Vector, Euler

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




def sample_shapenet_obj(loaded_objects, shapenet_json, shapenet_path, sample_number = 3, furniture_max=40):
    if not os.path.exists(shapenet_json) or  not os.path.exists(shapenet_path):
        raise OSError("One of the necessary files or folders does not exist!")
    with open(shapenet_json, 'r') as f:
        shape_data = json.load(f)

    # Extract all synsetId
    synset_ids = []
    for item in shape_data:
        synset_ids.append(item['synsetId'])

    #Collect all ShapeNet model paths
    shapenet_paths = []
    for synset_id in synset_ids:
        synset_dir = os.path.join(shapenet_path, synset_id)
        if os.path.exists(synset_dir):
            used_source_ids = os.listdir(synset_dir)
            for used_source_id in used_source_ids:
                shapenet_paths.append((synset_id, used_source_id))

    # Randomly select 50 model IDs
  

    sample_surface_objects = [obj for obj in loaded_objects if "table" in obj.get_name().lower() 
                              or "stool" in obj.get_name().lower() or "chair" in obj.get_name().lower() 
                              or "desk" in obj.get_name().lower() or "bed" in obj.get_name().lower() 
                              or "sofa" in obj.get_name().lower() or "shelf" in obj.get_name().lower()] 
    
    sample_surface_cabinet = [obj for obj in loaded_objects if "cabinet" in obj.get_name().lower()] 

    random.shuffle(sample_surface_objects)
    sample_surface_objects = sample_surface_objects[:min(20, int(len(sample_surface_objects)))]


    random.shuffle(sample_surface_cabinet)
    sample_surface_cabinet = sample_surface_cabinet[:min(20, int(len(sample_surface_cabinet)))]
    sample_surface_objects.extend(sample_surface_cabinet)

    print("************************* sample surface  furniture" + str(len(sample_surface_objects)))

    if len(sample_surface_objects) < 10:
        return False
    for obj in sample_surface_objects:
        random.shuffle(shapenet_paths)
        selected_shapenet_models = shapenet_paths[:sample_number]
        with bproc.utility.UndoAfterExecution():
            print("_________ sample on  " + str(obj.get_name()))
            choose_models = selected_shapenet_models
            if "stool" in obj.get_name().lower() or "chair" in obj.get_name().lower():
                choose_models = choose_models[:1]
            try:
                # Select the surfaces, where the objects should be sampled on
                surface_obj = bproc.object.slice_faces_with_normals(obj)
                if surface_obj is None:
                    continue
            except:
                print("_______fail to load surface " +str(obj.get_name()) )
                continue

            def sample_pose(obj: bproc.types.MeshObject):
                obj.set_location(bproc.sampler.upper_region(
                    objects_to_sample_on=[surface_obj],
                    min_height=1,
                    max_height=4,
                    use_ray_trace_check=False
                ))
            
            dropped_object_list = []
            for synset_id, used_source_id in choose_models:
                # Load the ShapeNet object, which should be sampled on the surface
                shapenet_obj = bproc.loader.load_shapenet(shapenet_path, used_synset_id=synset_id, used_source_id=used_source_id)
                scale_object_to_size(shapenet_obj, max_size=0.35)
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

            for dropped_object in dropped_object_list:
                dropped_object.enable_rigidbody(True)
            surface_obj.enable_rigidbody(False)

            # Run the physics simulation
            bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1, max_simulation_time=2, check_object_interval=1)
            
    return True

class FurnitureManage:
    def __init__(self, loaded_objects):
        self.loaded_objects = loaded_objects
        self.initial_positions = {}
        self.last_positions = []
        self.target_furniture = [obj for obj in loaded_objects if any(x in obj.get_name().lower() for x in ['chair', 'stool'])]
        print('manage the location of ',str(len(self.target_furniture)))
        for obj in self.target_furniture:
            self.initial_positions[obj] = obj.get_location().copy()
    
    def randommoving(self, camera_location, movingnumber=10,min_distance=1.3, max_distance = 5.0):
        self.clear_moving()
        if not self.target_furniture:
            print("No target furniture to move.")
            return
        cam_loc = Vector(camera_location)
         # Find furniture close to the camera
        nearby_furniture = [obj for obj in self.target_furniture if np.linalg.norm(obj.get_location() - cam_loc) <= max_distance]
        
        # If there are less furniture than movingnumber, we only move the available ones
        moving_furniture = random.sample(nearby_furniture, min(movingnumber, len(nearby_furniture)))

        for obj in moving_furniture:
            # Calculate a random offset
            while True:
                offset = Vector((random.uniform(-1.2, 1.2), random.uniform(-1.2, 1.2), 0))  # only move in x and y directions
                new_loc = obj.get_location() + offset
                distance = np.linalg.norm(new_loc - cam_loc)
                if min_distance <= distance <= max_distance:
                    break
            
            obj.set_location(new_loc)
        print('moving furniture number ', len(moving_furniture))
        bpy.context.view_layer.update()  

    def set_last(self):
        self.last_positions = [np.array(obj.get_location()) for obj in self.loaded_objects]

    def last_location(self):
        if not self.last_positions:
            return
        for obj, last_pos in zip(self.loaded_objects, self.last_positions):
            obj.set_location(Vector(last_pos))
        bpy.context.view_layer.update()
    
    def clear_moving(self):
        """
        Reset all target furniture to their original positions.
        """
        for obj, initial_loc in self.initial_positions.items():
            obj.set_location(initial_loc)
        bpy.context.view_layer.update()

    def invisible_all(self):
        for obj in self.target_furniture:
            obj.blender_obj.hide_render = True
        bpy.context.view_layer.update()

    def visible_all(self):
        for obj in self.target_furniture:
            obj.blender_obj.hide_render = False
        bpy.context.view_layer.update()

class MovingShapenetModels:
    def __init__(self, model_location,camera_location, shapenet_json, shapenet_path, sample_number=3):
        self.model_location = np.array(model_location)
        self.camera_location = np.array(camera_location)
        self.shapenet_path = shapenet_path
        self.sample_number = sample_number
        self.loaded_objects = []
        self.last_locations = []
        

        if not os.path.exists(shapenet_json) or  not os.path.exists(shapenet_path):
                raise OSError("One of the necessary files or folders does not exist!")
        with open(shapenet_json, 'r') as f:
            shape_data = json.load(f)

        # Extract all synsetId
        synset_ids = []
        for item in shape_data:
            synset_ids.append(item['synsetId'])

        #Collect all ShapeNet model paths
        shapenet_paths = []
        for synset_id in synset_ids:
            synset_dir = os.path.join(shapenet_path, synset_id)
            if os.path.exists(synset_dir):
                used_source_ids = os.listdir(synset_dir)
                for used_source_id in used_source_ids:
                    shapenet_paths.append((synset_id, used_source_id))

        # Randomly select 50 model IDs
        random.shuffle(shapenet_paths)
        selected_shapenet_models = shapenet_paths[:sample_number]
        for synset_id, used_source_id in selected_shapenet_models:
            # Load the ShapeNet object, which should be sampled on the surface
            shapenet_obj = bproc.loader.load_shapenet(shapenet_path, used_synset_id=synset_id, used_source_id=used_source_id)
            self.scale_object_to_size(shapenet_obj, max_size=0.35)
            position = self.sample_position_near_model(self.model_location, self.camera_location)
            shapenet_obj.set_location(position)
            self.loaded_objects.append(shapenet_obj)
        
        self.obj_location = []
        for obj in self.loaded_objects:
            obj_location = np.array(obj.get_location())
            self.obj_location.append(obj_location)

    def scale_object_to_size(self, obj, max_size):
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

    def sample_position_near_model(self, model_location, camera_location, radius=3.0, distance_to_camera=1.1): 
        while True:
            random_angle = np.random.uniform(0, 2 * np.pi)
            random_distance = np.random.uniform(0.2, radius)
            x_offset = random_distance * np.cos(random_angle)
            y_offset = random_distance * np.sin(random_angle)

            z_offset = np.random.uniform(0.1, 2.0)

            sample_position = model_location + np.array([x_offset, y_offset, z_offset])
            
            if np.linalg.norm(sample_position - camera_location) > distance_to_camera:
                break
        
        return Vector(sample_position)


    def moving(self,  random_factor=1.2, distance_to_camera = 1.3):
        # Adjust the number of models to move (random number)
        num_models_to_move = random.randint(int(len(self.loaded_objects) / 2), len(self.loaded_objects))

        # Randomly select indices of models to move
        indices_to_move = random.sample(range(len(self.loaded_objects)), num_models_to_move)

        # Adjust the position of selected models
        for i in indices_to_move:
            obj = self.loaded_objects[i]
            init_location = self.obj_location[i]
            
            # Add randomness to the offset
            random_offset = np.array([
                np.random.uniform(-random_factor, random_factor),
                np.random.uniform(-random_factor, random_factor),
                np.random.uniform(-random_factor, random_factor)
            ])
            new_location = init_location + random_offset

            if np.linalg.norm(new_location - self.camera_location) < distance_to_camera:
                continue
            
            # Ensure the model's height is not below ground
            if new_location[2] < 0.1:
                new_location[2] = 0.1
            if new_location[2] > 2.0:
                new_location[2] = 2.0
            
            obj.set_location(new_location)
            min_angle_rad = np.deg2rad(-30)
            max_angle_rad = np.deg2rad(30)
            rotation_angles = np.random.uniform(min_angle_rad, max_angle_rad, size=3)
            obj.set_rotation_euler((0, 0, 0))
            obj.set_rotation_euler(rotation_angles)
            #obj.set_rotation_euler(bproc.sampler.uniformSO3())
            bpy.context.view_layer.update()

    def delete_all_model(self):
        for obj in self.loaded_objects:
            if isinstance(obj, bpy.types.Object):
                bpy.data.objects.remove(obj, do_unlink=True)
            else:
                bpy.data.objects.remove(bpy.data.objects[obj.get_name()], do_unlink=True)
        self.loaded_objects.clear()
        bpy.context.view_layer.update()

    def set_last(self):
        self.last_positions = [np.array(obj.get_location()) for obj in self.loaded_objects]

    def last_location(self):
        if not self.last_positions:
            return
        for obj, last_pos in zip(self.loaded_objects, self.last_positions):
            obj.set_location(Vector(last_pos))
        bpy.context.view_layer.update()

    def invisible_all(self):
        for obj in self.loaded_objects:
            obj.blender_obj.hide_render = True
        bpy.context.view_layer.update()

    def visible_all(self):
        for obj in self.loaded_objects:
            obj.blender_obj.hide_render = False
        bpy.context.view_layer.update()

    


