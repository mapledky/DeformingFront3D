import blenderproc as bproc
import argparse
import os
import numpy as np
import datetime
import sys
import bpy
import random
import json
from mathutils import Matrix, Vector, Euler
import time
import shutil
import math

sys.path.append('code/BlenderProc/examples/datasets/front_3d/util')

from util.hdftorgb import save_normals_as_image, save_numpy_as_image
from util.anime_renderer import AnimeRenderer , MultiAnimeRenderer, sample_points # 导入 AnimeRenderer 类
from util.sameple_shapenet import sample_shapenet_obj, MovingShapenetModels, FurnitureManage
from util.tools import check_name, compare_depth_maps,compute_back_indice, getcameralocation,compute_overlap, compute_frame_offset_similarity
from util.pointcloud_tools import compute_rt, depth2pointcloud, save_point_cloud_to_numpy_and_pcd, transpoint, save_point_cloud_to_pcd, augment_point_cloud

def save_blend(file_path):
    os.makedirs(file_path, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(file_path))

def render_scenes_with_animations(config, front_path,
                                   future_folder, 
                                   output_dir, 
                                   anime_files, 
                                   anime_files_animals,
                                   shapenet_folder,
                                   shapenet_json="",
                                   flow_skip=4):
    bproc.init()
    scene_ouput_dir = os.path.join(output_dir, "scene",  str(int(time.time() * 1000)))
    output_dir = os.path.join(output_dir, "rawdata")
    scene_render = False
    if not os.path.exists(front_path) or not os.path.exists(future_folder):
        raise Exception("One of the necessary folders does not exist!")

    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    # Set the light bounces
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                     transmission_bounces=200, transparent_max_bounces=200)

    # Load the front 3D objects without textures
    loaded_objects = bproc.loader.load_front3d_wo_mat(
        json_path=front_path,
        future_model_path=future_folder,
        label_mapping=mapping
    )

    if not sample_shapenet_obj(config, loaded_objects,shapenet_json , shapenet_folder):
        return
    if config.ablation >= 3:
        furniture_manager = FurnitureManage(config, loaded_objects)

    point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)
    bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

    poses = 0
    special_objects = [obj.get_cp("category_id") for obj in loaded_objects if check_name(obj.get_name())]
    proximity_checks = {"min": config.camera_to_furniture, "avg": {"min": 2.5, "max": 10}, "no_background": True}

    anime_number = len(anime_files)
    camera_poses = []
    for i in range(12000):
        if i % 1000 == 0 and i != 0:
            print('check 1000')
        height = np.random.uniform(1.2, 1.8)
        location = point_sampler.sample(height)
        if location[2] < 1.2:
            location[2] = 1.2
        rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

        if bproc.camera.scene_coverage_score(cam2world_matrix, special_objects, special_objects_weight=10.0) > 0.8 \
                and bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
            camera_poses.append(cam2world_matrix)
            poses += 1
            print('found ' + str(poses) + ' interesting points')
        if poses >= anime_number or poses >= config.sample_camera_loc:
            break

    sample_camera_number = len(camera_poses)
    if sample_camera_number == 0: return
 
    camera_pose = camera_poses[0]
    bproc.camera.add_camera_pose(camera_pose, 0)

    width = config.render_width
    height = config.render_height

    bproc.camera.set_resolution(width, height)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(config.render_samples)

    distance_from_cam = config.anim_distance_to_camera
    
    print(str(anime_number)+' anime to be render')
    for index, anime_file in enumerate(anime_files):
        #multi = random.choice([0, 1])
        multi = config.multi_anime
        print('*********************new anime ' + str(anime_file) + ' multi choice ' + str(multi))
        #check multi
        
        multi_anime_files = [random.choice(anime_files), random.choice(anime_files_animals)]

        # add camera
        camera_pose = camera_poses[index % sample_camera_number]
        bproc.camera.add_camera_pose(camera_pose, 0)

        cam_location = Matrix(camera_pose).to_translation()
        cam_rot = Matrix(camera_pose).to_euler()

        direction = Vector((0.0, 0.0, -1.0))
        direction.rotate(cam_rot)
        model_location = cam_location + distance_from_cam * direction
        model_location.z = -0.05

        #place shapenet models in front of camera
        if config.ablation >= 3:
            movingshapenet = MovingShapenetModels(config, model_location,cam_location, shapenet_json, shapenet_folder)
        bpy.context.view_layer.update()
        #init_furniture_cam
        cam_loc_tem = cam_location
        cam_loc_tem[2] = 0.0
        if config.ablation >= 3:
            furniture_manager.set_cam_loc(cam_loc_tem)

        print("anime render")
        # Initialize animation renderer
        anime_renderer = AnimeRenderer(config, anime_file)
        num_frames = anime_renderer.offset_data.shape[0]
        anime_renderer.setInitLocation(model_location, cam_location)
        if multi:
            cam_loc_tem = cam_location
            cam_loc_tem[2] = 0.0
            multi_model_location = sample_points(config, model_location, cam_loc_tem)
            multi_anime_renderer = MultiAnimeRenderer(config, multi_anime_files)
            multi_anime_renderer.set_init_location(multi_model_location, cam_location)
            multi_num_frames = multi_anime_renderer.get_frames()

        anim_flow_skip = flow_skip

        pre_frame = None
        pre_pc = None

        pre_back_indices = None
        pre_pc_wo_fore = None

        pre_cam_info = None
        pre_offset = None
        
        pre_frame_num = None
        pre_multi_frame = [0, 0]

        neglect_proportion = 0
        neglect_overlap = 0

        for frame in range(0, num_frames, anim_flow_skip):
            if neglect_proportion >= 5 or neglect_overlap >= 5:break
            frame = frame + random.randint(1, anim_flow_skip)
            multi_frame = [0, 0]
            if multi:
                frame_1 = pre_multi_frame[0] + anim_flow_skip + random.randint(2, anim_flow_skip)
                if frame_1 > multi_num_frames[0] - 1:frame_1 = 0
                frame_2 = pre_multi_frame[1] + anim_flow_skip + random.randint(2, anim_flow_skip)
                if frame_2 > multi_num_frames[1] - 1:frame_2 = 0
                multi_frame[0] =  frame_1
                multi_frame[1] =  frame_2
            
            if(frame >= num_frames):break
            offset = anime_renderer.get_offset(frame)

            if pre_frame != None:
                offset_similarity = compute_frame_offset_similarity(offset, pre_offset)
                if offset_similarity < config.render_offset_thrs or math.isnan(offset_similarity):
                    print('neglect offset______',str(offset_similarity),' !!!!!!')
                    continue
            
            #center_location = model_location + Vector(offset.mean(axis=0)) + Vector((0, 0, 1.15))
            center_location = anime_renderer.get_location(frame)
            cam2world_matrix = getcameralocation(config, center_location, cam_location)
            # #check obstacle
            # obstacle_checks = {"min": 0.5, "avg": {"min": 2.0, "max": 10}, "no_background": True}
            # if not bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, obstacle_checks, bvh_tree):
            #     print('seriously obstacle !!!!!!')
            #     continue
            
            bproc.camera.add_camera_pose(cam2world_matrix, 0)
            cam = bpy.data.objects["Camera"]

            #render background
            anime_renderer.invisible_anim()
            if config.ablation >= 3:
                movingshapenet.invisible_all()
            if config.ablation >= 3:
                furniture_manager.invisible_all()
            if multi:
                multi_anime_renderer.invisible_anim()

            data_back = bproc.renderer.render()

            anime_renderer.visible_anim()
            anime_renderer.vis_frame(frame)
            data_ow_anim = bproc.renderer.render()
            proportion = compare_depth_maps(data_back, data_ow_anim)
            frame_output_dir = output_dir
            if proportion < config.render_proportion_min:
                neglect_proportion += 1
                print("anim out of range !!!!!!!")
                continue
            elif proportion < config.render_proportion_mid:
                frame_output_dir = os.path.join(output_dir, 'sp')
            elif proportion < config.render_proportion_max:
                frame_output_dir = os.path.join(output_dir, 'bp')
            else:
                neglect_proportion += 1
                print("too big proportion !!!!!!!")
                continue
            neglect_proportion = 0
            #render full scene
            if config.ablation >= 3:
                movingshapenet.visible_all()
            if config.ablation >= 3:
                furniture_manager.visible_all()
            if multi:
                multi_anime_renderer.visible_anim()

            if pre_frame != None:
                anime_renderer.vis_frame(pre_frame_num)
                data_wo_anim = bproc.renderer.render()
            anime_renderer.vis_frame(frame)
            # moving shapenet around human and moving furniture around human randomly
            if config.ablation >= 3:
                movingshapenet.moving()
            if config.ablation >= 3:
                furniture_manager.randommoving()
            if multi:
                multi_anime_renderer.vis_frame(multi_frame)

            data = bproc.renderer.render()
            
            cam_info = {
            "angle_x": cam.data.angle_x,
            "matrix_world": np.array(cam2world_matrix).tolist()
            }

            #check similarity
            if pre_frame == None:
                back_indices = compute_back_indice(data, data_back)
                pointcloud, back_indices = depth2pointcloud(config, data.get('depth'), cam_info, indices=back_indices)
                #pointcloud_wo_fore = pointcloud[back_indices]
                if pointcloud.shape[0] < config.render_min_points:
                    print("neglect serious burden or enormous of first frame!!!!")
                    continue
                pre_pc = pointcloud
                pre_back_indices = back_indices
                pre_pc_wo_fore = pointcloud[back_indices]
                pre_frame_num = frame
                pre_offset = offset
                pre_frame = data
                pre_cam_info = cam_info
                pre_multi_frame = multi_frame
                if config.ablation >= 3:
                    movingshapenet.set_last()
                if config.ablation >= 3:
                    furniture_manager.set_last()   
                if config.save_blend:
                    save_blend(os.path.join(scene_ouput_dir, 'src.blend'))
                continue
            
            #conpute back indices
            back_indices = compute_back_indice(data, data_back)
            #convert to point_cloud   
            relative_transform = compute_rt(pre_cam_info, cam_info)
            point_cloud_target, back_indices= depth2pointcloud(config, data.get('depth'), cam_info, indices=back_indices)
            
            point_cloud_target_wo_fore = point_cloud_target[back_indices]

            point_cloud_target_wo_anim, _ = depth2pointcloud(config, data_wo_anim.get('depth'), cam_info)

            if point_cloud_target.shape[0] < config.render_min_points:
                print("neglect serious burden or enormous!!!!")
                if config.ablation >= 3:
                    furniture_manager.last_location()
                if config.ablation >= 3:    
                    movingshapenet.last_location()
                if multi:
                    multi_anime_renderer.vis_frame(pre_multi_frame)
                continue
            
            #check pointcloud overlap
            overlap_ratio_pc1, overlap_ratio_pc2 = compute_overlap(pre_pc_wo_fore, point_cloud_target_wo_fore)
            if overlap_ratio_pc1 < 0.1 and overlap_ratio_pc2 < 0.1:
                print('neglect too small overlap ratio !!!')
                neglect_overlap += 1
                if config.ablation >= 3:
                    furniture_manager.last_location()
                if config.ablation >= 3:
                    movingshapenet.last_location()
                if multi:
                    multi_anime_renderer.vis_frame(pre_multi_frame)
                continue
            elif overlap_ratio_pc1 < 0.25 and overlap_ratio_pc1 < 0.25:
                frame_output_dir = os.path.join(frame_output_dir, 'low')
            else:
                frame_output_dir = os.path.join(frame_output_dir, 'high')

            frame_output_dir = os.path.join(frame_output_dir, str(int(time.time() * 1000)))
            os.makedirs(frame_output_dir, exist_ok=True)
            neglect_overlap = 0
            gt_output_dir = os.path.join(frame_output_dir, 'relative_transform.npy')
            # src_cam_ouput_dir = os.path.join(frame_output_dir, 'src_cam.json')
            # ref_cam_ouput_dir = os.path.join(frame_output_dir, 'ref_cam.json')
            src_pcd_back_indices = os.path.join(frame_output_dir, 'src_back_indices.json')
            ref_pcd_back_indices = os.path.join(frame_output_dir, 'ref_back_indices.json')

            point_cloud_src_save = augment_point_cloud(pre_pc)
            point_cloud_target_save = augment_point_cloud(transpoint(point_cloud_target, relative_transform))
            point_cloud_target_wo_anim_save = augment_point_cloud(transpoint(point_cloud_target_wo_anim, relative_transform))
            np.save(gt_output_dir, relative_transform)

            src_back_indices = {
            "back_indices": np.array(pre_back_indices).tolist()
            }
            ref_back_indices = {
            "back_indices": np.array(back_indices).tolist()
            }

            with open(src_pcd_back_indices, 'w', encoding='utf-8') as f:
                json.dump(src_back_indices, f, ensure_ascii=False, indent=4)

            with open(ref_pcd_back_indices, 'w', encoding='utf-8') as f:
                json.dump(ref_back_indices, f, ensure_ascii=False, indent=4)

            save_numpy_as_image(np.array(pre_frame.get('normals')).squeeze(), frame_output_dir, 'src.png')
            save_numpy_as_image(np.array(data.get('normals')).squeeze(), frame_output_dir, 'ref.png')

            frame_output_dir_src = os.path.join(frame_output_dir, 'src')
            # frame_output_dir_src_wo = os.path.join(frame_output_dir, 'src_wo_fore')

            #save_point_cloud_to_pcd(pre_pc_wo_fore,frame_output_dir_src_wo)
            save_point_cloud_to_pcd(point_cloud_src_save,frame_output_dir_src )

            frame_output_dir_ref = os.path.join(frame_output_dir, 'ref')
            frame_output_dir_ref_wo_anim = os.path.join(frame_output_dir, 'ref_wo_anim')

            save_point_cloud_to_numpy_and_pcd(point_cloud_src_save,frame_output_dir_src )

            save_point_cloud_to_numpy_and_pcd(point_cloud_target_save,frame_output_dir_ref )
            save_point_cloud_to_numpy_and_pcd(point_cloud_target_wo_anim_save,frame_output_dir_ref_wo_anim )

            save_point_cloud_to_pcd(point_cloud_target_save,frame_output_dir_ref )
            #save_point_cloud_to_pcd(transpoint(point_cloud_target_wo_anim, relative_transform),frame_output_dir_ref_wo_anim )

            if not scene_render and config.save_blend:
                save_blend(os.path.join(scene_ouput_dir, 'ref.blend'))
                scene_render = True

            pre_frame_num = frame
            pre_cam_info = cam_info
            pre_frame = data
            pre_offset = offset
            pre_pc = point_cloud_target
            pre_back_indices = back_indices
            pre_pc_wo_fore = point_cloud_target_wo_fore
            if config.ablation >= 3:
                furniture_manager.set_last()
            if config.ablation >= 3:
                movingshapenet.set_last()
            pre_multi_frame = multi_frame

        anime_renderer.invisible_anim()
        if multi:
            multi_anime_renderer.invisible_anim()
        if config.ablation >= 3:
            movingshapenet.delete_all_model()

    bproc.clean_up()
    if not scene_render and config.save_blend:
        if os.path.exists(scene_ouput_dir) and os.path.isdir(scene_ouput_dir):
            shutil.rmtree(scene_ouput_dir)
