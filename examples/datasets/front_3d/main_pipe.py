import blenderproc as bproc
import argparse
import os
import numpy as np
import datetime
import sys
import random
import json

sys.path.append('code/BlenderProc/examples/datasets/front_3d')

from main_wo_mat_ani_func import render_scenes_with_animations

"""
blenderproc run code/BlenderProc/examples/datasets/front_3d/main_pipe.py
code/BlenderProc/examples/datasets/front_3d/batch_render.sh 1
"""


def batch_render_scenes_with_animations(front_paths, future_folder, output_dir, anime_files,
                                        shapenet_folder, 
                                        shapenet_json,
                                        anime_files_animals,
                                        shapenet_number=5,
                                        flow_skip=4):
    count = 0
    for front_path in front_paths:
        count += 1
        print("---------------------------------------------------------")
        print("construct " + str(count) + " scene")
        try:
            render_scenes_with_animations(front_path, future_folder, output_dir, anime_files,anime_files_animals, shapenet_folder, shapenet_json=shapenet_json, shapenet_number=shapenet_number, flow_skip=flow_skip)
        except Exception as e:
            print(f"Failed to render {front_path}: {e}")

def sample_anime(anime_folder, json_file, number=10, animals=False):
    # 读取JSON文件中需要排除的动画路径
    with open(json_file, 'r') as f:
        excluded_paths = set(json.load(f))
    anime_files = []
    # 遍历文件夹，查找所有.anime文件
    for root, dirs, files in os.walk(anime_folder):
        for file in files:
            if file.endswith(".anime"):
                if animals and not ('bunny' in str(file).lower() or 'canie' in str(file).lower() or \
                    'chicken' in str(file).lower() or 'doggie' in str(file).lower() or \
                        'fox' in str(file).lower() or 'huskydog' in str(file).lower() or 'rabbit' in str(file).lower()):
                    continue
                full_path = os.path.join(root, file)
                if str(root) not in excluded_paths:
                    anime_files.append(full_path)
    
    # 随机打乱并选择指定数量的动画文件
    print('ramdom animes number '+ str(len(anime_files)))
    #177
    random.shuffle(anime_files)
    selected = anime_files[:number]
    
    return selected

def sample_scene(scene_folder, number=1):
    front_files = [os.path.join(scene_folder, file) for file in os.listdir(scene_folder) if file.endswith('.json')]
    random.shuffle(front_files)
    selected = front_files[:number]
    print(selected)
    return selected

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", default="dataset/3D-FRONT", help="Path to the folder containing 3D front files")
    parser.add_argument("future_folder", default="dataset/3D-FUTURE-model", help="Path to the 3D Future Model folder.")
    parser.add_argument("output_dir",default="code/BlenderProc/examples/datasets/front_3d/output", help="Path to where the data should be saved")
    parser.add_argument("anime_folder",default="dataset/DeformingThings4D/humanoids", help="Path to the folder containing animation files")
    parser.add_argument("shapenet_folder",default="dataset/", help="Path to the folder containing shapenet files")  
    parser.add_argument("shapenet_json",default="dataset/", help="Path to the folder containing shapenet json")  
    parser.add_argument("dt4_json",default="dataset/", help="Path to the folder containing dt4 json")  
    parser.add_argument("anime_folder_animals",default="dataset/DeformingThings4D/animals", help="Path to the folder containing deforming animals") 
    parser.add_argument("--flow_skip", type=int, default=4, help="Frame skip interval for animation")
    args = parser.parse_args()

    #front_files num set to 1，more files rendering is troublesome 
    front_files = sample_scene(args.front_folder, 1)
    anime_files = sample_anime(args.anime_folder, args.dt4_json, 30)
    anime_files_animals = sample_anime(args.anime_folder_animals, args.dt4_json, 48, animals=True)

    print(str(len(front_files)) + ' front3D scene choosed')
    print(str(len(anime_files)) + ' anime_files choosed')
    print(str(len(anime_files_animals)) + ' anime_files_animals choosed')

    batch_render_scenes_with_animations(front_files, args.future_folder, args.output_dir,
                                        anime_files, 
                                        args.shapenet_folder, 
                                        args.shapenet_json,
                                        anime_files_animals,
                                        shapenet_number=7,
                                        flow_skip=args.flow_skip)