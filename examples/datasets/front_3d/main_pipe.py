import blenderproc as bproc
import argparse
import os
import numpy as np
import datetime
import sys
import random
import json
import yaml

sys.path.append('code/BlenderProc/examples/datasets/front_3d')

from main_wo_mat_ani_func import render_scenes_with_animations

"""
blenderproc run code/BlenderProc/examples/datasets/front_3d/main_pipe.py
code/BlenderProc/examples/datasets/front_3d/batch_render.sh 1
"""


def batch_render_scenes_with_animations(config, front_paths, future_folder, output_dir, anime_files,
                                        shapenet_folder, 
                                        shapenet_json,
                                        anime_files_animals,
                                        flow_skip=4):
    count = 0
    for front_path in front_paths:
        count += 1
        print("---------------------------------------------------------")
        print("construct " + str(count) + " scene")
        try:
            render_scenes_with_animations(config, front_path, future_folder, output_dir, anime_files,anime_files_animals, shapenet_folder, shapenet_json=shapenet_json, flow_skip=flow_skip)
        except Exception as e:
            print(f"Failed to render {front_path}: {e}")

def sample_anime(anime_folder, json_file, number=10, animals=False):
    with open(json_file, 'r') as f:
        excluded_paths = set(json.load(f))
    anime_files = []
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

    print('ramdom animes number '+ str(len(anime_files)))
    random.shuffle(anime_files)
    selected = anime_files[:number]
    
    return selected

def sample_scene(scene_folder, number=1):
    front_files = [os.path.join(scene_folder, file) for file in os.listdir(scene_folder) if file.endswith('.json')]
    random.shuffle(front_files)
    selected = front_files[:number]
    print(selected)
    return selected

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config",default="code/BlenderProc/examples/datasets/front_3d/config/deformingfront.yaml", help="Path to the folder containing config")  
    args = parser.parse_args()

    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    config = Config(config)
    #front_files num set to 1，more files rendering is troublesome 
    front_files = sample_scene(config.front_folder)
    anime_files = sample_anime(config.anime_folder, config.dt4_json, config.anime_per_scene)
    anime_files_animals = sample_anime(config.anime_folder_animals, config.dt4_json, config.random_animals, animals=True)

    print(str(len(front_files)) + ' front3D scene choosed')
    print(str(len(anime_files)) + ' anime_files choosed')
    print(str(len(anime_files_animals)) + ' anime_files_animals choosed')

    batch_render_scenes_with_animations(config, front_files, config.future_folder, config.output_dir,
                                        anime_files, 
                                        config.shapenet_folder, 
                                        config.shapenet_json,
                                        anime_files_animals,
                                        flow_skip=config.flow_skip)