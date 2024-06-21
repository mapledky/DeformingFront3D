import json
import os
import warnings
from math import radians
from typing import List
from urllib.request import urlretrieve

import bpy
import mathutils
import numpy as np

from blenderproc.python.material import MaterialLoaderUtility
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping
from blenderproc.python.types.MeshObjectUtility import MeshObject, create_with_empty_mesh
from blenderproc.python.utility.Utility import resolve_path
from blenderproc.python.loader.ObjectLoader import load_obj


def load_front3d_wo_mat(json_path: str, future_model_path: str, label_mapping: LabelIdMapping,
                 ceiling_light_strength: float = 0.8, lamp_light_strength: float = 7.0) -> List[MeshObject]:
    """ Loads the 3D-Front scene specified by the given json file.

    :param json_path: Path to the json file, where the house information is stored.
    :param future_model_path: Path to the models used in the 3D-Front dataset.
    :param label_mapping: A dict which maps the names of the objects to ids.
    :param ceiling_light_strength: Strength of the emission shader used in the ceiling.
    :param lamp_light_strength: Strength of the emission shader used in each lamp.
    :return: The list of loaded mesh objects.
    """
    json_path = resolve_path(json_path)
    future_model_path = resolve_path(future_model_path)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The given path does not exist: {json_path}")
    if not json_path.endswith(".json"):
        raise FileNotFoundError(f"The given path does not point to a .json file: {json_path}")
    if not os.path.exists(future_model_path):
        raise FileNotFoundError(f"The 3D future model path does not exist: {future_model_path}")

    # load data from json file
    with open(json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    if "scene" not in data:
        raise ValueError(f"There is no scene data in this json file: {json_path}")

    created_objects = _Front3DLoader_wo_mat.create_mesh_objects_from_file(data, ceiling_light_strength, label_mapping, json_path)

    all_loaded_furniture = _Front3DLoader_wo_mat.load_furniture_objs(data, future_model_path, lamp_light_strength, label_mapping)

    created_objects += _Front3DLoader_wo_mat.move_and_duplicate_furniture(data, all_loaded_furniture)

    # add an identifier to the obj
    for obj in created_objects:
        obj.set_cp("is_3d_front", True)

    return created_objects


class _Front3DLoader_wo_mat:

    @staticmethod
    def create_mesh_objects_from_file(data: dict, ceiling_light_strength: float,
                                      label_mapping: LabelIdMapping, json_path: str) -> List[MeshObject]:
        """
        This creates for a given data json block all defined meshes and assigns the correct materials.
        This means that the json file contains some mesh, like walls and floors, which have to built up manually.

        It also already adds the lighting for the ceiling

        :param data: json data dir. Must contain "material" and "mesh"
        :param ceiling_light_strength: Strength of the emission shader used in the ceiling.
        :param label_mapping: A dict which maps the names of the objects to ids.
        :param json_path: Path to the json file, where the house information is stored.
        :return: The list of loaded mesh objects.
        """
        created_objects = []
        for mesh_data in data["mesh"]:
            # extract the obj name, which also is used as the category_id name
            used_obj_name = mesh_data["type"].strip()
            if used_obj_name == "":
                used_obj_name = "void"
            if "material" not in mesh_data:
                warnings.warn(f"Material is not defined for {used_obj_name} in this file: {json_path}")
                continue
            # create a new mesh
            obj = create_with_empty_mesh(used_obj_name, used_obj_name + "_mesh")
            created_objects.append(obj)

            # set two custom properties, first that it is a 3D_future object and second the category_id
            obj.set_cp("is_3D_future", True)
            obj.set_cp("category_id", label_mapping.id_from_label(used_obj_name.lower()))

            # extract the vertices from the mesh_data
            vert = [float(ele) for ele in mesh_data["xyz"]]
            # extract the faces from the mesh_data
            faces = mesh_data["faces"]
            # extract the normals from the mesh_data
            normal = [float(ele) for ele in mesh_data["normal"]]

            # map those to the blender coordinate system
            num_vertices = int(len(vert) / 3)
            vertices = np.reshape(np.array(vert), [num_vertices, 3])
            normal = np.reshape(np.array(normal), [num_vertices, 3])
            # flip the first and second value
            vertices[:, 1], vertices[:, 2] = vertices[:, 2], vertices[:, 1].copy()
            normal[:, 1], normal[:, 2] = normal[:, 2], normal[:, 1].copy()
            # reshape back to a long list
            vertices = np.reshape(vertices, [num_vertices * 3])
            normal = np.reshape(normal, [num_vertices * 3])

            # add this new data to the mesh object
            mesh = obj.get_mesh()
            mesh.vertices.add(num_vertices)
            mesh.vertices.foreach_set("co", vertices)
            mesh.vertices.foreach_set("normal", normal)

            # link the faces as vertex indices
            num_vertex_indicies = len(faces)
            mesh.loops.add(num_vertex_indicies)
            mesh.loops.foreach_set("vertex_index", faces)

            # the loops are set based on how the faces are arranged
            num_loops = int(num_vertex_indicies / 3)
            mesh.polygons.add(num_loops)
            # always 3 vertices form one triangle
            loop_start = np.arange(0, num_vertex_indicies, 3)
            # the total size of each triangle is therefore 3
            loop_total = [3] * num_loops
            mesh.polygons.foreach_set("loop_start", loop_start)
            mesh.polygons.foreach_set("loop_total", loop_total)

            # the uv coordinates are reshaped then the face coords are extracted
            uv_mesh_data = [float(ele) for ele in mesh_data["uv"] if ele is not None]
            # bb1737bf-dae6-4215-bccf-fab6f584046b.json includes one mesh which only has no UV mapping
            if uv_mesh_data:
                uv = np.reshape(np.array(uv_mesh_data), [num_vertices, 2])
                used_uvs = uv[faces, :]
                # and again reshaped back to the long list
                used_uvs = np.reshape(used_uvs, [2 * num_vertex_indicies])

                mesh.uv_layers.new(name="new_uv_layer")
                mesh.uv_layers[-1].data.foreach_set("uv", used_uvs)
            else:
                warnings.warn(f"This mesh {obj.get_name()} does not have a specified uv map!")

            # this update converts the upper data into a mesh
            mesh.update()

        return created_objects

    @staticmethod
    def load_furniture_objs(data: dict, future_model_path: str, lamp_light_strength: float,
                            label_mapping: LabelIdMapping) -> List[MeshObject]:
        """
        Load all furniture objects specified in the json file, these objects are stored as "raw_model.obj" in the
        3D_future_model_path. For lamp the lamp_light_strength value can be changed via the config.

        :param data: json data dir. Should contain "furniture"
        :param future_model_path: Path to the models used in the 3D-Front dataset.
        :param lamp_light_strength: Strength of the emission shader used in each lamp.
        :param label_mapping: A dict which maps the names of the objects to ids.
        :return: The list of loaded mesh objects.
        """
        # collect all loaded furniture objects
        all_objs = []
        # for each furniture element
        for ele in data["furniture"]:
            # create the paths based on the "jid"
            folder_path = os.path.join(future_model_path, ele["jid"])
            obj_file = os.path.join(folder_path, "raw_model.obj")
            # if the object exists load it -> a lot of object do not exist
            # we are unsure why this is -> we assume that not all objects have been made public
            if os.path.exists(obj_file) and not "7e101ef3-7722-4af8-90d5-7c562834fabd" in obj_file:
                # load all objects from this .obj file
                objs = load_obj(filepath=obj_file)
                # extract the name, which serves as category id
                used_obj_name = ""
                if "category" in ele:
                    used_obj_name = ele["category"]
                elif "title" in ele:
                    used_obj_name = ele["title"]
                    if "/" in used_obj_name:
                        used_obj_name = used_obj_name.split("/")[0]
                if used_obj_name == "":
                    used_obj_name = "others"
                for obj in objs:
                    obj.set_name(used_obj_name)
                    # add some custom properties
                    obj.set_cp("uid", ele["uid"])
                    # this custom property determines if the object was used before
                    # is needed to only clone the second appearance of this object
                    obj.set_cp("is_used", False)
                    obj.set_cp("is_3D_future", True)
                    obj.set_cp("3D_future_type", "Non-Object")  # is an non object used for the interesting score
                    # set the category id based on the used obj name
                    obj.set_cp("category_id", label_mapping.id_from_label(used_obj_name.lower()))
                all_objs.extend(objs)
            elif "7e101ef3-7722-4af8-90d5-7c562834fabd" in obj_file:
                warnings.warn(f"This file {obj_file} was skipped as it can not be read by blender.")
        return all_objs

    @staticmethod
    def move_and_duplicate_furniture(data: dict, all_loaded_furniture: list) -> List[MeshObject]:
        """
        Move and duplicate the furniture depending on the data in the data json dir.
        After loading each object gets a location based on the data in the json file. Some objects are used more than
        once these are duplicated and then placed.

        :param data: json data dir. Should contain "scene", which should contain "room"
        :param all_loaded_furniture: all objects which have been loaded in load_furniture_objs
        :return: The list of loaded mesh objects.
        """
        # this rotation matrix rotates the given quaternion into the blender coordinate system
        blender_rot_mat = mathutils.Matrix.Rotation(radians(-90), 4, 'X')
        created_objects = []
        # for each room
        for room_id, room in enumerate(data["scene"]["room"]):
            # for each object in that room
            for child in room["children"]:
                if "furniture" in child["instanceid"]:
                    # find the object where the uid matches the child ref id
                    for obj in all_loaded_furniture:
                        if obj.get_cp("uid") == child["ref"]:
                            # if the object was used before, duplicate the object and move that duplicated obj
                            if obj.get_cp("is_used"):
                                new_obj = obj.duplicate()
                            else:
                                # if it is the first time use the object directly
                                new_obj = obj
                            created_objects.append(new_obj)
                            new_obj.set_cp("is_used", True)
                            new_obj.set_cp("room_id", room_id)
                            new_obj.set_cp("3D_future_type", "Object")  # is an object used for the interesting score
                            new_obj.set_cp("coarse_grained_class", new_obj.get_cp("category_id"))
                            # this flips the y and z coordinate to bring it to the blender coordinate system
                            new_obj.set_location(mathutils.Vector(child["pos"]).xzy)
                            new_obj.set_scale(child["scale"])
                            # extract the quaternion and convert it to a rotation matrix
                            rotation_mat = mathutils.Quaternion(child["rot"]).to_euler().to_matrix().to_4x4()
                            # transform it into the blender coordinate system and then to an euler
                            new_obj.set_rotation_euler((blender_rot_mat @ rotation_mat).to_euler())
        return created_objects
