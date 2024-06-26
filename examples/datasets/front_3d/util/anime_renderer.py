import numpy as np
from mathutils import Vector
from mathutils import Matrix, Vector, Euler
import bmesh
import bpy
import mathutils

def anime_read(filename):
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    if len(offset_data) != (nf - 1) * nv * 3:
        raise Exception("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data

class AnimeRenderer:
    def __init__(self, anime_file):
        _, _, _, vert_data, face_data, offset_data = anime_read(anime_file)
        offset_data = np.concatenate([np.zeros((1, offset_data.shape[1], offset_data.shape[2])), offset_data], axis=0)
        vertices = vert_data.tolist()
        edges = []
        faces = face_data.tolist()
        mesh_data = bpy.data.meshes.new('mesh_data')
        mesh_data.from_pydata(vertices, edges, faces)
        mesh_data.update()
        the_mesh = bpy.data.objects.new('the_mesh', mesh_data)
        the_mesh.data.vertex_colors.new()  # init color
        bpy.context.collection.objects.link(the_mesh)
        self.the_mesh = the_mesh
        self.offset_data = offset_data
        self.vert_data = vert_data

        self.init_location = None
        self.cam_location = None
        self.random_offsets = None
        self.update_origin()
    
    def generate_random_offsets(self, min_distance =1.4):
        num_frames = self.offset_data.shape[0]
        random_offsets = np.zeros((num_frames, 3))
        for i in range(num_frames):
            attempts = 0
            valid_offset_found = False
            while attempts < 100:
                random_location_offset = np.random.uniform(-1.1, 1.1, (2))
                new_location = self.init_location[:2] + random_location_offset
                distance = np.linalg.norm(new_location - self.cam_location[:2])
                if distance > min_distance:
                    valid_offset_found = True
                    break
                attempts += 1
            if not valid_offset_found:
                random_location_offset = np.random.uniform(-0.8, 0.8, (2))

            random_rotation_offset = np.random.uniform(-np.pi, np.pi, (1))
            random_offsets[i] = np.concatenate((random_location_offset, random_rotation_offset), axis=0)
        print('generating offset ', len(random_offsets))
        return random_offsets

    def get_location(self, fid=None):
        if fid == None:
            return self.the_mesh.location
        random_offset = self.random_offsets[fid]
        ran_loc = [random_offset[0], random_offset[1], 0]
        new_location = self.init_location + Vector(ran_loc)
        if new_location[2] > 1.8:
            new_location[2] = 1.8
        return new_location

    def update_origin(self):
        # Update the origin to the geometry's center
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = self.the_mesh
        self.the_mesh.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
        bpy.context.view_layer.update()

    def setInitLocation(self, location, cam_location):
        self.update_origin()
        location[2] = np.array(self.the_mesh.location)[2]
        self.the_mesh.location = location
        self.init_location = location
        self.cam_location = cam_location
        self.random_offsets = self.generate_random_offsets()

    def get_offset(self, fid):
        src_offset = self.offset_data[fid]
        return src_offset

    def vis_frame(self, fid):
        # 更新动画模型的位置
        src_offset = self.offset_data[fid]
        bm = bmesh.new()
        bm.from_mesh(self.the_mesh.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        for i in range(len(bm.verts)):
            # 调整动画模型每一帧的移动距离
            new_co = Vector(self.vert_data[i] + src_offset[i] )
            bm.verts[i].co = new_co
        bm.to_mesh(self.the_mesh.data)
        bm.free()
        random_offset = self.random_offsets[fid]
        ran_loc = [random_offset[0], random_offset[1], 0]
        rotation_matrix = Matrix.Rotation(random_offset[2], 4, 'Z')
        new_location = self.init_location + Vector(ran_loc)
        if new_location[2] > 1.8:
            new_location[2] = 1.8
        self.update_origin()
        self.the_mesh.location = new_location
        self.the_mesh.rotation_euler = Euler((0, 0, 0), 'XYZ')
        self.the_mesh.rotation_euler.rotate(rotation_matrix)
        return src_offset


    def invisible_anim(self):
        self.the_mesh.hide_render = True

    def visible_anim(self):
        self.the_mesh.hide_render = False


    def check_collision(self, bvh_tree):
        # Create a new bmesh from the_mesh data
        bm = bmesh.new()
        bm.from_mesh(self.the_mesh.data)
        bm.faces.ensure_lookup_table()  # Ensure the lookup table is up to date
        # Create a BVH tree from the bmesh of the animation mesh
        bvh_tree_anime = mathutils.bvhtree.BVHTree.FromBMesh(bm)

        # Find overlapping indices between the two BVH trees
        overlap = bvh_tree_anime.overlap(bvh_tree)

        # Print the overlap ratio
        overlap_ratio = float(len(overlap)) / float(len(bm.verts))
        print('Overlap Ratio:', overlap_ratio)
        bm.free()
        return overlap_ratio
    
    @staticmethod
    def is_out_range(center_location, model_location, max_distance):
        """Check if center_location is within max_distance from model_location."""
        distance = (center_location - model_location).length
        return distance > max_distance
    

    @staticmethod
    def compute_image_difference(image1, image2):
        # 将图片转换为 numpy 数组
        array1 = np.array(image1)
        array2 = np.array(image2)
        
        # 计算差异值
        diff = np.abs(array1 - array2)
        
        # 取差异值的平均值作为整体差异度
        difference = np.mean(diff)
        
        return difference



class MultiAnimeRenderer:
    def __init__(self, anime_files):
        self.renderers = [AnimeRenderer(anime_file) for anime_file in anime_files]
    
    def set_init_location(self, locations, cam_location):
        for i, renderer in enumerate(self.renderers):
            renderer.setInitLocation(locations[i], cam_location)
        
    def get_frames(self):
        frames = []
        for renderer in self.renderers:
            frames.append(renderer.offset_data.shape[0])
        return frames

    def vis_frame(self, fid):
        for i,renderer in enumerate(self.renderers):
            renderer.vis_frame(fid[i])

    def invisible_anim(self):
        for renderer in self.renderers:
            renderer.invisible_anim()

    def visible_anim(self):
        for renderer in self.renderers:
            renderer.visible_anim()
import numpy as np

def sample_points(center, camera_location, radius=2.5, min_distance=0.5, num_points=2, distance_to_camera=2.8):
    points = []
    attempts = 0
    max_attempts = 1000  # 防止无限循环

    while len(points) < num_points and attempts < max_attempts:
        # 随机采样一个点，固定z坐标
        random_point = center[:2] + np.random.uniform(-radius, radius, 2)
        random_point = np.append(random_point, center[2])
        
        # 检查点是否在圆柱体内
        if np.linalg.norm(random_point[:2] - center[:2]) <= radius:
            # 检查新点与已有点之间的距离和与camera_location的距离
            if all(np.linalg.norm(random_point[:2] - p[:2]) >= min_distance for p in points) and np.linalg.norm(random_point - camera_location) > distance_to_camera:
                points.append(random_point)
        
        attempts += 1

    if len(points) < num_points:
        raise ValueError("无法在指定的尝试次数内找到满足条件的点")

    return points









