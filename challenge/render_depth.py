import open3d as o3d
import numpy as np
import json
import os
from PIL import Image
import pdb
import copy
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

img_width, img_height = (3840, 2160)
data_prefix = '/media/ayca/Elements/2023_06_15_17_59_57'
save_data_prefix = '/media/ayca/Elements/2023_06_15_17_59_57/depth_new'
mesh = o3d.io.read_triangle_mesh(os.path.join(data_prefix, 'export_refined.obj'))
mesh.compute_vertex_normals()  # only for easier debugging
near_plane = 0.1 #m
far_plane = 20. #m

# Camera poses in world coordinate system
poses = sorted([f for f in os.listdir(os.path.join(data_prefix, 'poses')) if f.endswith('.json') and f.startswith('frame')])

def move_forward(file):
    file_name = file[6:11]

        # Load camera extrinsics and intrinsics
    with open(os.path.join(data_prefix, 'poses', file[:-4]+'json')) as f:
        json_file = json.load(f)
    intrinsics = np.array(json_file['intrinsics']).reshape([3, 3])
    extrinsics = np.array(json_file['cameraPoseARFrame']).reshape([4, 4])

    # Rotation matrix around x-axis, 180 degrees (pi radians)
    a_x = np.pi
    rot_x = np.eye(3)
    rot_x[1, 1] = np.cos(a_x)
    rot_x[1, 2] = -np.sin(a_x)
    rot_x[2, 1] = np.sin(a_x)
    rot_x[2, 2] = np.cos(a_x)

    # Adapt camera pose for correct depth rendering
    rot = extrinsics[0:3, 0:3] @ rot_x
    t = extrinsics[0:3, 3]
    pose = np.eye(4)
    pose[0:3, 0:3] = rot
    pose[0:3, 3] =  t

    # Convert from world to camera coordinate system
    pose_inv = np.eye(4)
    pose_inv[0:3, 0:3] = rot.T
    pose_inv[0:3, 3] = -rot.T @ t

    cam = o3d.camera.PinholeCameraParameters()
    cam.extrinsic = np.array(pose_inv)
    cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2])

    transformed_mesh = copy.deepcopy(mesh)
    transformed_mesh.compute_vertex_normals() 

    # set renderers
    material = o3d.visualization.rendering.MaterialRecord()        
    renderer = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)

    # don't mess up with pixel values!
    renderer.scene.view.set_post_processing(False)

    # set camera projection
    renderer.setup_camera(cam.intrinsic, cam.extrinsic)
    renderer.scene.camera.set_projection(intrinsics, near_plane, far_plane, img_width, img_height)

    # add scene
    renderer.scene.add_geometry("mesh", transformed_mesh, material)

    # render depth image
    depth_image = renderer.render_to_depth_image(z_in_view_space=True)
    depth_image_array = np.asarray(depth_image)

    # normalize depth image - only for visualization
    normalized_depth_image = depth_image_array.copy()
    normalized_depth_image[normalized_depth_image>far_plane] = 0
    normalized_depth_image = (normalized_depth_image - normalized_depth_image.min()) / (normalized_depth_image.max() - normalized_depth_image.min())
    
    # save depth array and depth image
    np.save(f"{save_data_prefix}/depth_{file_name}.npy", depth_image_array)
    plt.imsave(f"{save_data_prefix}/depth_{file_name}.png", normalized_depth_image)

for file in tqdm(poses):
    move_forward(file)