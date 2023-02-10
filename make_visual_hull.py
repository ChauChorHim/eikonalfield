"""
Packages: trimesh, PyMCubes
"""
import cv2  # for unit test
import numpy as np
import json
import os
from tqdm import tqdm
import trimesh
import mcubes
import pickle
import imageio

from run_nerf import config_parser
from load_llff import load_llff_data
from load_blender import load_blender_data

def unit_test_proejct_origin(args, cam_mat, view_mats, imgs):

    X, Y, Z = np.meshgrid(np.linspace(0, 1, args.grid_size),
                          np.linspace(0, 1, args.grid_size),
                          np.linspace(0, 1, args.grid_size))

    X = X * 0.3 - 0.15
    Y = Y * 0.3 - 0.15
    Z = Z * 0.3 - 0.15

    pts = np.concatenate([np.stack([X, Y, Z], axis=-1),
                          np.ones((args.grid_size, args.grid_size, args.grid_size, 1))],
                         axis=-1)

    dictionary = args.basedir + "/unit_test/"

    j = 0
    for view_mat, img in tqdm(zip(view_mats, imgs)):
        uv, z = project_2d(pts, cam_mat, view_mat)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img *= 254
        uv = uv.reshape(-1, 3)
        for i in range(uv.shape[0]):
            # img = cv2.circle(img, (int(uv[i, 0]), int(uv[i, 1])), radius=1, thickness=20, color=(0, 0, i * 25))
            img = cv2.circle(img, (int(uv[i, 0]), int(uv[i, 1])), radius=1, thickness=20, color=(0, 0, 0))
            # img[(int(uv[i, 1]) - 10):(int(uv[i, 1]) + 10), (int(uv[i, 0]) - 10):(int(uv[i, 0]) + 10)] = 0

        cv2.imwrite(os.path.join(dictionary, str(j) + '.png'), img)
        j += 1

    quit()

def to_view_matrix(mat):
    """
    Args:
      - mat: np.ndarray, [4, 4]
    Returns:
      - ret: np.ndarray, [4, 4]
    """

    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = (-mat[:3, :3].T @ mat[:3, 3:]).reshape(-1)

    return ret

def blender_to_colmap(mat):
    ret = np.eye(4)
    ret[:3, 3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ mat[:3, 3]
    ret[:3, :3] = np.array([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]]) \
                  @ mat[:3, :3] @ \
                  np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]])

    return ret

def project_2d(pts, cam_mat, view_mat):
    """
    Args
      - pts: np.ndarray, [H, W, D, 4]
      - cam_mat: np.ndarray, [4, 4]
      - view_mat: np.ndarray, [4, 4]
    Returns:
      - uv: np.ndarray, [H, W, 2]
      - z: np.ndarray, [H, w]
    """
    pv_mat = cam_mat @ view_mat
    uv = np.einsum('ij,nklj->nkli', pv_mat, pts)
    z = uv[..., 2]
    uv[..., :2] /= uv[..., 2:3]
    return uv, z

def create_init_bounding_box(trans_mats):
    poses = []
    for pose in trans_mats:
        poses.append(blender_to_colmap(pose))

    poses = np.array(poses)[:, :3, 3]
    pose_avg = np.mean(poses, axis=0)
    max_point = np.max(poses, axis=0)
    min_point = np.min(poses, axis=0)
    side = np.max(max_point - min_point)
    return pose_avg + np.ones_like(pose_avg) * side * 0.5, pose_avg - np.ones_like(pose_avg) * side * 0.5


def main():
    parser = config_parser()
    parser.add_argument("--grid_size", type=int, default=128,
                        help='the size of ior field, number of the voxels per dim')
    args = parser.parse_args()

    if args.dataset_type == 'llff':

        # poses: blender/cg coordinates systems
        images, poses, _, _, _ = load_llff_data(args.datadir, args.factor,
                                           recenter=True, bd_factor=.75,
                                           spherify=args.spherify)

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]

        masks_dir = os.path.join(args.datadir, 'masks')

        def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)

        mask_files = [os.path.join(masks_dir, f) for f in sorted(os.listdir(masks_dir)) if
                      f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

        masks_tmp = [imread(f) == 255 for f in mask_files]
        masks = []
        for mask in masks_tmp:
            if mask.ndim == 3:
                mask = mask[..., 0]
            masks.append(mask)

        print('Loaded llff', images.shape, hwf, args.datadir)

    elif args.dataset_type == 'blender':
        args.half_res = False
        images, poses, _, hwf, _ = load_blender_data(args.datadir, args.half_res, args.testskip)

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

        masks = images[..., 0] != 0

        print('Loaded blender', images.shape, hwf, args.datadir)

    H, W, focal = hwf
    H, W = int(H), int(W)
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    cam_mat = K

    p_mat = np.concatenate([cam_mat, np.zeros((3, 1))], axis=1)

    num_imgs = len(masks)

    trans_mats = poses

    view_mats = []
    for pose in trans_mats:
        view_mats.append(to_view_matrix(blender_to_colmap(pose)))
    view_mats = np.array(view_mats)

    # unit test
    # unit_test_proejct_origin(args, p_mat, view_mats, images)

    # Create voxel grid
    max_point, min_point = create_init_bounding_box(trans_mats)
    print("max_point: ", max_point)
    print("min_point: ", min_point)
    x_max, y_max, z_max = max_point
    x_min, y_min, z_min = min_point

    # adjust x_max, y_max, z_max, x_min, y_min, z_min according to the initial values above
    x_max = y_max = z_max = 0.4
    x_min = y_min = z_min = -0.4

    X, Y, Z = np.meshgrid(np.linspace(0, 1, args.grid_size),
                          np.linspace(0, 1, args.grid_size),
                          np.linspace(0, 1, args.grid_size))

    X = X * (x_max - x_min) + x_min
    Y = Y * (y_max - y_min) + y_min
    Z = Z * (z_max - z_min) + z_min
    pts = np.concatenate([np.stack([X, Y, Z], axis=-1),
                          np.ones((args.grid_size, args.grid_size, args.grid_size, 1))],
                         axis=-1)
    voxel_grid = np.ones((args.grid_size, args.grid_size, args.grid_size))

    # Project visual hull
    for view_mat, mask_img, img in tqdm(zip(view_mats, masks, images), total=num_imgs):
        uvs, zs = project_2d(pts, p_mat, view_mat)
        us = np.clip(np.round(uvs[..., 0]), 0, mask_img.shape[1] - 1).astype(int)  # width
        vs = np.clip(np.round(uvs[..., 1]), 0, mask_img.shape[0] - 1).astype(int)  # height

        # outside = mask_img[vs, us] == False
        outside = np.where(mask_img[vs, us] == False)
        voxel_grid[outside] = 0


    import pandas as pd
    from pyntcloud import PyntCloud

    mask = np.where(voxel_grid > 0)    # 0: air, 1: object
    cloud = PyntCloud(pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=np.hstack((pts[mask][..., 0:3].reshape(-1, 3),
                        voxel_grid[mask].reshape(-1, 1),
                        voxel_grid[mask].reshape(-1, 1),
                        voxel_grid[mask].reshape(-1, 1))),
        columns=["x", "y", "z", "red", "green", "blue"]))

    cloud.to_file(os.path.join(args.datadir, f'pcd_{args.grid_size}.ply'))

    with open(os.path.join(args.datadir, 'voxel_grid.pkl'), 'wb') as f:
        pickle.dump({
            "data": voxel_grid,
            "extent": 0,
            "min_point": min_point,
            "max_point": max_point,
            "num_voxels": args.grid_size,
        }, f)

    # Marching cube
    # vertices, triangles = mcubes.marching_cubes(voxel_grid, args.threshold)
    # print(f'Marching cube: {vertices.shape} vertices, {triangles.shape} triangles')
    #
    # vertices /= args.grid_size
    # vertices[..., 0] = vertices[..., 0] * (x_max - x_min) + x_min
    # vertices[..., 1] = vertices[..., 1] * (y_max - y_min) + y_min
    # vertices[..., 2] = vertices[..., 2] * (z_max - z_min) + z_min
    #
    # mesh = trimesh.Trimesh(vertices, triangles)
    # mesh.export(os.path.join(args.datadir, f'mesh_{args.grid_size}_{args.threshold}.obj'))


if __name__ == '__main__':
    main()