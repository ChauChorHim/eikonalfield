"""
Packages: trimesh, PyMCubes
"""
import cv2
import numpy as np
import json
from os import path
from tqdm import tqdm
import trimesh
import mcubes
import pickle

from run_nerf import config_parser
from load_llff import load_llff_data
from load_blender import load_blender_data


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
    # need to adjust the axis, I don't why :(
    # ret[0, 0] *= -1
    # ret[1, 1] *= -1
    # ret[2, 1] *= -1
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


def unit_test_proejct_origin(args, cam_mat, view_mats, imgs):
    pts = np.array([[0.15, 0.15, 0.15, 1.0],
                    [-0.15, 0.15, 0.15, 1.0],
                    [0.15, -0.15, 0.15, 1.0],
                    [0.15, 0.15, -0.15, 1.0],
                    [-0.15, -0.15, 0.15, 1.0],
                    [-0.15, 0.15, -0.15, 1.0],
                    [0.15, -0.15, -0.15, 1.0],
                    [-0.15, -0.15, -0.15, 1.0]]).reshape(1, 1, 8, 4)
    dictionary = args.basedir + "/unit_test/"

    j = 0
    for view_mat, img in tqdm(zip(view_mats, imgs)):
        uv, z = project_2d(pts, cam_mat, view_mat)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img *= 254
        uv = uv.reshape(-1, 3)
        for i in range(uv.shape[0]):
            img = cv2.circle(img, (int(uv[i, 0]), int(uv[i, 1])), radius=1, thickness=20, color=(0, 0, i * 25))
            # img[(int(uv[i, 1]) - 10):(int(uv[i, 1]) + 10), (int(uv[i, 0]) - 10):(int(uv[i, 0]) + 10)] = 0

        cv2.imwrite(path.join(dictionary, str(j) + '.png'), img)
        j += 1

    quit()


def create_init_bounding_box(trans_mats):
    poses = np.array(trans_mats)[:, :3, 3]
    pose_avg = np.mean(poses, axis=0)
    max_point = np.max(poses, axis=0)
    min_point = np.min(poses, axis=0)
    side = np.max(max_point - min_point) * 0.5
    return pose_avg + np.ones_like(pose_avg) * side * 0.5, pose_avg - np.ones_like(pose_avg) * side * 0.5


def main():
    parser = config_parser()
    parser.add_argument("--grid_size", type=int, default=128,
                        help='the size of ior field, number of the voxels per dim')
    parser.add_argument("--blender_near", type=float, default=0.1,
                        help='near value in blender')
    parser.add_argument("--blender_far", type=float, default=6.,
                        help='far value in blender')
    parser.add_argument("--threshold", type=float, default=0.9,
                        help='threshold for marching cube')
    args = parser.parse_args()

    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        i_test = []
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.

        if args.spherify:
            far = np.minimum(far, 2.0)

    elif args.dataset_type == 'blender':
        args.half_res = False
        images, poses, render_poses, hwf, i_split = \
            load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = args.blender_near
        far = args.blender_far

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    print('NEAR FAR', near, far)


    H, W, focal = hwf
    H, W = int(H), int(W)
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    cam_mat = K

    p_mat = np.concatenate([cam_mat, np.zeros((3, 1))], axis=1)

    # Load or create mask for the images
    masks = images[..., 0] != 0

    num_imgs = len(masks)
    trans_mats = poses

    view_mats = []
    for pose in trans_mats:
        view_mats.append(to_view_matrix(pose))
    view_mats = np.array(view_mats)

    # unit test
    # unit_test_proejct_origin(args, p_mat, view_mats, images)

    # Create voxel grid

    max_point, min_point = create_init_bounding_box(trans_mats)

    X, Y, Z = np.meshgrid(np.linspace(0, 1, args.grid_size),
                          np.linspace(0, 1, args.grid_size),
                          np.linspace(0, 1, args.grid_size))
    x_max, y_max, z_max = max_point
    x_min, y_min, z_min = min_point

    X = X * (x_max - x_min) + x_min
    Y = Y * (y_max - y_min) + y_min
    Z = Z * (z_max - z_min) + z_min
    pts = np.concatenate([np.stack([X, Y, Z], axis=-1),
                          np.ones((args.grid_size, args.grid_size, args.grid_size, 1))],
                         axis=-1)
    voxel_grid = np.ones((args.grid_size, args.grid_size, args.grid_size))

    # Project visual hull
    for view_mat, mask_img in tqdm(zip(view_mats, masks), total=num_imgs):
        uvs, zs = project_2d(pts, p_mat, view_mat)

        us = np.clip(np.round(uvs[..., 0]), 0, mask_img.shape[1] - 1).astype(int)  # width
        vs = np.clip(np.round(uvs[..., 1]), 0, mask_img.shape[0] - 1).astype(int)  # height

        # outside = mask_img[vs.reshape(-1), us.reshape(-1)] == False
        # outside = outside.reshape(args.grid_size, args.grid_size, args.grid_size)
        outside = mask_img[us, vs] == False

        voxel_grid[outside] = 0

    np.save("logs/voxel_grid.npy", voxel_grid)

    # voxel_grid = np.load("logs/voxel_grid.npy")

    # from matplotlib import pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X, Y, Z, alpha=voxel_grid.reshape(-1))
    # plt.show()

    quit()

    # Marching cube
    # with open(path.join(args.datadir, 'mesh.pkl'), 'wb') as f:
    #     pickle.dump({
    #         "data": (count > args.threshold).reshape(-1, 1) * 0.33 + 1.0,  # IoR of glass is 1.5
    #         "extent": 0,
    #         "min_point": min_point,
    #         "max_point": max_point,
    #         "num_voxels": args.grid_size,
    #     }, f)

    # vertices, triangles = mcubes.marching_cubes(count >= args.threshold, 0.5)
    vertices, triangles = mcubes.marching_cubes(voxel_grid, args.threshold)
    print(f'Marching cube: {vertices.shape} vertices, {triangles.shape} triangles')

    vertices /= args.grid_size
    vertices[..., 0] = vertices[..., 0] * (x_max - x_min) + x_min
    vertices[..., 1] = vertices[..., 1] * (y_max - y_min) + y_min
    vertices[..., 2] = vertices[..., 2] * (z_max - z_min) + z_min

    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(path.join(args.basedir, f'mesh_{args.grid_size}_{args.threshold}.obj'))


if __name__ == '__main__':
    main()