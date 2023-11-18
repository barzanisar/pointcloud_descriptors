import open3d as o3d
import time
import random
import numpy as np
import pickle
import copy


################################################################### ICP with FPFH correspondences ###############
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target_temp])


def point_to_point_icp(source, target, threshold, trans_init):
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation, "\n")
    draw_registration_result(source, target, reg_p2p.transformation)


def point_to_plane_icp(source, target, threshold, trans_init):
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation, "\n")
    draw_registration_result(source, target, reg_p2l.transformation)

def preprocess_point_cloud(pcd, voxel_size, downsample=True):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    if downsample:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = pcd

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, downsample=True, source_pc=None, target_pc=None, source_target_same_demo=False):
    print(":: Load two point clouds and disturb initial pose.")

    if source_pc is not None:
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_pc)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_pc)
    else:
        demo_icp_pcds = o3d.data.DemoICPPointClouds()
        source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
        target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    
    if source_target_same_demo or source is None:
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size, downsample=downsample)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size, downsample=downsample)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, voxel_size, trans_init):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

if __name__ == "__main__":
    path = '/home/barza/DepthContrast/data/waymo/waymo_processed_data_10_short_waymo_dbinfos_train_sampled_1.pkl'
    with open(path, 'rb') as f:
        dbinfos = pickle.load(f)
    
    sample=15
    path = '/home/barza/DepthContrast/data/waymo/' + dbinfos['Cyclist'][sample]['path']
    obj_points = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])


    sample=20
    path = '/home/barza/DepthContrast/data/waymo/' + dbinfos['Vehicle'][sample]['path']
    obj_points1 = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])


    voxel_size = 0.05  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
    voxel_size, downsample=False, source_pc=obj_points[:,:3], target_pc=obj_points1[:,:3], source_target_same_demo=False)
    
    start = time.time()
    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    print("Global registration took %.3f sec.\n" % (time.time() - start))
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    result_icp = refine_registration(source, target, voxel_size, result_ransac.transformation)
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)

    # start = time.time()
    # result_fast = execute_fast_global_registration(source_down, target_down,
    #                                             source_fpfh, target_fpfh,
    #                                             voxel_size)
    # print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    # print(result_fast)
    # draw_registration_result(source_down, target_down, result_fast.transformation)

    b=1


    # pcd_data = o3d.data.DemoICPPointClouds()
    # source = o3d.io.read_point_cloud(pcd_data.paths[0])
    # target = o3d.io.read_point_cloud(pcd_data.paths[1])
    # threshold = 0.02
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                          [-0.139, 0.967, -0.215, 0.7],
    #                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    # draw_registration_result(source, target, trans_init)

    # print("Initial alignment")
    # evaluation = o3d.pipelines.registration.evaluate_registration(
    #     source, target, threshold, trans_init)
    # print(evaluation, "\n")

    # point_to_point_icp(source, target, threshold, trans_init)
    # point_to_plane_icp(source, target, threshold, trans_init)
    


# Compute ISS Keypoints on ArmadilloMesh
# armadillo = o3d.data.ArmadilloMesh()
# mesh = o3d.io.read_triangle_mesh(armadillo.path)
# mesh.compute_vertex_normals()

# pcd = o3d.geometry.PointCloud()
# pcd.points = mesh.vertices

# tic = time.time()
# keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
# toc = 1000 * (time.time() - tic)
# print("ISS Computation took {:.0f} [ms]".format(toc))

# mesh.compute_vertex_normals()
# mesh.paint_uniform_color([0.5, 0.5, 0.5])
# keypoints.paint_uniform_color([1.0, 0.75, 0.0])
# o3d.visualization.draw_geometries([keypoints, mesh])

# # This function is only used to make the keypoints look better on the rendering
# def keypoints_to_spheres(keypoints):
#     spheres = o3d.geometry.TriangleMesh()
#     for keypoint in keypoints.points:
#         sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
#         sphere.translate(keypoint)
#         spheres += sphere
#     spheres.paint_uniform_color([1.0, 0.75, 0.0])
#     return spheres

# # Compute ISS Keypoints on Standford BunnyMesh, changing the default parameters
# bunny = o3d.data.BunnyMesh()
# mesh = o3d.io.read_triangle_mesh(bunny.path)
# mesh.compute_vertex_normals()

# pcd = o3d.geometry.PointCloud()
# pcd.points = mesh.vertices

# tic = time.time()
# keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd,
#                                                         salient_radius=0.005,
#                                                         non_max_radius=0.005,
#                                                         gamma_21=0.5,
#                                                         gamma_32=0.5)
# toc = 1000 * (time.time() - tic)
# print("ISS Computation took {:.0f} [ms]".format(toc))

# mesh.compute_vertex_normals()
# mesh.paint_uniform_color([0.5, 0.5, 0.5])
# o3d.visualization.draw_geometries([keypoints_to_spheres(keypoints), mesh])

