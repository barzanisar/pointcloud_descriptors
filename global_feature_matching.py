import pclpy
from pclpy import pcl
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
# from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
import torch
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from pathlib import Path


def isFinite(point):
    if math.isnan(point.normal_x) or math.isnan(point.normal_y) or math.isnan(point.normal_z or math.isnan(point.curvature)):
        return False
    else:
        return True


def extract_feats(obj_pc, method='vfh', reject_infinite_normal_objs=True):
    cloud = pcl.PointCloud.PointXYZ.from_array(obj_pc)

    if method not in ['gasd', 'esf']:
        # print(cloud.size())
        ne = pcl.features.NormalEstimation.PointXYZ_Normal()
        ne.setInputCloud(cloud)

        tree = pcl.search.KdTree.PointXYZ()
        ne.setSearchMethod(tree)

        normals = pcl.PointCloud.Normal()
        ne.setRadiusSearch(0.2)
        ne.compute(normals)
        # print(normals.size())

        cloud_normals = pcl.PointCloud.PointNormal().from_array(
            np.hstack((cloud.xyz, normals.normals, normals.curvature.reshape(-1, 1))))
        
        finiteNormalpts = pcl.PointIndices()

        for i in range(cloud_normals.size()):
            if isFinite(cloud_normals.at(i)):
                finiteNormalpts.indices.append(i)
            else:
                # print(f'cloud_normals[{i}] is not finite\n')
                if reject_infinite_normal_objs:
                    return None
        
        if len(finiteNormalpts.indices) < 5:
            return None
        #Extract finite normals and their points
        cloud_finite = pcl.PointCloud.PointXYZ()
        normals_finite = pcl.PointCloud.PointXYZ()

        extract = pcl.filters.ExtractIndices.PointXYZ()
        extract.setInputCloud(cloud)
        extract.setIndices(finiteNormalpts)
        extract.setNegative(False)
        extract.filter(cloud_finite)

        extract.setInputCloud(pcl.PointCloud.PointXYZ(normals.normals))
        extract.setIndices(finiteNormalpts)
        extract.setNegative(False)
        extract.filter(normals_finite)
        normals_finite = pcl.PointCloud.Normal(normals_finite.xyz)

    if method == 'vfh':
        return compute_vfh_desc(cloud_finite, normals_finite)
    elif method == 'gasd':
        return compute_gasd_desc(cloud)
    elif method == 'grsd':
        return compute_grsd_desc(cloud_finite, normals_finite)
    elif method == 'esf':
        return compute_esf_desc(cloud)
    else:
        NotImplementedError

def compute_esf_desc(cloud):
    # Create the GASD estimation class, and pass the input dataset to it
    gasd = pcl.features.ESFEstimation.PointXYZ_ESFSignature640()

    gasd.setInputCloud(cloud)

    descriptors = pcl.PointCloud.ESFSignature640()
    
    gasd.compute(descriptors)

    # plt.plot(descriptors.histogram[0])
    # plt.show()

    if len(descriptors.histogram):
        return descriptors.histogram[0]
    else:
        return None


def compute_gasd_desc(cloud):
    # Create the GASD estimation class, and pass the input dataset to it
    gasd = pcl.features.GASDEstimation.PointXYZ_GASDSignature512()

    gasd.setInputCloud(cloud)

    descriptors = pcl.PointCloud.GASDSignature512()
    
    gasd.compute(descriptors)

    # plt.plot(descriptors.histogram[0])
    # plt.show()

    if len(descriptors.histogram):
        return descriptors.histogram[0]
    else:
        return None

def compute_grsd_desc(cloud, normals):
    # Create the GASD estimation class, and pass the input dataset to it
    grsd = pcl.features.GRSDEstimation.PointXYZ_Normal_GRSDSignature21()


    grsd.setInputCloud(cloud)
    grsd.setRadiusSearch(0.2)
    tree = pcl.search.KdTree.PointXYZ()
    grsd.setSearchMethod(tree)
    grsd.setInputNormals(normals)

    descriptors = pcl.PointCloud.GRSDSignature21()
    
    grsd.compute(descriptors)

    # plt.plot(descriptors.histogram[0])
    # plt.show()

    if len(descriptors.histogram):
        return descriptors.histogram[0]
    else:
        return None


def compute_vfh_desc(cloud, normals):
    
    vfh = pcl.features.VFHEstimation.PointXYZ_Normal_VFHSignature308()
    vfh.setInputCloud(cloud)
    vfh.setInputNormals(normals)

    tree = pcl.search.KdTree.PointXYZ()
    vfh.setSearchMethod(tree)

    vfhs = pcl.PointCloud.VFHSignature308()

    vfh.compute(vfhs)

    #print(vfhs.size())


    # plt.plot(vfhs.histogram[0])
    # plt.show()
    if len(vfhs.histogram):
        return vfhs.histogram[0]
    else:
        return None
def visualize_tsne(pointwise_gt_cls, tsne, class_names):

    num_classes = len(class_names)
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", num_classes))

    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for idx, label in enumerate(class_names):
        # extract the coordinates of the points of this class only
        current_tx = tx[pointwise_gt_cls == idx]
        current_ty = ty[pointwise_gt_cls == idx]

        # if idx == 0:
        #     continue
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, label=label, marker='x', linewidth=0.7)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.grid()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"TSNE on point features")
    plt.show()
    b=1



def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def draw_feats(X,y, class_names):
    colors = ['blue', 'pink', 'green']
    plt.figure()

    for class_id in range(len(class_names)):
        name = class_names[class_id]
        feats = X[y == class_id] #(nsamples for this class, feat dim)
        plt.plot([0,0], color=colors[class_id], label=name)
        for i in range(feats.shape[0]):
            plt.plot(feats[i], color=colors[class_id])
    plt.legend()
    plt.savefig('gasd.png')
    plt.show()

def check_feat_robustness(points, method, reject_infinite_normal_objs):
    desc0 = extract_feats(points[:,:3], method=method, reject_infinite_normal_objs=reject_infinite_normal_objs)

    angles = np.linspace(-90, 90, 5)
    for i in angles:
        angle= np.deg2rad(i)
        rotated_points = rotate_points_along_z(points[np.newaxis, :, :], np.array([angle]))[0]
        desc = extract_feats(rotated_points[:,:3], method=method, reject_infinite_normal_objs=reject_infinite_normal_objs)
        print(f"Euclidean dist for rot_{i}_deg: {np.linalg.norm(desc0-desc)}")
        plt.plot(desc, label=f'rot_{i}_deg')

    trans = np.linspace(0, 10, 2)
    for i in trans:
        trans_points = points[:,:3]+i
        desc = extract_feats(rotated_points[:,:3], method=method, reject_infinite_normal_objs=reject_infinite_normal_objs)
        print(f"Euclidean dist for trans_{i}_m: {np.linalg.norm(desc0-desc)}")
        plt.plot(desc, label=f'trans_{i}_m')

    plt.legend()
    plt.show()

def extract_all_feats(dbinfos, method, reject_infinite_normal_objs, class_names):
    obj_class_idx = []
    desc_mat = []
    valid_info_idx = [] 

    for class_idx, class_name in enumerate(class_names):
        count_valid = 0
        for i, info in tqdm(enumerate(dbinfos[class_name])):
            path = '/home/barza/OpenPCDet/data/waymo/' + info['path']
            obj_points = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])
            if len(obj_points) > 5:
                # print(f'Computing vfh: {class_names} {i}')
                desc = extract_feats(obj_points[:,:3], method=method, reject_infinite_normal_objs=reject_infinite_normal_objs)
                if desc is not None:
                    desc_mat.append(desc)
                    valid_info_idx.append(i)
                    obj_class_idx.append(class_idx)
                    count_valid+=1
        
        print(f'Found {count_valid} Feats for {class_name}#########')

    desc_mat = np.array(desc_mat)
    valid_info_idx = np.array(valid_info_idx)
    obj_class_idx = np.array(obj_class_idx)

    data_dict = {'labels': obj_class_idx,
                 'feats': desc_mat,
                 'valid_info_idx': valid_info_idx}

    pickle.dump(data_dict, open(f"{method}_feats_data_dict.pkl", "wb"))

    tsne = TSNE(n_components=2).fit_transform(desc_mat) #N obj, 308 feature vector

    pickle.dump(tsne, open(f"{method}_tsne.pkl", "wb"))

# def get_infos(split, frame_sampling_interval):
#     split_txt_file = f'/home/barza/DepthContrast/data/waymo/ImageSets/{split}.txt'
#     seq_list = [x.strip().split('.')[0] for x in open(split_txt_file).readlines()]
#     data_root_path = Path('/home/barza/DepthContrast/data/waymo/waymo_processed_data_v_1_2_0')

#     waymo_infos = []
#     for seq_name in seq_list:
#         seq_info_path = data_root_path / seq_name /  ('%s.pkl' % seq_name)
#         with open(seq_info_path, 'rb') as f:
#             infos = pickle.load(f) # loads 20 infos for one seq pkl i.e. 20 frames if seq pkl was formed by sampling every 10th frame
#             waymo_infos.extend(infos) # each info is one frame

#     if frame_sampling_interval > 1:
#             sampled_waymo_infos = []
#             for k in range(0, len(waymo_infos), frame_sampling_interval):
#                 sampled_waymo_infos.append(waymo_infos[k])
#             waymo_infos = sampled_waymo_infos
#             print('Total sampled samples(frames) for Waymo dataset: %d' % len(waymo_infos))
    
#     return waymo_infos

# def get_lidar(sequence_name, sample_idx):
#         lidar_file = Path('/home/barza/DepthContrast/data/waymo/waymo_processed_data_v_1_2_0') / sequence_name / ('%04d.npy' % sample_idx)
#         point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

#         points_all = point_features[:, 0:4] #points_all: x,y,z,i, skip elongation
#         points_all[:, 3] = np.tanh(points_all[:, 3]) * 255.0  #TODO:
#         return points_all #only get xyzi


if __name__ == "__main__":

    # frame_sampling_interval = 5
    # infos = get_infos('train_short', frame_sampling_interval)

    # X, y=[], []
    # for info in infos:
    #     pc_info = info['point_cloud']
    #     sequence_name = pc_info['lidar_sequence']
    #     sample_idx = pc_info['sample_idx']
    #     frame_id = info['frame_id']

    #     points = get_lidar(sequence_name, sample_idx)

    #     for lbl in np.unique(pt_seg_labels):
    #         if lbl == 0:
    #             continue
    #         obj_pts = points[pt_seg_labels==lbl]

    #path = '/home/barza/DepthContrast/data/waymo/waymo_processed_data_10_short_waymo_dbinfos_train_sampled_1.pkl'
    path = '/home/barza/OpenPCDet/data/waymo/waymo_processed_data_v_1_2_0_waymo_dbinfos_train_sampled_100.pkl'
    with open(path, 'rb') as f:
        dbinfos = pickle.load(f)
    class_names=['Vehicle', 'Pedestrian', 'Cyclist']
    METHOD='vfh' #vfh, esf, gasd
    reject_infinite_normal_objs=False
    random_seed = 42 

    # ########### check feat robustness
    # sample=15
    # class_name = 'Pedestrian'
    # path = '/home/barza/DepthContrast/data/waymo/' + dbinfos[class_name][sample]['path']
    # points = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])
    # check_feat_robustness(points, METHOD, reject_infinite_normal_objs)

    # ########### compare feats 
    # sample=15
    # obj_class1 = 'Pedestrian'
    # path = '/home/barza/DepthContrast/data/waymo/' + dbinfos[obj_class1][sample]['path']
    # points = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])
    # obj_points1 = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])
    # desc1 = extract_feats(obj_points1[:,:3], method=METHOD, reject_infinite_normal_objs=reject_infinite_normal_objs)


    # sample=20
    # obj_class2 = 'Pedestrian'
    # path = '/home/barza/DepthContrast/data/waymo/' + dbinfos[obj_class2][sample]['path']
    # obj_points2 = np.fromfile(str(path), dtype=np.float32).reshape([-1, 5])
    # desc2 = extract_feats(obj_points2[:,:3], method=METHOD, reject_infinite_normal_objs=reject_infinite_normal_objs)

    # plt.plot(desc1, label=obj_class1+'1')
    # plt.plot(desc2, label=obj_class2+'2')
    # plt.title(f'Euclidean Distance: {np.linalg.norm(desc1-desc2)}')
    # plt.legend()
    # plt.show()
    
    ################## Extract feats
    extract_all_feats(dbinfos, METHOD, reject_infinite_normal_objs, class_names)
    
    
    # ############ load and visualize tsne
    # data_dict = pickle.load(open(f"{METHOD}_feats_data_dict.pkl", "rb"))
    # tsne = pickle.load(open("tsne.pkl", "rb"))
    # visualize_tsne(data_dict['labels'], tsne, class_names)

    # ############ visualize features
    # data_dict = pickle.load(open(f"{METHOD}_feats_data_dict.pkl", "rb"))
    # X = data_dict['feats'] #(nsamples, nfeats)
    # y = data_dict['labels'] #(nsamples,)
    # draw_feats(X,y, class_names)

    # ############ classify features
    data_dict = pickle.load(open(f"{METHOD}_feats_data_dict.pkl", "rb"))
    X = data_dict['feats'] #(nsamples, nfeats)
    y = data_dict['labels'] #(nsamples,)

    # X_train, X_test, y_train, y_test = [], [], [], []
    # for class_id in range(len(class_names)):
    #     name = class_names[class_id]
    #     X_cls = X[y == class_id]
    #     y_cls = y[y == class_id]
    #     X_tr_cls, X_tst_cls, y_tr_cls, y_tst_cls = train_test_split(X_cls, y_cls, test_size=0.33, random_state=42)
    #     X_train.append(X_tr_cls)
    #     X_test.append(X_tst_cls)
    #     y_train.append(y_tr_cls)
    #     y_test.append(y_tst_cls)

    # X_train = np.concatenate(X_train)
    # X_test = np.concatenate(X_test)
    # y_train = np.concatenate(y_train)
    # y_test = np.concatenate(y_test)
    classifier = svm.SVC(kernel='poly', random_state=random_seed)
    #classifier = MLPClassifier(hidden_layer_sizes=(128, 128, 3), early_stopping=True, random_state=random_seed)
    clf = make_pipeline(classifier) #(StandardScaler(), classifier)
    #clf.fit(X_train, y_train) #'poly', 'rbf' cross_val_score(clf, X, y, cv=5)
    #score = clf.score(X_test, y_test)

    cv_scores = cross_val_score(clf, X, y, cv=5)
    mean_score = np.mean(cv_scores)
    print(f'{METHOD} val score: {cv_scores}\t mean score: {mean_score}')
