import numpy as np
import os
from torch.utils.data import Dataset
from ..build import DATASETS
import warnings
import h5py

warnings.filterwarnings('ignore')

def load_data(data_path, corruption, severity):
    h5_name = os.path.join(data_path, corruption + '_' + str(severity) + '.h5')
    if severity == -1:
        h5_name = os.path.join(data_path, corruption + '.h5')
    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    all_index = np.array([i for i in range(len(data))])
    return data, label, all_index

def normalize(new_pc):
    new_pc[:,0] -= (np.max(new_pc[:,0]) + np.min(new_pc[:,0])) / 2
    new_pc[:,1] -= (np.max(new_pc[:,1]) + np.min(new_pc[:,1])) / 2
    new_pc[:,2] -= (np.max(new_pc[:,2]) + np.min(new_pc[:,2])) / 2
    leng_x, leng_y, leng_z = np.max(new_pc[:,0]) - np.min(new_pc[:,0]), np.max(new_pc[:,1]) - np.min(new_pc[:,1]), np.max(new_pc[:,2]) - np.min(new_pc[:,2])
    if leng_x >= leng_y and leng_x >= leng_z:
        ratio = 2.0 / leng_x
    elif leng_y >= leng_x and leng_y >= leng_z:
        ratio = 2.0 / leng_y
    else:
        ratio = 2.0 / leng_z
    new_pc *= ratio
    return new_pc

@DATASETS.register_module()
class ModelNet40_C6(Dataset):
    classes = ['airplane',
            'bathtub',
            'bed',
            'bench',
            'bookshelf',
            'bottle',
            'bowl',
            'car',
            'chair',
            'cone',
            'cup',
            'curtain',
            'desk',
            'door',
            'dresser',
            'flower_pot',
            'glass_box',
            'guitar',
            'keyboard',
            'lamp',
            'laptop',
            'mantel',
            'monitor',
            'night_stand',
            'person',
            'piano',
            'plant',
            'radio',
            'range_hood',
            'sink',
            'sofa',
            'stairs',
            'stool',
            'table',
            'tent',
            'toilet',
            'tv_stand',
            'vase',
            'wardrobe',
            'xbox']

    def __init__(self,
                num_points=1024,
                data_dir = "./data/modelnet_c",
                split = 'test',
                corruption = 'uniform',
                severity = 1,
                transform=None,
                transformc=None,
                shapley = False,
                cfg = None
                ):
        assert split == 'test'
        self.num_category = 40
        self.split = split
        self.data_path = {
            "test":  data_dir
        }[self.split]
        self.corruption = corruption
        self.severity = severity

        self.data, self.label, self.all_index = load_data(self.data_path, self.corruption, self.severity)
        self.partition =  'test'

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, item):
        pointcloud = self.data[item] # [:self.num_points]
        label = self.label[item]
        index = self.all_index[item]
        # pointcloud = normalize(pointcloud)
        pointcloud[:, 0:3] = self.pc_normalize(pointcloud[:, 0:3])

        return {'x': pointcloud, 'y': label.item(), 'pos': pointcloud, 'index': index}

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_classes(self):
        return np.max(self.label) + 1