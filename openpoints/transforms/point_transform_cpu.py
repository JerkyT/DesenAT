import numpy as np
import torch
from .point_transformer_gpu import DataTransforms
from scipy.linalg import expm, norm


@DataTransforms.register_module()
class PointsToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if not torch.is_tensor(data[key]):
                if str(data[key].dtype) == 'float64':
                    data[key] = data[key].astype(np.float32)
                data[key] = torch.from_numpy(np.array(data[key]))
        return data


@DataTransforms.register_module()
class RandomRotate(object):
    def __init__(self, angle=[0, 0, 1], **kwargs):
        self.angle = angle

    def __call__(self, data):
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        data['pos'] = np.dot(data['pos'], np.transpose(R))
        return data


@DataTransforms.register_module()
class RandomRotateZ(object):
    def __init__(self, angle=1.0, rotate_dim=2, random_rotate=True, **kwargs):
        self.angle = angle * np.pi
        self.random_rotate = random_rotate
        axis = np.zeros(3)
        axis[rotate_dim] = 1
        self.axis = axis
        self.rotate_dim = rotate_dim

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        if self.random_rotate:
            rotate_angle = np.random.uniform(-self.angle, self.angle)
        else:
            rotate_angle = self.angle
        R = self.M(self.axis, rotate_angle)
        data['pos'] = np.dot(data['pos'], R)  # anti clockwise
        return data

    def __repr__(self):
        return 'RandomRotate(rotate_angle: {}, along_z: {})'.format(self.rotate_angle, self.along_z)


@DataTransforms.register_module()
class RandomScale(object):
    def __init__(self, scale=[0.8, 1.2],
                 scale_anisotropic=False,
                 scale_xyz=[True, True, True],
                 mirror=[-1, -1, -1],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale = scale
        self.scale_xyz = scale_xyz
        self.anisotropic = scale_anisotropic
        self.mirror = np.array(mirror)
        self.use_mirroring = np.sum(self.mirror > 0) != 0

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        if len(scale) == 1:
            scale = scale.repeat(3)
        if self.use_mirroring:
            mirror = (np.random.rand(3) > self.mirror).astype(np.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        data['pos'] *= scale
        return data

    def __repr__(self):
        return 'RandomScale(scale_low: {}, scale_high: {})'.format(self.scale_min, self.scale_max)


@DataTransforms.register_module()
class RandomScaleAndJitter(object):
    def __init__(self,
                 scale=[0.8, 1.2],
                 scale_xyz=[True, True, True],  # ratio for xyz dimenions
                 scale_anisotropic=False,  # scaling in different ratios for x, y, z
                 jitter_sigma=0.01, jitter_clip=0.05,
                 mirror=[-1, -1, -1],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale = scale
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.scale_xyz = scale_xyz
        self.noise_sigma = jitter_sigma
        self.noise_clip = jitter_clip
        self.anisotropic = scale_anisotropic
        self.mirror = np.array(mirror)
        self.use_mirroring = np.sum(self.mirror > 0) != 0

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)

        if len(scale) == 1:
            scale = scale.repeat(3)
        if self.use_mirroring:
            mirror = (np.random.rand(3) > self.mirror).astype(np.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        jitter = np.clip(self.noise_sigma * np.random.randn(data['pos'].shape[0], 3), -self.noise_clip, self.noise_clip)
        data['pos'] = data['pos'] * scale + jitter
        return data


@DataTransforms.register_module()
class RandomShift(object):
    def __init__(self, shift=[0.2, 0.2, 0], **kwargs):
        self.shift = shift

    def __call__(self, data):
        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        data['pos'] += shift
        return data

    def __repr__(self):
        return 'RandomShift(shift_range: {})'.format(self.shift_range)


@DataTransforms.register_module()
class RandomScaleAndTranslate(object):
    def __init__(self,
                 scale=[0.9, 1.1],
                 shift=[0.2, 0.2, 0],
                 scale_xyz=[1, 1, 1],
                 **kwargs):
        self.scale = scale
        self.scale_xyz = scale_xyz
        self.shift = shift

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        scale *= self.scale_xyz

        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        data['pos'] = np.add(np.multiply(data['pos'], scale), shift)

        return data


@DataTransforms.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        if np.random.rand() < self.p:
            data['pos'][:, 0] = -data['pos'][:, 0]
        if np.random.rand() < self.p:
            data['pos'][:, 1] = -data['pos'][:, 1]
        return data


@DataTransforms.register_module()
class RandomJitter(object):
    def __init__(self, jitter_sigma=0.01, jitter_clip=0.05, **kwargs):
        self.noise_sigma = jitter_sigma
        self.noise_clip = jitter_clip

    def __call__(self, data):
        jitter = np.clip(self.noise_sigma * np.random.randn(data['pos'].shape[0], 3), -self.noise_clip, self.noise_clip)
        data['pos'] += jitter
        return data


@DataTransforms.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None, **kwargs):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data):
        if np.random.rand() < self.p:
            lo = np.min(data['x'][:, :3], 0, keepdims=True)
            hi = np.max(data['x'][:, :3], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data['x'][:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            data['x'][:, :3] = (1 - blend_factor) * data['x'][:, :3] + blend_factor * contrast_feat
            """vis
            from openpoints.dataset import vis_points
            vis_points(data['pos'], data['x']/255.)
            """
        return data


@DataTransforms.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05, **kwargs):
        self.p = p
        self.ratio = ratio

    def __call__(self, data):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data['x'][:, :3] = np.clip(tr + data['x'][:, :3], 0, 255)
        return data


@DataTransforms.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005, **kwargs):
        self.p = p
        self.std = std

    def __call__(self, data):
        if np.random.rand() < self.p:
            noise = np.random.randn(data['x'].shape[0], 3)
            noise *= self.std * 255
            data['x'][:, :3] = np.clip(noise + data['x'][:, :3], 0, 255)
        return data


@DataTransforms.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]

        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2, **kwargs):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(data['x'][:, :3])
        hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        data['x'][:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return data


@DataTransforms.register_module()
class RandomDropFeature(object):
    def __init__(self, feature_drop=0.2,
                 drop_dim=[0, 3],
                 **kwargs):
        self.p = feature_drop
        self.dim = drop_dim

    def __call__(self, data):
        if np.random.rand() < self.p:
            data['x'][:, self.dim[0]:self.dim[-1]] = 0
        return data


@DataTransforms.register_module()
class NumpyChromaticNormalize(object):
    def __init__(self,
                 color_mean=None,
                 color_std=None,
                 **kwargs):

        self.color_mean = np.array(color_mean).astype(np.float32) if color_mean is not None else None
        self.color_std = np.array(color_std).astype(np.float32) if color_std is not None else None

    def __call__(self, data):
        if data['x'][:, :3].max() > 1:
            data['x'][:, :3] /= 255.
        if self.color_mean is not None:
            data['x'][:, :3] = (data['x'][:, :3] - self.color_mean) / self.color_std
        return data

@DataTransforms.register_module()
class Supervoxel(object):
    def __init__(self, **kwargs):
        self.npoints = 1024
        pass

    def __call__(self, data):
        weights = data['shapely']
        weights -= min(weights)
        idx = np.argsort(weights)[::-1]

        p = 0
        s = np.sum(weights)
        for i in range(len(idx) - 1, -1, -1):
            p += weights[idx[i]]
            if p > 0.005 * s:
                idx = idx[i::]
                break

        # 将距离参考点小于0.2的点标记为True
        distances = np.linalg.norm(data['pos'] - data['pos'][idx][:, np.newaxis], axis=2)
        radius = np.mean(np.sort(distances, 1)[:, 1:10])
        mask = np.any(distances <= radius, axis=0) # 距离小的点

        data['pos'] = data['pos'][~mask]

        npoints = self.npoints
        point  = data['pos']
        N, D = point.shape
        xyz = point[:,:3]
        centroids = np.zeros((npoints,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoints):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        point = point[centroids.astype(np.int32)]
        data['pos'] = point

        return data

#################################################################
##  Generate Various Common Corruptions ###
import numpy as np
from . import distortion
from .util import *

# np.random.seed(2021)
ORIG_NUM = 1024
import torch
import random
### Transformation ###
'''
Rotate the point cloud
'''

def rotation(pointcloud,severity = 1):
    B, N, C = pointcloud.shape
    c = [2.5, 5, 7.5, 10, 15][severity-1]
    theta = np.random.uniform(c-2.5,c+2.5, B) * np.random.choice([-1,1], B) * np.pi / 180.
    gamma = np.random.uniform(c-2.5,c+2.5, B) * np.random.choice([-1,1], B) * np.pi / 180.
    beta = np.random.uniform(c-2.5,c+2.5, B) * np.random.choice([-1,1], B) * np.pi / 180.

    matrix_1, matrix_2, matrix_3 = np.zeros((B,3,3)),np.zeros((B,3,3)),np.zeros((B,3,3))
    matrix_1[:,0,0], matrix_1[:,1,1], matrix_1[:,1,2], matrix_1[:,2,1], matrix_1[:,2,2], = \
                                                            1, np.cos(theta), -np.sin(theta), np.sin(theta),np.cos(theta)
    matrix_2[:,0,0], matrix_2[:,0,2], matrix_2[:,1,1], matrix_2[:,2,0], matrix_2[:,2,2], = \
                                                            np.cos(gamma), np.sin(gamma), 1, -np.sin(gamma), np.cos(gamma)
    matrix_3[:,0,0], matrix_3[:,0,1], matrix_3[:,1,0], matrix_3[:,1,1], matrix_3[:,2,2], = \
                                                            np.cos(beta), -np.sin(beta), np.sin(beta), np.cos(beta), 1

    # matrix_1 = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    # matrix_2 = np.array([[np.cos(gamma),0,np.sin(gamma)],[0,1,0],[-np.sin(gamma),0,np.cos(gamma)]])
    # matrix_3 = np.array([[np.cos(beta),-np.sin(beta),0],[np.sin(beta),np.cos(beta),0],[0,0,1]])

    new_pc = np.matmul(pointcloud, matrix_1)
    new_pc = np.matmul(new_pc, matrix_2)
    new_pc = np.matmul(new_pc, matrix_3).astype('float32')

    return new_pc # normalize(new_pc)

'''
Shear the point cloud
'''
def shear(pointcloud,severity = 1):
    B, N, C = pointcloud.shape
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
    # a = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    b = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    d = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    e = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    f = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    # g = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)

    matrix = np.zeros((B, 3, 3))
    matrix[:,0,0], matrix[:,0,1], matrix[:,0,2] = 1, 0, b
    matrix[:,1,0], matrix[:,1,1], matrix[:,1,2] = d, 1, e
    matrix[:,2,0], matrix[:,2,1], matrix[:,2,2] = f, 0, 1

    new_pc = np.matmul(pointcloud, matrix).astype('float32')
    return new_pc #normalize(new_pc)

'''
Scale the point cloud
'''

def scale(pointcloud, severity=1):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    pointcloud = torch.from_numpy(pointcloud.transpose(0,2,1))
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            B x C x N array, original batch of point clouds
        Return:
            B x C x N array, scaled batch of point clouds
    """
    scales = (torch.rand(pointcloud.shape[0], 3, 1) * 2. - 1.) * c + 1.
    pointcloud *= scales
    pointcloud = pointcloud.numpy().transpose(0,2,1)
    return pointcloud
### Noise ###
'''
Add Uniform noise to point cloud 
'''
def uniform_noise(pointcloud,severity = 1):
    #TODO
    B, N, C = pointcloud.shape
    c = [0.01, 0.02, 0.03, 0.04, 0.05][severity-1]
    jitter = np.random.uniform(-c,c,(B, N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc # normalize(new_pc)

'''
Add Gaussian noise to point cloud 
'''
def gaussian_noise(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    c = [0.01, 0.015, 0.02, 0.025, 0.03][severity-1]
    jitter = np.random.normal(size=(B, N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    # new_pc = np.clip(new_pc,-1,1)
    return new_pc

'''
Add noise to the edge-length-2 cude
'''
def background_noise(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    # c = [N//45, N//40, N//35, N//30, N//20][severity-1]
    c = N // 10
    jitter = np.random.uniform(-1,1,(B, c, C))
    pointcloud[:,N - c::,:] = jitter
    # new_pc = np.concatenate((pointcloud,jitter),axis=0).astype('float32')
    return pointcloud #normalize(new_pc)

'''
Upsampling
'''
def upsampling(pointcloud,severity = 1):
    B, N, C = pointcloud.shape
    # c = [N//5, N//4, N//3, N//2, N][severity-1]
    c = N // 10
    index = np.random.choice(ORIG_NUM, (B, c), replace=False)
    add = pointcloud[index] + np.random.uniform(-0.05,0.05, (B, c, C))
    pointcloud[:,N - c::,:] = add
    # new_pc = np.concatenate((pointcloud,add),axis=0).astype('float32')
    return pointcloud # normalize(new_pc)
    
'''
Add impulse noise
'''
def impulse_noise(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    # c = [N//30, N//25, N//20, N//15, N//10][severity-1]
    c = N // 60    
    for i in range(B):
        index = np.random.choice(ORIG_NUM, c, replace=False)
        pointcloud[i,index] += np.random.choice([-1,1], size=(c,C)) * 0.1

    return pointcloud #normalize(pointcloud)

### Point Number Modification ###

'''
Uniformly sampling the point cloud
'''
def uniform_sampling(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    # c = [N//15, N//10, N//8, N//6, N//2, 3 * N//4][severity-1]
    c = N // 30
    index = np.random.choice(ORIG_NUM, (B, ORIG_NUM - c), replace=False)
    return pointcloud[index]

def ffd_distortion(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    c = [0.1,0.2,0.3,0.4,0.5][severity-1]
    for i in range(B):
        pointcloud[i] = normalize(distortion.distortion(pointcloud[i].copy(), severity=c))
    return pointcloud

def rbf_distortion(pointcloud, severity = 1):
    N, C = pointcloud.shape
    c = [(0.025,5),(0.05,5),(0.075,5),(0.1,5),(0.125,5)][severity-1]
    new_pc = distortion.distortion_2(pointcloud,severity=c,func='multi_quadratic_biharmonic_spline')
    return normalize(new_pc).astype('float32')

def rbf_distortion_inv(pointcloud, severity = 1):
    N, C = pointcloud.shape
    c = [(0.025,5),(0.05,5),(0.075,5),(0.1,5),(0.125,5)][severity-1]
    new_pc = distortion.distortion_2(pointcloud,severity=c,func='inv_multi_quadratic_biharmonic_spline')
    return normalize(new_pc).astype('float32')

def height(batch_data, severity = 1):
    batch_data = batch_data
    height = np.min(batch_data, axis = 1)
    batch_data = batch_data - height[:,np.newaxis,:]

    return batch_data

def original(pointcloud, severity = 1):
    return pointcloud

'''
Cutout several part in the point cloud
'''
def cutout(pointcloud,severity = 1):
    B, N, C = pointcloud.shape
    c = [(2,30), (3,30), (5,30), (7,30), (10,30)][severity-1]
    for j in range(B):
        for _ in range(c[0]):
            i = np.random.choice(pointcloud[j].shape[0],1)
            picked = pointcloud[j][i]
            dist = np.sum((pointcloud[j] - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            pointcloud[j][idx.squeeze()] = 0
        # pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    return pointcloud

'''
Density-based up-sampling the point cloud
'''
def density_inc(pointcloud, severity):
    B, N, C = pointcloud.shape
    c = [(1,100), (2,100), (3,100), (4,100), (5,100)][severity-1]
    # idx = np.random.choice(N,c[0])
    # 
    for j in range(B):
        temp = []
        p_temp = pointcloud[j].copy()
        for _ in range(c[0]):
            i = np.random.choice(p_temp.shape[0],1)
            picked = p_temp[i]
            dist = np.sum((p_temp - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            # idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
            # idx = idx[idx_2]
            temp.append(p_temp[idx.squeeze()])
            p_temp = np.delete(p_temp, idx.squeeze(), axis=0)

        idx = np.random.choice(p_temp.shape[0],1024 - c[0] * c[1])
        temp.append(p_temp[idx.squeeze()])
        p_temp = np.concatenate(temp)

        pointcloud[j] = p_temp

    return pointcloud

'''
Density-based sampling the point cloud
'''
def density(pointcloud, severity):
    B, N, C = pointcloud.shape
    c = [(1,100), (2,100), (3,100), (4,100), (5,100)][severity-1]

    for j in range(B):
        p_temp = pointcloud[j].copy()
        for _ in range(c[0]):
            i = np.random.choice(p_temp.shape[0],1)
            picked = p_temp[i]
            dist = np.sum((p_temp - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
            idx = idx[idx_2]
            # p_temp = np.delete(p_temp, idx.squeeze(), axis=0)
            p_temp[idx.squeeze()] = 0

        pointcloud[j] = p_temp
    return pointcloud

def shapely(pointcloud, severity = 1):
    B, N, C = pointcloud.shape

    c = [0.01, 0.015, 0.02, 0.025, 0.03][severity-1]
    jitter = np.random.normal(size=(B, N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    # new_pc = np.clip(new_pc,-1,1)
    return new_pc

def get_trans(use_trans = True):
    MAP = {
        "Density":{
            # 'cutout': cutout,
            # 'density': density,
            # 'density_inc': density_inc,
            'original' : original, # 不变
            # 'occlusion': occlusion,
            # 'lidar': lidar,
        },
        "noise":{
            # 'uniform': uniform_noise,
            'gaussian': gaussian_noise,
            # 'impulse': impulse_noise,
            # 'upsampling': upsampling,
            # 'background': background_noise,
            'original' : original, # 不变
        },
        "Transformation":{
            # 'rotation': rotation,
            'shear': shear,
        },
        'size' : {
            'scale': scale,
            'original': original,
        },
        'height' : height,
    }

    trans_list = [original]

    if use_trans:
        trans_Transformation = random.choice(list(MAP['Transformation']))
        trans_noise = random.choice(list(MAP['noise']))
        trans_Density = random.choice(list(MAP['Density']))
        trans_size = random.choice(list(MAP['size']))

        trans_list = [MAP["Transformation"][trans_Transformation], MAP["noise"][trans_noise], MAP["Density"][trans_Density], MAP["size"][trans_size]]

    return trans_list

@DataTransforms.register_module()
class SpT(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        data['pos'] = data['pos'][np.newaxis]
        trans_list = get_trans(True)
        for f in trans_list:
            data['pos'][:,:,0:3] = f(data['pos'][:,:,0:3], severity = np.random.randint(1, 6))

        data['pos'] = data['pos'][0]

        return data

@DataTransforms.register_module()
class FarthestPS(object): # farthest_point_sample
    def __init__(self, **kwargs):
        self.npoints = 1024

    def __call__(self, data):
        if data['pos'].shape[0] == self.npoints:
            pass
        else:
            npoints = self.npoints
            point  = data['pos']
            N, D = point.shape
            xyz = point[:,:3]
            centroids = np.zeros((npoints,))
            distance = np.ones((N,)) * 1e10
            farthest = np.random.randint(0, N)
            for i in range(npoints):
                centroids[i] = farthest
                centroid = xyz[farthest, :]
                dist = np.sum((xyz - centroid) ** 2, -1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = np.argmax(distance, -1)
            point = point[centroids.astype(np.int32)]
            data['pos'] = point
        data['pos2'] = data['pos'].copy()

        return data

def trans(points, NT_trans_list, trans_trans_list):
    """
    points B N 3
    """
    points = points.numpy()

    if len(trans_trans_list) > 0:
        points_trans = points.copy()
        for f in trans_trans_list:
            points_trans = f(points_trans, severity = np.random.randint(1, 6))

        points_trans = torch.from_numpy(points_trans.transpose(0,2,1)).cuda()

    else:
        points_trans = None

    for f in NT_trans_list:
        points = f(points)
    points = torch.from_numpy(points.transpose(0,2,1)).cuda()

    return points, points_trans
