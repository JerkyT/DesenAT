import numpy as np
import torch
import sys
sys.path.append("./emd/")
import emd_module as emd
from torch.utils.tensorboard import SummaryWriter
import random

def normalize_array_torch(arr):
    # 矩阵归一化
    min_val = torch.min(arr)
    max_val = torch.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def chamfer_distance(pc1, pc2):
    """
    Calculate the Chamfer Distance between two point clouds.
    Input:
        pc1: Point cloud 1, [B, N, C]
        pc2: Point cloud 2, [B, M, C]
    Output:
        dist: Chamfer Distance, scalar
    """
    # Compute pairwise distances
    dist1 = square_distance_torch(pc1, pc2)  # [B, N, M]
    dist2 = square_distance_torch(pc2, pc1)  # [B, M, N]

    # Find the minimum distance from each point in pc1 to pc2
    dist1_min = torch.min(dist1, dim=2)[0]  # [B, N]
    dist2_min = torch.min(dist2, dim=2)[0]  # [B, M]

    print(dist1_min.shape, dist2_min.shape)
    # Sum the minimum distances
    chamfer_dist = torch.mean(dist1_min, dim = 1) + torch.mean(dist2_min, dim = 1)

    return chamfer_dist

def square_distance_torch(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).unsqueeze(2)
    dist += torch.sum(dst ** 2, -1).unsqueeze(1)

    return dist

def cutmix_r(data_batch, BETA = 1., PROB = 0.5, num_points = 1024):
    r = np.random.rand(1)
    if BETA > 0 and r < PROB:
        lam = np.random.beta(BETA, BETA)
        B = data_batch['x'].size()[0]

        rand_index = torch.randperm(B).cuda()
        target_a = data_batch['y']
        target_b = data_batch['y'][rand_index]

        point_a = torch.zeros(B, 1024, 3)
        point_b = torch.zeros(B, 1024, 3)
        point_c = torch.zeros(B, 1024, 3)
        point_a = data_batch['x']
        point_b = data_batch['x'][rand_index]
        point_c = data_batch['x'][rand_index]
        # point_a, point_b, point_c = point_a.to(device), point_b.to(device), point_c.to(device)

        remd = emd.emdModule()
        remd = remd.cuda()

        dis, ind = remd(point_a, point_b, 0.005, 300)
        for ass in range(B):
            point_c[ass, :, :] = point_c[ass, ind[ass].long(), :]

        int_lam = int(1024 * lam)
        int_lam = max(1, int_lam)
        gamma = np.random.choice(num_points, int_lam, replace=False, p=None)
        for i2 in range(B):
            data_batch['x'][i2, gamma, :] = point_c[i2, gamma, :]

        # adjust lambda to exactly match point ratio
        lam = int_lam * 1.0 / num_points
        # points = data_batch['pc'].transpose(2, 1)
        data_batch['y2'] = target_b
        data_batch['lam'] = lam

    return data_batch
        # pred, trans_feat = model(points)
        # loss = criterion(pred, target_a.long()) * (1. - lam) + criterion(pred, target_b.long()) * lam

def cutmix_k(data_batch, BETA = 1., PROB = 0.5, num_points = 1024):
    r = np.random.rand(1)
    if BETA > 0 and r < PROB:
        lam = np.random.beta(BETA, BETA)
        B = data_batch['x'].size()[0]

        rand_index = torch.randperm(B).cuda()
        target_a = data_batch['y']
        target_b = data_batch['y'][rand_index]

        point_a = torch.zeros(B, 1024, 3)
        point_b = torch.zeros(B, 1024, 3)
        point_c = torch.zeros(B, 1024, 3)
        point_a = data_batch['x']
        point_b = data_batch['x'][rand_index]
        point_c = data_batch['x'][rand_index]

        remd = emd.emdModule()
        remd = remd.cuda()
        dis, ind = remd(point_a, point_b, 0.005, 300)
        for ass in range(B):
            point_c[ass, :, :] = point_c[ass, ind[ass].long(), :]

        int_lam = int(num_points * lam)
        int_lam = max(1, int_lam)

        random_point = torch.from_numpy(np.random.choice(1024, B, replace=False, p=None))
        # kNN
        ind1 = torch.tensor(range(B))
        query = point_a[ind1, random_point].view(B, 1, 3)
        dist = torch.sqrt(torch.sum((point_a - query.repeat(1, num_points, 1)) ** 2, 2))
        idxs = dist.topk(int_lam, dim=1, largest=False, sorted=True).indices
        for i2 in range(B):
            data_batch['x'][i2, idxs[i2], :] = point_c[i2, idxs[i2], :]
        # adjust lambda to exactly match point ratio
        lam = int_lam * 1.0 / num_points
        # points = points.transpose(2, 1)
        # pred, trans_feat = model(points)
        # loss = criterion(pred, target_a.long()) * (1. - lam) + criterion(pred, target_b.long()) * lam
        data_batch['y2'] = target_b
        data_batch['lam'] = lam
        
    return data_batch

def mixup(data_batch, MIXUPRATE = 0.4):

    batch_size = data_batch['x'].size()[0]
    idx_minor = torch.randperm(batch_size)
    mixrates = (0.5 - np.abs(np.random.beta(MIXUPRATE, MIXUPRATE, batch_size) - 0.5))
    label_main = data_batch['y']
    label_minor = data_batch['y'][idx_minor]
    label_new = torch.zeros(batch_size, 40)
    for i in range(batch_size):
        if label_main[i] == label_minor[i]: # same label
            label_new[i][label_main[i]] = 1.0
        else:
            label_new[i][label_main[i]] = 1 - mixrates[i]
            label_new[i][label_minor[i]] = mixrates[i]
    label = label_new

    data_minor = data_batch['x'][idx_minor]
    mix_rate = torch.tensor(mixrates).float()
    mix_rate = mix_rate.unsqueeze_(1).unsqueeze_(2)

    mix_rate_expand_xyz = mix_rate.expand(data_batch['x'].shape).cuda()

    remd = emd.emdModule()
    remd = remd.cuda()
    _, ass = remd(data_batch['x'], data_minor, 0.005, 300)
    ass = ass.long()
    for i in range(batch_size):
        data_minor[i] = data_minor[i][ass[i]]
    data_batch['x'] = data_batch['x'] * (1 - mix_rate_expand_xyz) + data_minor * mix_rate_expand_xyz
    data_batch['y2'] = label_minor
    data_batch['lam'] = torch.tensor(mix_rate).squeeze_().cuda()

    return data_batch

def rsmix(data, BETA = 1., PROB = 0.5, n_sample=512, KNN=False):
    cut_rad = np.random.beta(BETA, BETA)
    data_batch = data['x'].cpu().numpy()
    label_batch = data['y'].cpu().numpy()

    rand_index = np.random.choice(data_batch.shape[0],data_batch.shape[0], replace=False) # label dim : (16,) for model
    
    if len(label_batch.shape) == 1:
        label_batch = np.expand_dims(label_batch, axis=1)
        
    label_a = label_batch[:,0]
    label_b = label_batch[rand_index][:,0]
        
    data_batch_rand = data_batch[rand_index] # BxNx3(with normal=6)
    rand_idx_1 = np.random.randint(0,data_batch.shape[1], (data_batch.shape[0],1))
    rand_idx_2 = np.random.randint(0,data_batch.shape[1], (data_batch.shape[0],1))
    if KNN:
        knn_para = min(int(np.ceil(cut_rad*n_sample)),n_sample)
        pts_erase_idx, query_point_1 = cut_points_knn(data_batch, rand_idx_1, cut_rad, nsample=n_sample, k=knn_para) # B x num_points_in_radius_1 x 3(or 6)
        pts_add_idx, query_point_2 = cut_points_knn(data_batch_rand, rand_idx_2, cut_rad, nsample=n_sample, k=knn_para) # B x num_points_in_radius_2 x 3(or 6)
    else:
        pts_erase_idx, query_point_1 = cut_points(data_batch, rand_idx_1, cut_rad, nsample=n_sample) # B x num_points_in_radius_1 x 3(or 6)
        pts_add_idx, query_point_2 = cut_points(data_batch_rand, rand_idx_2, cut_rad, nsample=n_sample) # B x num_points_in_radius_2 x 3(or 6)
    
    query_dist = query_point_1[:,:,:3] - query_point_2[:,:,:3]
    
    pts_replaced = np.zeros((1,data_batch.shape[1],data_batch.shape[2]))
    lam = np.zeros(data_batch.shape[0],dtype=float)

    for i in range(data_batch.shape[0]):
        if pts_erase_idx[i][0][0]==data_batch.shape[1]:
            tmp_pts_replaced = np.expand_dims(data_batch[i], axis=0)
            lam_tmp = 0
        elif pts_add_idx[i][0][0]==data_batch.shape[1]:
            pts_erase_idx_tmp = np.unique(pts_erase_idx[i].reshape(n_sample,),axis=0)
            tmp_pts_erased = np.delete(data_batch[i], pts_erase_idx_tmp, axis=0) # B x N-num_rad_1 x 3(or 6)
            dup_points_idx = np.random.randint(0,len(tmp_pts_erased), size=len(pts_erase_idx_tmp))
            tmp_pts_replaced = np.expand_dims(np.concatenate((tmp_pts_erased, data_batch[i][dup_points_idx]), axis=0), axis=0)
            lam_tmp = 0
        else:
            pts_erase_idx_tmp = np.unique(pts_erase_idx[i].reshape(n_sample,),axis=0)
            pts_add_idx_tmp = np.unique(pts_add_idx[i].reshape(n_sample,),axis=0)
            pts_add_idx_ctrled_tmp = pts_num_ctrl(pts_erase_idx_tmp,pts_add_idx_tmp)
            tmp_pts_erased = np.delete(data_batch[i], pts_erase_idx_tmp, axis=0) # B x N-num_rad_1 x 3(or 6)
            # input("INPUT : ")
            tmp_pts_to_add = np.take(data_batch_rand[i], pts_add_idx_ctrled_tmp, axis=0)
            tmp_pts_to_add[:,:3] = query_dist[i]+tmp_pts_to_add[:,:3]
            
            tmp_pts_replaced = np.expand_dims(np.vstack((tmp_pts_erased,tmp_pts_to_add)), axis=0)
            
            lam_tmp = len(pts_add_idx_ctrled_tmp)/(len(pts_add_idx_ctrled_tmp)+len(tmp_pts_erased))
        
        pts_replaced = np.concatenate((pts_replaced, tmp_pts_replaced),axis=0)
        lam[i] = lam_tmp
    
    data_batch_mixed = np.delete(pts_replaced, [0], axis=0)    
        
    data['x'] = torch.FloatTensor(data_batch_mixed).cuda()
    data['y'] = torch.tensor(label_a).cuda()
    data['y2'] = torch.tensor(label_b).cuda()
    data['lam'] = torch.tensor(lam).cuda()

    return data

def knn_points(k, xyz, query, nsample=512):
    B, N, C = xyz.shape
    _, S, _ = query.shape # S=1
    
    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis,np.newaxis,:], B, axis=0)
    sqrdists = square_distance(query, xyz) # Bx1,N #제곱거리
    tmp = np.sort(sqrdists, axis=2)
    knn_dist = np.zeros((B,1))
    for i in range(B):
        knn_dist[i][0] = tmp[i][0][k]
        group_idx[i][sqrdists[i]>knn_dist[i][0]]=N
    # group_idx[sqrdists > radius ** 2] = N
    # print("group idx : \n",group_idx)
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:,:,0]
    group_first = np.repeat(tmp_idx[:,np.newaxis,:], nsample, axis=2)
    # repeat the first value of the idx in each batch 
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def cut_points_knn(data_batch, idx, radius, nsample=512, k=512):
    """
        input
        points : BxNx3(=6 with normal)
        idx : Bx1 one scalar(int) between 0~len(points)
        
        output
        idx : Bxn_sample
    """
    B, N, C = data_batch.shape
    B, S = idx.shape
    query_points = np.zeros((B,1,C))
    # print("idx : \n",idx)
    for i in range(B):
        query_points[i][0]=data_batch[i][idx[i][0]] # Bx1x3(=6 with normal)
    # B x n_sample
    group_idx = knn_points(k=k, xyz=data_batch[:,:,:3], query=query_points[:,:,:3], nsample=nsample)
    return group_idx, query_points # group_idx: 16x?x6, query_points: 16x1x6

def cut_points(data_batch, idx, radius, nsample=512):
    """
        input
        points : BxNx3(=6 with normal)
        idx : Bx1 one scalar(int) between 0~len(points)
        
        output
        idx : Bxn_sample
    """
    B, N, C = data_batch.shape
    B, S = idx.shape
    query_points = np.zeros((B,1,C))
    # print("idx : \n",idx)
    for i in range(B):
        query_points[i][0]=data_batch[i][idx[i][0]] # Bx1x3(=6 with normal)
    # B x n_sample
    group_idx = query_ball_point_for_rsmix(radius, nsample, data_batch[:,:,:3], query_points[:,:,:3])
    return group_idx, query_points # group_idx: 16x?x6, query_points: 16x1x6

def query_ball_point_for_rsmix(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample], S=1
    """
    # device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis,np.newaxis,:], B, axis=0)
    
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:,:,0]
    group_first = np.repeat(tmp_idx[:,np.newaxis,:], nsample, axis=2)
    # repeat the first value of the idx in each batch 
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # dist += torch.sum(dst ** 2, -1).view(B, 1, M)

    dist = -2 * np.matmul(src, dst.transpose(0, 2, 1))
    dist += np.sum(src ** 2, -1).reshape(B, N, 1)
    dist += np.sum(dst ** 2, -1).reshape(B, 1, M)
    
    return dist

def pts_num_ctrl(pts_erase_idx, pts_add_idx):
    '''
        input : pts - to erase 
                pts - to add
        output :pts - to add (number controled)
    '''
    if len(pts_erase_idx)>=len(pts_add_idx):
        num_diff = len(pts_erase_idx)-len(pts_add_idx)
        if num_diff == 0:
            pts_add_idx_ctrled = pts_add_idx
        else:
            pts_add_idx_ctrled = np.append(pts_add_idx, pts_add_idx[np.random.randint(0, len(pts_add_idx), size=num_diff)])
    else:
        pts_add_idx_ctrled = np.sort(np.random.choice(pts_add_idx, size=len(pts_erase_idx), replace=False))
    return pts_add_idx_ctrled

def get_m(cfg):
    if 'ST_' in cfg.pretrained_path:
        m = '_ST'
    elif 'AT_' in cfg.pretrained_path:
        m = '_AT'
    elif 'TN_' in cfg.pretrained_path:
        m = '_TN'
    elif 'cutmix_r' in cfg.pretrained_path:
        m = '_cutmix_r'
    elif 'cutmix_k' in cfg.pretrained_path:
        m = '_cutmix_k'
    elif 'mixup' in cfg.pretrained_path:
        m = '_mixup'
    elif 'rsmix' in cfg.pretrained_path:
        m = '_rsmix'
    elif 'round_' in cfg.pretrained_path:
        m = '_round'
    elif 'Quantify_' in cfg.pretrained_path:
        m = '_Quantify'
    elif 'Supervoxel_' in cfg.pretrained_path:
        m = '_Supervoxel'
    elif 'TND_' in cfg.pretrained_path:
        m = '_TND'
    elif 'TQ_' in cfg.pretrained_path:
        m = '_TQ'
    elif '_ASE' in cfg.pretrained_path:
        m = '_ASE'
    elif 'FFT' in cfg.pretrained_path:
        m = '_FFT'
    elif 'Decoupling' in cfg.pretrained_path:
        m = '_Decoupling'
    else:
        m = '_M1'
    return m

def vis_weights(model):
    ########
    # 可视化
    writer = SummaryWriter('./AT')
    for i, (name, param) in enumerate(model.named_parameters()):
        # print(name, '   ', param.shape)

        weights = param.clone().cpu().data.numpy()
        if len(weights.shape) >= 2:
            for j in range(weights.shape[1]): 
                writer.add_histogram(name , weights[:,j], j)
        # if name == 'module.sa1.mlp_convs.1.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa1.mlp_convs.1.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa2.mlp_convs.0.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa2.mlp_convs.0.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa2.mlp_convs.1.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa2.mlp_convs.1.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa3.mlp_convs.0.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa3.mlp_convs.0.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa3.mlp_convs.1.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa3.mlp_convs.1.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa4.mlp_convs.0.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa4.mlp_convs.0.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa4.mlp_convs.1.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa4.mlp_convs.1.weight' , weights[:,j,:,:], j)
    writer.close()
    ########
