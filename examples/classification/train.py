import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.modeling import build_modeling_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
import shap
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast, GradScaler

import random
from .aug import *

def get_features_by_keys(input_features_dim, data):
    if input_features_dim == 3:
        features = data['pos']
    elif input_features_dim == 4:
        features = torch.cat(
            (data['pos'], data['heights']), dim=-1)
        raise NotImplementedError("error")
    return features.transpose(1, 2).contiguous()

def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
    accs_table = [f'{item:.2f}' for item in accs]
    header = ['method', 'OA', 'mAcc'] + \
        cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
        str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()

def print_cls_results(oa, macc, accs, epoch, cfg):
    s = f'\nClasses\tAcc\n'
    for name, acc_tmp in zip(cfg.classes, accs):
        s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
    s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
    logging.info(s)

def main(gpu, cfg, profile=False):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank) 
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0 :
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    ### 建模
    modeling = build_modeling_from_cfg(cfg.modeling)
    ###
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    num_points = val_loader.dataset.num_points if hasattr(
        val_loader.dataset, 'num_points') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}, "
                 f"number of points sampled from dataset: {num_points}, "
                 f"number of points as model input: {cfg.num_points}")
    cfg.classes = cfg.get('classes', None) or val_loader.dataset.classes if hasattr(
        val_loader.dataset, 'classes') else None or np.range(num_classes)

    # cfg.classes = 40
    
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
            print_cls_results(oa, macc, accs, cfg.start_epoch, cfg)
        else:
            if cfg.mode == 'test' or cfg.mode == 'attact':
                # test mode
                epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                            ###
                # by outs attact
                if cfg.mode == 'attact':
                    # test mode
                    epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)

                    macc, oa, accs, cm = attact(model, test_loader, Graph_Domain_Skeleton_Attack, cfg)
                    print_cls_results(oa, macc, accs, epoch, cfg)

                    return True
                ###
                macc, oa, accs, cm = validate_fn(model, test_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)

                #### by ours
                m = get_m(cfg)

                if cfg.dataset.common.NAME in ['ModelNet40_C', 'ModelNet40_C6']:
                    with open('./' + cfg.dataset.common.NAME + '_' + cfg.model.encoder_args.NAME + m + '.txt', "a+") as file:
                        file.write(str(cfg.dataset.common.corruption) + '-' + str(cfg.dataset.common.severity) + ":" + '\n')
                        file.write('    mAcc: %.2f' % (macc) + '\n')
                        file.write('    OA: %.2f' % (oa) + '\n')

                    if cfg.dataloader.gen_shaply:
                        shaply(model, test_loader, cfg, root_path = './data/shaply_' + cfg.dataset.common.NAME + '/' + \
                                cfg.model.encoder_args.NAME + m \
                                + '/' + cfg.mode + str(cfg.dataloader.n_eval) + '/' + str(cfg.dataset.common.corruption) + '_' + str(cfg.dataset.common.severity) + '/')
                        return True
                if cfg.dataloader.gen_shaply:
                        shaply(model, test_loader, cfg, root_path = './data/shaply_' + cfg.dataset.common.NAME + '/' + \
                                cfg.model.encoder_args.NAME + m \
                                + '/' + cfg.mode + str(cfg.dataloader.n_eval) + '/')
                        return True

                return True
                ####

            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
    else:
        logging.info('Training from scratch')

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                    cfg.dataset,
                                    cfg.dataloader,
                                    datatransforms_cfg=cfg.datatransforms,
                                    split='train',
                                    distributed=cfg.distributed,
                                    )

    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    if cfg.dataloader.gen_shaply:
        if 'ST_' in cfg.pretrained_path:
            m = '_ST'
        elif 'AT_' in cfg.pretrained_path:
            m = '_AT'
        else:
            m = '_ATDKD'
        if not cfg.mode == 'test':
            epoch, best_val = load_checkpoint(
                        model, pretrained_path=cfg.pretrained_path)
            shaply(model, train_loader, cfg, root_path = './data/shaply_' + cfg.dataset.common.NAME + '/' + \
                    cfg.model.encoder_args.NAME + m \
                    + '/' + cfg.mode + str(cfg.dataloader.n_eval) + '/')
            return True

    # ===> start training
    val_macc, val_oa, val_accs, best_val, macc_when_best, best_epoch = 0., 0., [], 0., 0., 0

    scaler = GradScaler() if cfg.dataloader.shapley else None
    #
    model.zero_grad()
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        # try:
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        train_loss, train_macc, train_oa, _, _ = \
            train_one_epoch(model, train_loader,
                            optimizer, scheduler, epoch, cfg, scaler, modeling)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_macc, val_oa, val_accs, val_cm = validate_fn(
                model, val_loader, cfg)
            is_best = val_oa > best_val
            if is_best:
                best_val = val_oa
                macc_when_best = val_macc
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')
                print_cls_results(val_oa, val_macc, val_accs, epoch, cfg)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                    f'train_oa {train_oa:.2f}, val_oa {val_oa:.2f}, best val oa {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_oa', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('mAcc_when_best', macc_when_best, epoch)
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('epoch', epoch, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
        # except:
        #     print(epoch)
        #     continue
    # test the last epoch
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, epoch)
        writer.add_scalar('test_macc', test_macc, epoch)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, best_epoch)
        writer.add_scalar('test_macc', test_macc, best_epoch)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)

    if writer is not None:
        writer.close()
    dist.destroy_process_group()

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg, scaler = None, modeling = None):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), ascii=True)
    num_iter = 0
    
    for idx, data in pbar:
        num_iter += 1

        # points = data['x']
        # points = point_debug(npoints, points)
        #### by ours
        if 'Spatial_geometry_Enhancement' in cfg.datatransforms.train:
            num_p = random.randint(512, 1024)
            data['x'] = data['x'][:, :num_p, :]
        else:
            data['x'] = data['x']
        # Additional steps to help manage memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            cfg.datatransforms.c
        except:
            pass
        else:
            if 'Spatial_geometry_Enhancement' in cfg.datatransforms.c:
                num_p = random.randint(512, 1024)
                data['xc'] = data['xc'][:, :num_p, :]

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        ####
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        if cfg.dataloader.shapley:
            if 'xc' in data.keys():
                if 'shapely' in data.keys():
                    shapely = data['shapely']
                    num_p = random.randint(512, 1024)

                    _, sort_indices = torch.sort(shapely, dim=-1) # 升序
                    indices = sort_indices[:, 0:num_p]
                    target = data['y']
                    data['xc'] = data['xc'][torch.arange(target.shape[0]).unsqueeze(1), indices, :]
            else:
                if 'shapely' in data.keys():
                    num_p = random.randint(512, 1024)

                    # data['x'] = data['x'][:,0:num_p,:3]

                    _, sort_indices = torch.sort(data['shapely'], dim=-1) # 升序
                    indices = sort_indices[:, 0:num_p]
                    target = data['y']
                    data['x'] = data['x'][torch.arange(target.shape[0]).unsqueeze(1), indices, :]

        # print(data['x'].shape)
        # print(data['xc'].shape)

        ###### aug
        if cfg.AUG == 'cutmix_r':
            data = cutmix_r(data) # x B N C
        elif cfg.AUG == 'cutmix_k':
            data = cutmix_k(data)
        elif cfg.AUG == 'mixup':
            data = mixup(data)
        elif cfg.AUG == 'rsmix':
            data = rsmix(data)

        # print(data['pos'].shape)
        points = data['x']
        target = data['y']
        
        ######
        for key, key_pos in [('xc', 'posc')]:
            if key in data.keys():
                pointsc = data[key]

                data[key_pos] = pointsc[:, :, :3].contiguous()
                data[key] = pointsc[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        ######
        if modeling:
            data = modeling(data)

        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()

        logits, loss = model.get_logits_loss(data, target, cfg, epoch) if not hasattr(model, 'module') else model.module.get_logits_loss(data, target, cfg, epoch)
        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        if not target.shape[0] == logits.shape[0]:
            target = torch.cat([target, target], 0)

        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    macc, overallacc, accs = cm.all_acc()
    return loss_meter.avg, macc, overallacc, accs, cm

def point_debug(npoints, points):
    """ bebug
    from openpoints.dataset import vis_points 
    vis_points(data['pos'].cpu().numpy()[0])
    """
    num_curr_pts = points.shape[1]

    if num_curr_pts != npoints:  # point resampling strategy
        if npoints == 1024:
            point_all = 1200
        elif npoints == 4096:
            point_all = 4800
        elif npoints == 8192:
            point_all = 8192
        elif npoints == 2048:
            point_all == 2048
        else:
            raise NotImplementedError()
        if  points.size(1) < point_all:
            point_all = points.size(1)
        fps_idx = furthest_point_sample(
            points[:, :, :3].contiguous(), point_all)
        fps_idx = fps_idx[:, np.random.choice(
            point_all, npoints, False)]
        points = torch.gather(
            points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))
    return points

def norm_l2_loss(adv_pc, ori_pc): # Dnorm
    return ((adv_pc - ori_pc)**2).sum(1).sum(1)

def hausdorff_loss(adv_pc, ori_pc): # Dh
    #dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    #hd_loss = torch.max(torch.min(dis, dim=2)[0], dim=1)[0]
    adv_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    hd_loss = adv_KNN.dists.contiguous().squeeze(-1).max(-1)[0] #[b]
    return hd_loss

@torch.no_grad()
def validate(model, val_loader, cfg): # get_top_25_indices
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), ascii=True)

    # 存储每个类别的样本及其对应的预测概率和索引号
    class_samples = {i: [] for i in range(cfg.num_classes)}

    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        points = data['x']
        points = points[:, :npoints]
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
            
        if hasattr(model, 'forward_cls'):
            logits = model.forward_cls(data)
        else:
            logits = model(data)
        
        ####
        probabilities  = torch.softmax(logits, dim=1)

        for i in range(points.size(0)):
            label = target[i].item()
            prob = probabilities[i][label].item()
            index = idx * val_loader.batch_size + i  # 计算样本在数据集中的索引号
            class_samples[label].append((prob, index))
        ####
        cm.update(logits.argmax(dim=1), target)

    tp, count = cm.tp, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    # Sort by probability and take the top 25

    return macc, overallacc, accs, cm

@torch.no_grad()
def shaply(model, val_loader, cfg, root_path):
    shaplys_p, shaplys_n, fft, sources = [], [], [], []

    n_evals = cfg.dataloader.n_eval
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    model.eval() # set model to eval mode
    npoints = cfg.num_points
    if not os.path.exists(root_path): 
        os.makedirs(root_path)

    def predict(points):
        ###
        points = points[:,:,:,0]
        points = torch.from_numpy(points).cuda()
        ###
        points = points[:, :npoints]
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)
        return logits

    for idx, data in enumerate(val_loader):

        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        points = data['x']
        target = data['y']

        if 'v_real' in data:
            v_real = data['v_real']

        inx_target = int(target[0])
        x_ = torch.einsum('bij,bjk->bik', v_real.transpose(1,2), points).cpu().numpy()[:,:,:, np.newaxis]  # Use v_real here
        logits = predict(x_)
        source = F.softmax(logits, 1).cpu().numpy() # [:, inx_target] # n c
        sources.append(source)
        ###
        # 定义一个SHAP的背景数据集，这里使用一个全零的tensor
        background = torch.zeros_like(points[0:1].unsqueeze(-1))

        # # 定义一个SHAP的解释器
        masker_blur = shap.maskers.Image("blur(128,128)", background[0].shape)
        explainer = shap.Explainer(predict, masker_blur)
        shap_values = explainer(x_, max_evals=n_evals)[:,:,:,0,:].values # (1, 1024, C, 1, cls)

        p = shap_values[:,:,:,inx_target]
        n = np.concatenate((shap_values[:,:,:,:inx_target], shap_values[:,:,:, inx_target+1:]), 3)

        shaplys_p.append(p)
        shaplys_n.append(n.mean(-1))
        # ###
        cm.update(logits.argmax(dim=1), target)

    tp, count = cm.tp, cm.count

    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    print(overallacc)
    np.save(root_path + 'final_p.npy', np.concatenate(shaplys_p, 0))
    np.save(root_path + 'final_n.npy', np.concatenate(shaplys_n, 0))
