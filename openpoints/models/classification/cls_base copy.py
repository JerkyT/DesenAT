import torch
import torch.nn as nn
import logging
from typing import List
from ..layers.norm import create_norm1d
from ..layers import create_linearblock
from ...utils import get_missing_parameters_message, get_unexpected_parameters_message
from ..build import MODELS, build_model_from_cfg
from ...loss import build_criterion_from_cfg
from ...utils import load_checkpoint

from typing import List
from torch import Tensor
import torch.nn.functional as F

@MODELS.register_module()
class BaseCls(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 cls_args=None,
                 criterion_args=None, 
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        
        if cls_args is not None:
            in_channels = self.encoder.out_channels if hasattr(self.encoder, 'out_channels') else cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.prediction = build_model_from_cfg(cls_args)
        else:
            self.prediction = nn.Identity()
        self.criterion = build_criterion_from_cfg(criterion_args) if criterion_args is not None else None

    def forward(self, data, gt = None):
        global_feat = self.encoder.forward_cls_feat(data)
        if isinstance(self.prediction, nn.Identity):
            return self.prediction(global_feat)
        else:
            out = self.prediction(global_feat, gt)
            if not self.training and isinstance(self.prediction, list):
                return out[0]
            else:
                return out

    def get_loss(self, pred, gt, inputs=None):
        return self.criterion(pred, gt.long())

    def get_logits_loss(self, data, gt, cfg = None, epoch = None):
        if 'posc' in data:
            data['pos'] = torch.cat([data['posc'], data['pos']], 0)
            data['x'] = torch.cat([data['xc'], data['x']], 0)
            gt = torch.cat([gt, gt])
            if 'y2' in data:
                data['y2'] = torch.cat([data['y2'], data['y2']], 0)
        
        out = self.forward(data, gt)
        loss_extra = 0
        if isinstance(out, list):
            logits, logits_extra = out[0], out[1]
            loss_extra = self.criterion(logits_extra, gt.long())
        else:
            logits = out
            
        # mixup cutmix ~ 
        if 'y2' in data:
            gt2 = data['y2'].long()
            if isinstance(data['lam'], torch.Tensor):
                loss = 0
                for i in range(data['x'].shape[0]):
                    loss_tmp = self.criterion(logits[i].unsqueeze(0), gt[i].unsqueeze(0).long()) * (1 - data['lam'][i]) + self.criterion(logits[i].unsqueeze(0), gt2[i].unsqueeze(0).long()) * data['lam'][i]
                    loss += loss_tmp
                loss = loss / data['x'].shape[0]
            else:
                loss = self.criterion(logits, gt.long()) * (1 - data['lam']) + self.criterion(logits, gt2.long()) * data['lam']
        
            return logits, loss

        # DKD
        if not cfg == None:
            if cfg.criterion_args.NAME in ['DKD']:
                l_ce = self.criterion.get_ce(logits, gt)
                B, _ = logits.shape
                l_kd = self.criterion(logits[0:B // 2, :], logits[B // 2 ::, :], gt[0:B//2].long(), epoch)
                return logits, l_ce + l_kd

        return logits, self.criterion(logits, gt.long()) + loss_extra

@MODELS.register_module()
class DistillCls(BaseCls):
    def __init__(self,
                 encoder_args=None,
                 cls_args=None,
                 distill_args=None, 
                 criterion_args=None, 
                 **kwargs):
        super().__init__(encoder_args, cls_args, criterion_args)
        self.distill = encoder_args.get('distill', True)
        in_channels = self.encoder.distill_channels
        distill_args.distill_head_args.in_channels = in_channels
        self.dist_head = build_model_from_cfg(distill_args.distill_head_args)
        self.dist_model = build_model_from_cfg(distill_args).cuda()
        load_checkpoint(self.dist_model, distill_args.pretrained_path)
        self.dist_model.eval()

    def forward(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0['x']
        if self.distill and self.training: 
            global_feat, distill_feature = self.encoder.forward_cls_feat(p0, f0)
            return self.prediction(global_feat), self.dist_head(distill_feature)
        else:
            global_feat = self.encoder.forward_cls_feat(p0, f0)
            return self.prediction(global_feat)
         
    def get_loss(self, pred, gt, inputs):
        return self.criterion(inputs, pred, gt.long(), self.dist_model)

    def get_logits_loss(self, data, gt):
        logits, dist_logits = self.forward(data)
        return logits, self.criterion(data, [logits, dist_logits], gt.long(), self.dist_model)

@MODELS.register_module()
class ClsHead_ASE(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 mlps: List[int]=[256],
                 norm_args: dict=None,
                 act_args: dict={'act': 'relu'},
                 dropout: float=0.5,
                 cls_feat: str=None,
                 **kwargs
                 ):
        """A general classification head. supports global pooling and [CLS] token
        Args:
            num_classes (int): class num
            in_channels (int): input channels size
            mlps (List[int], optional): channel sizes for hidden layers. Defaults to [256].
            norm_args (dict, optional): dict of configuration for normalization. Defaults to None.
            act_args (_type_, optional): dict of configuration for activation. Defaults to {'act': 'relu'}.
            dropout (float, optional): use dropout when larger than 0. Defaults to 0.5.
            cls_feat (str, optional): preprocessing input features to obtain global feature. 
                                      $\eg$ cls_feat='max,avg' means use the concatenateion of maxpooled and avgpooled features. 
                                      Defaults to None, which means the input feautre is the global feature
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.cls_feat = cls_feat.split(',') if cls_feat is not None else None
        in_channels = len(self.cls_feat) * in_channels if cls_feat is not None else in_channels
        if mlps is not None:        
            mlps = [in_channels] + mlps + [num_classes]
        else:
            mlps = [in_channels, num_classes]

        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_linearblock(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))

        self.head = nn.Sequential(*heads)

        self.fc = nn.Linear(mlps[-2], mlps[-1], bias=True)
        # self.extra_fc = nn.Linear(mlps[-2], mlps[-1], bias=True)

    def forward(self, end_points, gt = None):
        if self.cls_feat is not None: 
            global_feats = [] 
            for preprocess in self.cls_feat:
                if 'max' in preprocess:
                    global_feats.append(torch.max(end_points, dim=1, keepdim=False)[0])
                elif preprocess in ['avg', 'mean']:
                    global_feats.append(torch.mean(end_points, dim=1, keepdim=False))
            end_points = torch.cat(global_feats, dim=1)

        ### by ours
        feature = self.head(end_points)
        N, C = feature.shape

        if self.training:
            logits_extra = self.fc(feature)
            fake_label = gt[torch.randperm(gt.size(0))]
            mask = self.fc.weight[gt, :]
            mask2 = self.fc.weight[fake_label, :]
            feature = feature * (0.6 * mask + 0.4 * mask2)
        else:
            logits_extra = self.fc(feature)
            pred_label = torch.max(logits_extra, dim=1)[1]
            mask = self.fc.weight[pred_label, :]
            feature = feature * mask.view(N, C)

        logits = self.fc(feature)
        ###

        return [logits, logits_extra]

@MODELS.register_module()
class ClsHead(nn.Module):
    def __init__(self,
                 num_classes: int, 
                 in_channels: int, 
                 mlps: List[int]=[256],
                 norm_args: dict=None,
                 act_args: dict={'act': 'relu'},
                 dropout: float=0.5,
                 cls_feat: str=None, 
                 **kwargs
                 ):
        """A general classification head. supports global pooling and [CLS] token
        Args:
            num_classes (int): class num
            in_channels (int): input channels size
            mlps (List[int], optional): channel sizes for hidden layers. Defaults to [256].
            norm_args (dict, optional): dict of configuration for normalization. Defaults to None.
            act_args (_type_, optional): dict of configuration for activation. Defaults to {'act': 'relu'}.
            dropout (float, optional): use dropout when larger than 0. Defaults to 0.5.
            cls_feat (str, optional): preprocessing input features to obtain global feature. 
                                      $\eg$ cls_feat='max,avg' means use the concatenateion of maxpooled and avgpooled features. 
                                      Defaults to None, which means the input feautre is the global feature
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.cls_feat = cls_feat.split(',') if cls_feat is not None else None
        in_channels = len(self.cls_feat) * in_channels if cls_feat is not None else in_channels
        if mlps is not None:        
            mlps = [in_channels] + mlps + [num_classes]
        else:
            mlps = [in_channels, num_classes]

        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_linearblock(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))
        heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points, gt = None):
        if self.cls_feat is not None: 
            global_feats = [] 
            for preprocess in self.cls_feat:
                if 'max' in preprocess:
                    global_feats.append(torch.max(end_points, dim=1, keepdim=False)[0])
                elif preprocess in ['avg', 'mean']:
                    global_feats.append(torch.mean(end_points, dim=1, keepdim=False))
            end_points = torch.cat(global_feats, dim=1)

        logits = self.head(end_points)
        return logits

@MODELS.register_module()
class APESClassifier(nn.Module):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 criterion_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__()
        self.prediction = nn.Identity()
        self.criterion = build_criterion_from_cfg(criterion_args) if criterion_args is not None else None

        self.backbone = build_model_from_cfg(backbone)
        self.neck = build_model_from_cfg(neck) if neck is not None else None
        self.head = build_model_from_cfg(head)

    def forward(self, data):
        x = self.extract_features(data['x'])
        pred_cls_logits = self.head(x)

        return pred_cls_logits

    def extract_features(self, inputs: Tensor) -> Tensor:
        x = self.backbone(inputs)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def get_loss(self, pred, gt, inputs=None):
        return self.criterion(pred, gt.long())

    def get_logits_loss(self, data, gt, cfg = None, epoch = None):
        if not 'posc' in data:
            logits = self.forward(data)

            return logits, self.criterion(logits, gt.long())
        else:
            data_input = {}
            logits = self.forward(data)

            # posc
            if not 'pnc' in data:
                data_input = {}
                data_input['pos'] = data['posc']
                data_input['x'] = data['xc']
                logits1 = self.forward(data_input)
            else:
                # pn
                data_input = {}
                data_input['pos'] = data['pnc']
                data_input['x'] = data['pn']
                logits1 = self.forward(data_input)

            if not cfg == None:
                if cfg.criterion_args.NAME in ['DKD']:
                    l_ce = self.criterion.get_ce(torch.cat([logits, logits1], 0), torch.cat([gt, gt], 0))
                    l_kd = self.criterion(logits, logits1, gt.long(), epoch)
                    return logits, l_ce + l_kd

            gt = torch.cat([gt, gt], 0)
            return logits, self.criterion(torch.cat([logits, logits1], 0), gt.long())
