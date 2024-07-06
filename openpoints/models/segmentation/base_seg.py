"""
Author: PointNeXt
"""
import copy
from typing import List
import torch
import torch.nn as nn
import logging
from ...utils import get_missing_parameters_message, get_unexpected_parameters_message
from ..build import MODELS, build_model_from_cfg
from ..layers import create_linearblock, create_convblock1d
from ...loss import build_criterion_from_cfg

@MODELS.register_module()
class BaseSeg(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 longrange_args=None,
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        if decoder_args is not None:
            decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
            decoder_args_merged_with_encoder.update(decoder_args)
            decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder,
                                                                                                         'channel_list') else None
            self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder)
        else:
            self.decoder = None

        if longrange_args is not None:
            self.decoder.longrange = build_model_from_cfg(longrange_args)
        else:
            self.longrange = None

        if cls_args is not None:
            if hasattr(self.decoder, 'out_channels'):
                in_channels = self.decoder.out_channels
            elif hasattr(self.encoder, 'out_channels'):
                in_channels = self.encoder.out_channels
            else:
                in_channels = cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.head = build_model_from_cfg(cls_args)
        else:
            self.head = None

    def forward(self, data):
        p, f = self.encoder.forward_seg_feat(data)
        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)
        if self.head is not None:
            f = self.head(f)
        return f


@MODELS.register_module()
class BasePartSeg(BaseSeg):
    def __init__(self, encoder_args=None, decoder_args=None, cls_args=None, criterion_args=None, **kwargs):
        super().__init__(encoder_args, decoder_args, cls_args, **kwargs)

        self.criterion = build_criterion_from_cfg(criterion_args) if criterion_args is not None else None

        if cls_args is not None:
            in_channels = self.encoder.out_channels if hasattr(self.encoder, 'out_channels') else cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.prediction = build_model_from_cfg(cls_args)
        else:
            self.prediction = nn.Identity()

    def forward(self, p0, f0=None, cls0=None):
        if hasattr(p0, 'keys'):
            p0, f0, cls0 = p0['pos'], p0['x'], p0['cls']
        else:
            if f0 is None:
                f0 = p0.transpose(1, 2).contiguous()
        p, f = self.encoder.forward_seg_feat(p0, f0)
        if self.decoder is not None:
            f = self.decoder(p, f, cls0).squeeze(-1)
        elif isinstance(f, list):
            f = f[-1]
        if self.head is not None:
            f = self.head(f)
        return f
    
    def forward_cls(self, p0, f0=None, cls0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0['x']
        else:
            if f0 is None:
                f0 = p0.transpose(1, 2).contiguous()
        p, f = self.encoder.forward_seg_feat(p0, f0)
        return self.prediction(f[-1]).mean(-1).squeeze(-1)

    ###############
    def get_logits_loss(self, data, gt, cfg = None, epoch = None):
        if not 'posc' in data:
            logits = self.forward_cls(data)
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
            ####
            # print(logits.shape)
            # print(gt.shape)
            # exit()
            return logits, self.criterion(logits, gt.long())
        else:
            data_input = {}
            # print(data['xc'].shape, data['x'][:,3:6,:].shape, data['heightsc'].shape)
            # data_input['pos'] = torch.cat([data['pos'], data['posc']], 0)
            # data_input['x'] = torch.cat([data['x'], data['xc']], 0)
            # logits = self.forward(data_input)

            logits = self.forward_cls(data)
            # posc
            if not 'pnc' in data:
                data_input = {}
                data_input['pos'] = data['posc']
                data_input['x'] = data['xc']
                logits1 = self.forward_cls(data_input)
            else:
                # pn
                data_input = {}
                data_input['pos'] = data['pnc']
                data_input['x'] = data['pn']
                logits1 = self.forward_cls(data_input)

            if not cfg == None:
                if cfg.criterion_args.NAME in ['DKD']:
                    l_ce = self.criterion.get_ce(torch.cat([logits, logits1], 0), torch.cat([gt, gt], 0))
                    l_kd = self.criterion(logits, logits1, gt.long(), epoch)
                    return logits, l_ce + l_kd

            gt = torch.cat([gt, gt], 0)
            logits = torch.cat([logits, logits1], 0)
            if 'y2' in data:
                gt2 = torch.cat([data['y2'], data['y2']], 0)
                return logits, self.criterion(logits, gt.long()) * (1 - data['lam']) + self.criterion(logits, gt2.long()) * data['lam']
            return logits, self.criterion(logits, gt.long())

@MODELS.register_module()
class VariableSeg(BaseSeg):
    def __init__(self,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__(encoder_args, decoder_args, cls_args)
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

    def forward(self, data):
        p, f, b = self.encoder.forward_seg_feat(data)
        f = self.decoder(p, f, b).squeeze(-1)
        return self.head(f)


@MODELS.register_module()
class SegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 mlps=None,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0.5,
                 globals=None, 
                 **kwargs
                 ):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
            globals: global features to concat. [max,avg]. Set to None if do not concat any.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        if globals is not None:
            self.globals = globals.split(',')
            multiplier = len(self.globals) + 1
        else:
            self.globals = None
            multiplier = 1
        in_channels *= multiplier
        
        if mlps is None:
            mlps = [in_channels, in_channels] + [num_classes]
        else:
            if not isinstance(mlps, List):
                mlps = [mlps]
            mlps = [in_channels] + mlps + [num_classes]
        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_convblock1d(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))

        heads.append(create_convblock1d(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        if self.globals is not None: 
            global_feats = [] 
            for feat_type in self.globals:
                if 'max' in feat_type:
                    global_feats.append(torch.max(end_points, dim=-1, keepdim=True)[0])
                elif feat_type in ['avg', 'mean']:
                    global_feats.append(torch.mean(end_points, dim=-1, keepdim=True))
            global_feats = torch.cat(global_feats, dim=1).expand(-1, -1, end_points.shape[-1])
            end_points = torch.cat((end_points, global_feats), dim=1)
        logits = self.head(end_points)
        return logits


@MODELS.register_module()
class VariableSegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0.5,
                 **kwargs
                 ):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        mlps = [in_channels, in_channels] + [num_classes]

        heads = []
        print(mlps, norm_args, act_args)
        for i in range(len(mlps) - 2):
            heads.append(create_linearblock(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))

        heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        logits = self.head(end_points)
        return logits

@MODELS.register_module()
class MultiSegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0,
                 shape_classes=16,
                 num_parts=[4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3],
                 **kwargs
                 ):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        mlps = [in_channels, in_channels] + [shape_classes]
        self.multi_shape_heads = []

        self.num_parts=num_parts
        print(mlps, norm_args, act_args)
        self.shape_classes = shape_classes
        self.multi_shape_heads = nn.ModuleList()
        for i in range(shape_classes):
            head=[]
            for j in range(len(mlps) - 2):

                head.append(create_convblock1d(mlps[j], mlps[j + 1],
                                                norm_args=norm_args,
                                                act_args=act_args))
                if dropout:
                    head.append(nn.Dropout(dropout))
                head.append(nn.Conv1d(mlps[-2], num_parts[i], kernel_size=1, bias=True))
            self.multi_shape_heads.append(nn.Sequential(*head))

        # heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))

    def forward(self, end_points):
        logits_all_shapes = []
        for i in range(self.shape_classes):# per 16 shapes
            logits_all_shapes.append(self.multi_shape_heads[i](end_points))
        # logits = self.head(end_points)
        return logits_all_shapes


# # TODO: add distill for segmentation
# @MODELS.register_module()
# class DistillBaseSeg(BaseSeg):
#     def __init__(self,
#                  encoder_args=None,
#                  decoder_args=None,
#                  cls_args=None,
#                  distill_args=None, 
#                  criterion_args=None, 
#                  **kwargs):
#         super().__init__()
#         self.encoder = build_model_from_cfg(encoder_args)
#         if decoder_args is not None:
#             decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
#             decoder_args_merged_with_encoder.update(decoder_args)
#             decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder,
#                                                                                                          'channel_list') else None
#             self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder)
#         else:
#             self.decoder = None

#         if cls_args is not None:
#             if hasattr(self.decoder, 'out_channels'):
#                 in_channels = self.decoder.out_channels
#             elif hasattr(self.encoder, 'out_channels'):
#                 in_channels = self.encoder.out_channels
#             else:
#                 in_channels = cls_args.get('in_channels', None)
#             cls_args.in_channels = in_channels
#             self.head = build_model_from_cfg(cls_args)
#         else:
#             self.head = None

#     def forward(self, data):
#         p, f = self.encoder.forward_seg_feat(data)
#         if self.decoder is not None:
#             f = self.decoder(p, f).squeeze(-1)
#         if self.head is not None:
#             f = self.head(f)
#         return f