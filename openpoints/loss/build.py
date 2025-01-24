import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from openpoints.utils import registry

LOSS = registry.Registry('loss')
LOSS.register_module(name='CrossEntropy', module=CrossEntropyLoss)
LOSS.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
LOSS.register_module(name='BCEWithLogitsLoss', module=BCEWithLogitsLoss)

@LOSS.register_module()
class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2, 
                 ignore_index=None, 
                 num_classes=None, 
                 weight=None, 
                 return_valid=False
                 ):
        super(SmoothCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.return_valid = return_valid
        # Reduce label values in the range of logit shape
        if ignore_index is not None:
            reducing_list = torch.range(0, num_classes).long().cuda(non_blocking=True)
            inserted_value = torch.zeros((1, )).long().cuda(non_blocking=True)
            self.reducing_list = torch.cat([
                reducing_list[:ignore_index], inserted_value,
                reducing_list[ignore_index:]
            ], 0)
        if weight is not None:
            self.weight = torch.from_numpy(weight).float().cuda(
                non_blocking=True).squeeze()
        else:
            self.weight = None
            
    def forward(self, pred, gt):
        if len(pred.shape)>2:
            pred = pred.transpose(1, 2).reshape(-1, pred.shape[1])
        gt = gt.contiguous().view(-1)
        
        if self.ignore_index is not None: 
            valid_idx = gt != self.ignore_index
            pred = pred[valid_idx, :]
            gt = gt[valid_idx]        
            gt = torch.gather(self.reducing_list, 0, gt)
            
        if self.label_smoothing > 0:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            if self.weight is not None:
                loss = -(one_hot * log_prb * self.weight).sum(dim=1).mean()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, weight=self.weight)
        
        if self.return_valid:
            return loss, pred, gt
        else:
            return loss

@LOSS.register_module()
class MaskedCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2):
        super(MaskedCrossEntropy, self).__init__()
        self.creterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, logit, target, mask):
        logit = logit.transpose(1, 2).reshape(-1, logit.shape[1])
        target = target.flatten()
        mask = mask.flatten()
        idx = mask == 1
        loss = self.creterion(logit[idx], target[idx])
        return loss

@LOSS.register_module()
class BCELogits(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = BCEWithLogitsLoss(**kwargs)
        
    def forward(self, logits, targets):
        if len(logits.shape)>2:
            logits = logits.transpose(1, 2).reshape(-1, logits.shape[1])
        targets = targets.contiguous().view(-1)
        num_clsses = logits.shape[-1]
        targets_onehot = F.one_hot(targets, num_classes=num_clsses).to(device=logits.device,dtype=logits.dtype)
        return self.criterion(logits, targets_onehot)

@LOSS.register_module()
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, logit, target):
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)  # N,C,H,W => N,C,H*W
            logit = logit.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logit = logit.contiguous().view(-1, logit.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(logit)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != logit.data.type():
                self.alpha = self.alpha.type_as(logit.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

@LOSS.register_module()
class Poly1CrossEntropyLoss(torch.nn.Module):
    def __init__(self,
                 num_classes: int =50,
                 epsilon: float = 1.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        if len(logits.shape)>2:
            logits = logits.transpose(1, 2).reshape(-1, logits.shape[1])
        labels = labels.contiguous().view(-1)

        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


@LOSS.register_module()
class Poly1FocalLoss(torch.nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False, 
                 **kwargs
                 ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon. the main one to finetune. larger values -> better performace in imagenet
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        num_classes = logits.shape[1]
        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1

@LOSS.register_module()
class MultiShapeCrossEntropy(torch.nn.Module):
    def __init__(self, criterion_args, **kwargs):
        super(MultiShapeCrossEntropy, self).__init__()
        self.criterion = build_criterion_from_cfg(criterion_args)

    def forward(self, logits_all_shapes, points_labels, shape_labels):
        batch_size = shape_labels.shape[0]
        losses = 0
        for i in range(batch_size):
            sl = shape_labels[i]
            logits = torch.unsqueeze(logits_all_shapes[sl][i], 0)
            pl = torch.unsqueeze(points_labels[i], 0)
            loss = self.criterion(logits, pl)
            losses += loss
        return losses / batch_size

def build_criterion_from_cfg(cfg, **kwargs):
    """
    Build a criterion (loss function), defined by cfg.NAME.
    Args:
        cfg (eDICT): 
    Returns:
        criterion: a constructed loss function specified by cfg.NAME
    """
    return LOSS.build(cfg, **kwargs)

##############################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# @LOSS.register_module()
class SmoothClsLoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothClsLoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

        loss = -(one_hot * pred).sum(dim=1).mean()
        return loss

def dkd_loss(logits_student, logits_teacher, target, alpha = 8, beta = 1, temperature = 4):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

@LOSS.register_module()
class DKD(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, **kwargs):
        super(DKD, self).__init__()
        print("-----------------------------------DKD-------------------------------------")
        self.ce_loss_weight = 1.0
        self.kd_loss_weight = 1.0
        self.alpha = 1.0
        self.beta = 8.0
        self.temperature = 4.0
        self.warmup = 20
        self.eps = 0.1

        self.ce = SmoothClsLoss()

    def forward(self, logits_student, logits_teacher, target, epoch):
        # pred = F.log_softmax(logits_student, -1)

        loss_dkd = self.kd_loss_weight * min(epoch / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )

        return loss_dkd
    
    def get_ce(self, logits, target):
        pred = F.log_softmax(logits, -1)
        loss_ce = self.ce_loss_weight * self.ce(pred, target)

        return loss_ce

def get_A(t1, t2):
    symbol1 = t1.clone().detach()
    symbol2 = t2.clone().detach()

    symbol1[symbol1 >= 0] = True
    symbol1[symbol1 < 0] = False

    symbol2[symbol2 >= 0] = True
    symbol2[symbol2 < 0] = False
    
    symbol = torch.matmul(symbol1, symbol2.permute(0,2,1))
    symbol[symbol == 1] = 1
    symbol[symbol == 0] = -1
    A = torch.matmul(t1.abs(), t2.permute(0,2,1).abs()) / t1.size(-1)

    A *= symbol
    return A

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def seje(logits_student, logits_teacher, target, temperature = 4): # B C C; B C C; B C
    one_hot = torch.zeros_like(logits_teacher).scatter(1, target.view(-1, 1), 1)
    one_hot = one_hot.unsqueeze(2)

    mask = torch.matmul(one_hot.float(), one_hot.permute(0,2,1).float())
    other_mask = torch.zeros_like(mask)
    other_mask[mask == 0] = 1

    B, C = logits_student.shape
    pred_student = F.softmax(logits_student/ temperature, -1)
    pred_teacher = F.softmax(logits_teacher/ temperature, -1)

    As = torch.matmul(pred_student.unsqueeze(2), pred_student.unsqueeze(2).permute(0,2,1)) / pred_student.size(-1)
    log_As = torch.log(As)
    At = torch.matmul(pred_teacher.unsqueeze(2), pred_teacher.unsqueeze(2).permute(0,2,1)) / pred_teacher.size(-1)
    # print((-(log_As * mask) * (At * mask)).view(-1, C * C).shape)
    # print((-(log_As * other_mask) * (At * other_mask) / C).view(-1, C * C).shape)
    s_tc = torch.cat([(log_As * mask).view(-1, C * C).sum(dim=1, keepdims=True) , (log_As * other_mask).view(-1, C * C).sum(dim=1, keepdims=True) / C], dim = 1)
    t_tc = torch.cat([(At * mask).view(-1, C * C).sum(dim=1, keepdims=True) , (At * other_mask).view(-1, C * C).sum(dim=1, keepdims=True) / C], dim = 1)

    tckd = (F.kl_div(s_tc, t_tc, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )

    gt_mask = _get_gt_mask(logits_student, target)
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )

    As2 = torch.matmul(log_pred_student_part2.unsqueeze(2), log_pred_student_part2.unsqueeze(2).permute(0,2,1)) / log_pred_student_part2.size(-1)
    log_As2 = torch.log(As2)
    At2 = torch.matmul(pred_teacher_part2.unsqueeze(2), pred_teacher_part2.unsqueeze(2).permute(0,2,1)) / pred_teacher_part2.size(-1)
    nckd = ((-log_As2 * At2).view(-1, C * C).sum(dim=1) / C).mean(dim=0)

    return 1 * tckd + 8 * nckd

@LOSS.register_module()
class JGEKD():
    """GKD JGEKD(by ours)"""
    def __init__(self, **kwargs):
        super(JGEKD, self).__init__()
        self.ce_loss_weight = 1.0
        self.kd_loss_weight = 1
        self.temperature = 4.0
        self.warmup = 5

        self.ce = SmoothClsLoss()

    def forward_train(self, logits_student, logits_teacher, target, epoch):

        loss_ce = self.criterion(logits_student,  target)

        loss_kd = self.kd_loss_weight * min(epoch / self.warmup, 1.0) * seje(logits_student, logits_teacher, target, self.temperature)

        loss = self.ce_loss_weight * loss_ce + loss_kd

        return loss
##########################################################################

@LOSS.register_module()
class DLB(nn.Module):
    """Self-Distillation from the Last Mini-Batch for Consistency Regularization(CVPR 2022)"""
    def __init__(self, **kwargs):
        super(DLB, self).__init__()
        print("-----------------------------------DLB-------------------------------------")
        self.ce_loss_weight = 1.0
        self.kd_loss_weight = 1.0
        self.temperature = 3.0

        self.ce = SmoothClsLoss()

    def forward(self, logits_student, logits_teacher, target, epoch):

        dml_loss = (
                        F.kl_div(
                            F.log_softmax(logits_student / self.temperature, dim=1),
                            F.softmax(logits_teacher.detach() / self.temperature, dim=1),  # detach
                            reduction="batchmean",
                        )
                        * self.temperature
                        * self.temperature
                    )

        return self.kd_loss_weight * dml_loss
    
    def get_ce(self, logits, target):

        return self.ce_loss_weight * F.cross_entropy(logits, target)

@LOSS.register_module()
class PSKD(nn.Module):
    def __init__(self, **kwargs):
        super(PSKD, self).__init__()

        print("-----------------------------------PSKD-------------------------------------")
        self.ce_loss_weight = 1.0
        self.kd_loss_weight = 1.0
        self.alpha_T = 3.0
        self.end_epoch = 300
    
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.all_predictions = torch.zeros(9840, 40, dtype=torch.float32)

    def forward(self, logits_student, logits_teacher, target, epoch, input_indices = 0):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """

        alpha_t = self.alpha_T * ((epoch + 1) / self.end_epoch)
        alpha_t = max(0, alpha_t)

        targets_numpy = target.cpu().detach().numpy()
        identity_matrix = torch.eye(40) 
