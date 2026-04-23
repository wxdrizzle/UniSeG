from core.models.base import Base
from core.losses.NCC import NCC
from core.networks.ours2 import Ours2Net
from core.losses.kl_v import SpatialSmoothKL
from core.utils.seg import logits2seg, seg2onehot
import torch
import torch.distributions as tcdist
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from collections import OrderedDict
from einops import rearrange, repeat
import monai
from geomloss import SamplesLoss
from core.losses.distance_qz import GaussianDistance
from skimage import measure
import torch.nn as nn
import time
from itertools import combinations
from core.losses.probability_distance import hellinger_distance, fisher_rao_distance
from core.losses.LocalDisplacementEnergy import JacobianDeterminant
from monai.inferers import sliding_window_inference
import os
from monai.data import MetaTensor
import SimpleITK as sitk
from scipy.stats import spearmanr

torch._dynamo.config.traceable_tensor_subclasses.update({MetaTensor})

mpl.use('agg')


def get_img_grid(n_imgs=1, size_img=(224, 224), padding_min=4, thickness=1, spacing=9):
    """Get images of white grids. 

    Args:
        n_imgs:
        size_img: Height and width of the image.
        padding_min: Minimum padding.
        thickness: Thickness of each line.
        spacing: Spacing between two adjacent lines.
    Returns:
        img_grid: Shape of [n_imgs, 1, *size_img]. img_grid[i] is identical for i.
    """
    img_grid = torch.zeros(n_imgs, 1, *size_img)

    for dim in [0, 1]:
        n_lines = np.floor((size_img[dim] - 2*padding_min - thickness) / (thickness+spacing)) + 1
        padding = (size_img[dim] - ((n_lines-1) * (thickness+spacing) + thickness)) // 2
        idxs = np.arange(n_lines) * (thickness+spacing) + padding
        idxs = idxs.astype(int)

        for i in range(thickness):
            if dim == 0:
                img_grid[..., idxs + i, :] = 1
            else:
                img_grid[..., :, idxs + i] = 1
    return img_grid

class BufferDict(nn.Module):
    def __init__(self, input_dict=None):
        super().__init__()
        if input_dict is not None:
            for k,v in input_dict.items():
                self.register_buffer(k, v)

    def update(self, k, v):
        self.register_buffer(k, v)

    def get_values(self):
        return list(self._buffers.values())

class Ours2(Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.net: Ours2Net = Ours2Net(cfg)
        if cfg.exp.compile_model:
            print('------------- Compiling model -------------')
            start = time.time()
            self.net = torch.compile(self.net, fullgraph=True, backend="eager")
            print(f'------------- Compilation finished, took {time.time() - start:.2f} s -------------')
        self.recorder = {}

        self.calc_loss_kl_v = SpatialSmoothKL(cfg.dataset.dim, prior_lambda=10, disp_levels=tuple(cfg.net.reg.levels))

        self.calc_loss_dice = monai.losses.DiceLoss(include_background=True, softmax=True, squared_pred=False,
                                                    jaccard=False, reduction='none', smooth_nr=1e-05, smooth_dr=1e-05,
                                                    batch=False, weight=None)
        if self.cfg.model.ws_loss.kl_c_domain_diff > 0.:
            assert self.cfg.model.losses.kl_c_domain_diff.method == 'wasserstein'
            if self.cfg.model.losses.kl_c_domain_diff.cost == 'l2':
                self.calc_loss_kl_c_domain_diff = SamplesLoss(
                    loss='sinkhorn', p=2, blur=self.cfg.model.losses.kl_c_domain_diff.wasserstein.blur, 
                    scaling=self.cfg.model.losses.kl_c_domain_diff.wasserstein.scaling)
            elif self.cfg.model.losses.kl_c_domain_diff.cost == 'fisher_rao':
                self.calc_loss_kl_c_domain_diff = SamplesLoss(
                    loss='sinkhorn', p=1, blur=self.cfg.model.losses.kl_c_domain_diff.wasserstein.blur,
                    cost=fisher_rao_distance,
                    scaling=self.cfg.model.losses.kl_c_domain_diff.wasserstein.scaling)
        if self.cfg.model.ws_loss.v_domain_diff_region_wise > 0.:
            self.calc_loss_v_domain_diff_region_wise = SamplesLoss(
                loss=self.cfg.model.losses.v_domain_diff_region_wise.method, p=2,
                blur=self.cfg.model.losses.v_domain_diff_region_wise.blur)

        if self.cfg.model.ws_loss.recon_ncc > 0.:
            self.calc_loss_recon_ncc = NCC(win=7)

        if self.cfg.model.losses.ce.ws_class is not None:
            ws = torch.tensor(self.cfg.model.losses.ce.ws_class)
            assert ws[0] == 1, 'background class should always have weight 1; ths weights are pre-normalized'
            ws = ws / ws.sum()
        else:
            ws = None
        self.calc_loss_ce = torch.nn.CrossEntropyLoss(weight=ws, ignore_index=-100, reduction='none',
                                                      label_smoothing=0.0)

        queue_size = self.cfg.model.losses.kl_c_domain_diff.queue_size
        if queue_size > 0:
            if self.cfg.exp.mode == 'test':
                self.register_buffer('queue_qc_source', torch.empty(queue_size, self.cfg.net.atlas.num))
            elif self.cfg.exp.mode == 'train':
                if self.cfg.model.source_free_stage_2.enable:
                    self.register_buffer('queue_qc_source', torch.empty(queue_size, self.cfg.net.atlas.num))
                else:
                    self.register_buffer('queue_qc_source', torch.empty(0, self.cfg.net.atlas.num))
            else:
                raise ValueError(f'unknown mode: {self.cfg.exp.mode}')
            
        if self.cfg.model.source_free_stage_2.enable:
            self.cfg.var.domains = ['target']

        if self.cfg.exp.val.metrics.logdet_covar_w or self.cfg.exp.test.metrics.logdet_covar_w:
            A = self.cfg.net.atlas.num
            one = np.ones(A)
            u = one / np.linalg.norm(one)
            rng = np.random.default_rng(42)
            M = rng.standard_normal((A, A))
            M[:, 0] = u
            Qmat, _ = np.linalg.qr(M)
            U = Qmat[:, 1:]    # [A, A-1]
            self.U_for_logdet_covar_w = U

    def clear_nested_dict_or_list(self, d):
        if isinstance(d, dict):
            keys = list(d.keys())
        elif isinstance(d, list):
            keys = reversed(list(range(len(d))))
        else:
            print(type(d))
            raise TypeError

        for k in keys:
            v = d[k]
            if isinstance(v, dict) or isinstance(v, list):
                self.clear_nested_dict_or_list(v)
            else:
                try:
                    assert type(v) is torch.Tensor or type(v) is str or type(v) is np.ndarray or v is None, type(v)
                except Exception as e:
                    breakpoint()
            del d[k]

    def forward(self, data):
        self.clear_nested_dict_or_list(self.recorder)
        data = super().forward(data)

        for k, v in data.items():
            if type(v) is monai.data.meta_tensor.MetaTensor:
                data[k] = v.as_tensor()

        if hasattr(self.net, 'data'):
            del self.data
            self.clear_nested_dict_or_list(self.net.data)
            del self.net.data
        self.net.data = self.data = data
        
        if self.training and (self.cfg.dataset.aug.rand_rotation or self.cfg.dataset.aug.rand_flip):
            for domain in ['source', 'target']:
                if f'img_{domain}' in data:
                    if self.cfg.dataset.aug.rand_rotation:
                        if data[f'img_{domain}'].shape[-1] == data[f'img_{domain}'].shape[-2]:
                            # Generate random rotation angles (0, 90, 180, or 270 degrees)
                            angles = torch.randint(0, 4, (data[f'img_{domain}'].shape[0],), device=data[f'img_{domain}'].device) * 90
                        else:
                            # only allow 0 or 180 degree rotation for non-square images
                            angles = torch.randint(0, 2, (data[f'img_{domain}'].shape[0],), device=data[f'img_{domain}'].device) * 180
                        # Store rotation angles
                        data[f'rotation_{domain}'] = angles
                    if self.cfg.dataset.aug.rand_flip:
                        # 0: no flip; 1: horizontal flip; 2: vertical flip
                        flips = torch.randint(0, 3, (data[f'img_{domain}'].shape[0],), device=data[f'img_{domain}'].device)
                        data[f'flip_{domain}'] = flips

                    # Apply rotation to images and segmentation masks
                    for i in range(data[f'img_{domain}'].shape[0]):
                        if self.cfg.dataset.aug.rand_rotation:
                            data[f'img_{domain}'][i] = torch.rot90(data[f'img_{domain}'][i], k=angles[i]//90, dims=[-2, -1])
                        if self.cfg.dataset.aug.rand_flip:
                            if flips[i] == 1:
                                data[f'img_{domain}'][i] = torch.flip(data[f'img_{domain}'][i], dims=[-1])
                            elif flips[i] == 2:
                                data[f'img_{domain}'][i] = torch.flip(data[f'img_{domain}'][i], dims=[-2])
                        if f'seg_{domain}' in data:
                            if self.cfg.dataset.aug.rand_rotation:
                                data[f'seg_{domain}'][i] = torch.rot90(data[f'seg_{domain}'][i], k=angles[i]//90, dims=[-2, -1])
                            if self.cfg.dataset.aug.rand_flip:
                                if flips[i] == 1:
                                    data[f'seg_{domain}'][i] = torch.flip(data[f'seg_{domain}'][i], dims=[-1])
                                elif flips[i] == 2:
                                    data[f'seg_{domain}'][i] = torch.flip(data[f'seg_{domain}'][i], dims=[-2])

        
        if self.training:
            queue_size = self.cfg.model.losses.kl_c_domain_diff.queue_size
            if queue_size > 0:
                assert not self.cfg.net.atlas.qc.calc.levelwise
                if self.cfg.model.source_free_stage_2.enable:
                    pass
                else:
                    probs = self.net.get_probs_atlas('source')['0'].detach() # [B, A]
                    if self.cfg.model.losses.kl_c_domain_diff.queue_mode == 'append':
                        queue_qc_source = self.queue_qc_source
                        queue_qc_source = torch.cat([queue_qc_source, probs])
                        if queue_qc_source.shape[0] > queue_size:
                            queue_qc_source = queue_qc_source[-queue_size:]
                        self.register_buffer('queue_qc_source', queue_qc_source)
                    elif self.cfg.model.losses.kl_c_domain_diff.queue_mode == 'replace_same_slice':
                        patient_slice2qc_source = getattr(self, 'patient_slice2qc_source', BufferDict())
                        for i, patient_slice in enumerate(data['patient_slice_source']):
                            patient_slice2qc_source.update(patient_slice, probs[i]) # [A]
                        self.patient_slice2qc_source = patient_slice2qc_source
                        self.register_buffer('queue_qc_source', torch.stack(patient_slice2qc_source.get_values()))
                    else:
                        raise ValueError(f'unknown queue_mode: {self.cfg.model.losses.kl_c_domain_diff.queue_mode}')
                
                if self.cfg.model.ws_loss.pos_slice_soft_contrastive > 0.: 
                    if self.cfg.model.losses.kl_c_domain_diff.queue_mode == 'append':
                        poss_source = data[f'pos_slice_source'] # [B]
                        queue_poss_source = getattr(self, 'queue_poss_source', torch.empty(0, device=poss_source.device))
                        self.queue_poss_source = torch.cat([queue_poss_source, poss_source])
                        if self.queue_poss_source.shape[0] > queue_size:
                            self.queue_poss_source = self.queue_poss_source[-queue_size:]
                    elif self.cfg.model.losses.kl_c_domain_diff.queue_mode == 'replace_same_slice':
                        patient_slice2poss_source = getattr(self, 'patient_slice2poss_source', {})
                        for i, patient_slice in enumerate(data['patient_slice_source']):
                            patient_slice2poss_source[patient_slice] = data['pos_slice_source'][i]
                        self.patient_slice2poss_source = patient_slice2poss_source
                        self.queue_poss_source = torch.tensor(list(patient_slice2poss_source.values()), device=poss_source.device)
                    else:
                        raise ValueError(f'unknown queue_mode: {self.cfg.model.losses.kl_c_domain_diff.queue_mode}')

        if self.training:
            out = self.net(data)
        else:
            with torch.no_grad():
                img = data.get('img_source', data.get('img_target'))
                shape_atlas = self.net.atlas_lv0.shape[2:] # [...]
                if img.shape[2:] == shape_atlas:
                    self.test_by_sliding_win = False
                    out = self.net(data)
                else:
                    self.test_by_sliding_win = True
                    pred = sliding_window_inference(img, roi_size=shape_atlas, sw_batch_size=4,
                                                    predictor=self.net, mode='gaussian')
                    if 'img_source' in data:
                        out = {'logits_source': pred}
                        self.recorder['logits_seg_ori_source'] = pred
                        data['img_source'] = img
                    elif 'img_target' in data:
                        out = {'logits_target': pred}
                        self.recorder['logits_seg_ori_target'] = pred
                        data['img_target'] = img

                    else:
                        raise ValueError(data.keys())

        return out

    def loss_dice(self, data, output):
        input = self.net.get_logits_seg('source') # [BxS, K, ...]
        target = data['seg_source'] # [B, 1, ...], values in [0, ..., n_classes-1]
        target = seg2onehot(target, n_classes=input.shape[1], dim_class=1) # [B, K, ...]
        target = repeat(target, 'B K ... -> (B S) K ...', S=int(input.shape[0] // target.shape[0])) # [BxS, K, ...]
        loss = self.calc_loss_dice(input, target) # [BxS, K, ...]
        assert len(loss.shape) == len(target.shape)
        loss = torch.mean(loss, dim=list(range(1, loss.dim()))) # [BxS]
        if self.cfg.net.atlas.soft_idx:
            assert loss.shape[0] == target.shape[0]
            loss = torch.mean(loss)
        else:
            if self.cfg.net.atlas.n_samples == 'deterministic':
                loss = rearrange(loss, '(B S) -> B S', B=data['seg_source'].shape[0]) # [B, S]
                assert not self.cfg.net.atlas.qc.calc.levelwise
                probs_a = self.net.get_probs_atlas('source')['0'] # [B, S=A]
                loss = torch.sum(loss * probs_a, dim=1) # [B]
                loss = torch.mean(loss)
            else:
                assert isinstance(self.cfg.net.atlas.n_samples, int)
                loss = torch.mean(loss)
        return loss

    def loss_focal(self, data, output):
        input = self.net.get_logits_seg('source') # [BxS, K, ...]
        target = data['seg_source'] # [B, 1, ...], values in [0, ..., n_classes-1]
        target = target.long() # [B, 1, ...]
        target = repeat(target, 'B ... -> (B S) ...', S=int(input.shape[0] // target.shape[0])) # [BxS, ...]

        # Calculate class weights if needed
        ws = None
        if self.cfg.model.losses.focal.ws_class == "1/freq":
            # Calculate class frequencies from the current batch
            class_counts = torch.bincount(target.flatten(), minlength=input.shape[1])
            class_freqs = class_counts / class_counts.sum()
            ws = 1.0 / (class_freqs + 1e-6)  # Add small epsilon to avoid division by zero
            ws = ws / ws.sum()  # Normalize weights
        elif self.cfg.model.losses.focal.ws_class is not None:
            ws = torch.tensor(self.cfg.model.losses.focal.ws_class)
            assert ws[0] == 1, 'background class should always have weight 1; ths weights are pre-normalized'
            ws = ws / ws.sum()

        # Initialize Focal Loss with current weights
        calc_loss_focal = monai.losses.FocalLoss(
            include_background=True,
            to_onehot_y=True,
            gamma=self.cfg.model.losses.focal.gamma,
            weight=ws,
            reduction='none'
        )

        loss = calc_loss_focal(input, target) # [BxS, ...]
        assert len(loss.shape) == len(target.shape)
        loss = torch.mean(loss, dim=list(range(1, loss.dim()))) # [BxS]
        if self.cfg.net.atlas.soft_idx:
            assert loss.shape[0] == target.shape[0]
            loss = torch.mean(loss)
        else:
            if self.cfg.net.atlas.n_samples == 'deterministic':
                loss = rearrange(loss, '(B S) -> B S', B=data['seg_source'].shape[0]) # [B, S]
                assert not self.cfg.net.atlas.qc.calc.levelwise
                probs_a = self.net.get_probs_atlas('source')['0'] # [B, S=A]
                loss = torch.sum(loss * probs_a, dim=1) # [B]
                loss = torch.mean(loss)
            else:
                assert isinstance(self.cfg.net.atlas.n_samples, int)
                loss = torch.mean(loss)
        return loss

    def loss_ce(self, data, output):
        input = self.net.get_logits_seg('source') # [BxS, K, ...]
        target = data['seg_source'] # [B, 1, ...], values in [0, ..., n_classes-1]
        target = target.squeeze(1).long() # [B, ...]
        target = repeat(target, 'B ... -> (B S) ...', S=int(input.shape[0] // target.shape[0])) # [BxS, ...]
        loss = self.calc_loss_ce(input, target) # [BxS, ...]
        assert len(loss.shape) == len(target.shape)
        loss = torch.mean(loss, dim=list(range(1, loss.dim()))) # [BxS]
        if self.cfg.net.atlas.soft_idx:
            assert loss.shape[0] == target.shape[0]
            loss = torch.mean(loss)
        else:
            if self.cfg.net.atlas.n_samples == 'deterministic':
                loss = rearrange(loss, '(B S) -> B S', B=data['seg_source'].shape[0]) # [B, S]
                assert not self.cfg.net.atlas.qc.calc.levelwise
                probs_a = self.net.get_probs_atlas('source')['0'] # [B, S=A]
                loss = torch.sum(loss * probs_a, dim=1) # [B]
                loss = torch.mean(loss)
            else:
                assert isinstance(self.cfg.net.atlas.n_samples, int)
                loss = torch.mean(loss)
        return loss

    def loss_adv_enc_feat(self, data, output):
        lv2out_source = self.net.get_output_discriminator_enc_feat('source') # [B, 1]
        lv2out_target = self.net.get_output_discriminator_enc_feat('target') # [B, 1]
        losses = []
        for lv in lv2out_source.keys():
            out_source = lv2out_source[lv]
            out_target = lv2out_target[lv]
            domain_labels_source = torch.ones(out_source.shape[0], 1, device=out_source.device)
            domain_labels_target = torch.zeros(out_target.shape[0], 1, device=out_target.device)
            domain_preds = torch.cat([out_source, out_target]) # [2B, 1, ...]
            domain_labels = torch.cat([domain_labels_source, domain_labels_target]) # [2B, 1, ...]
            loss = nn.BCELoss()(domain_preds, domain_labels)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        return loss

    def loss_adv_seg(self, data, output):
        out_source = self.net.get_output_discriminator_seg('source') # [B, 1, ...]
        out_target = self.net.get_output_discriminator_seg('target') # [B, 1, ...]
        domain_labels_source = torch.ones(*out_source.shape, device=out_source.device)
        domain_labels_target = torch.zeros(*out_target.shape, device=out_target.device)
        domain_preds = torch.cat([out_source, out_target]) # [2B, 1]
        domain_labels = torch.cat([domain_labels_source, domain_labels_target]) # [2B, 1]
        loss = nn.BCEWithLogitsLoss()(domain_preds, domain_labels)
        return loss

    def loss_kl_c_domain_diff(self, data, output):
        """
        Minimize difference of probabilities of atlas assignments between the source and target domains.
        """
        assert self.cfg.model.losses.kl_c_domain_diff.method == 'wasserstein'
        assert not self.cfg.net.atlas.qc.calc.levelwise
        queue_size = self.cfg.model.losses.kl_c_domain_diff.queue_size
        if queue_size > 0:
            probs_source = self.queue_qc_source
        else:
            probs_source = self.net.get_probs_atlas('source')['0'] # [B, A]
        if self.cfg.model.losses.kl_c_domain_diff.qc_target_from == 'before_post_process':
            probs_target = self.net.get_probs_atlas_before_post_process('target')['0'] # [B, A]
        elif self.cfg.model.losses.kl_c_domain_diff.qc_target_from == 'after_post_process':
            probs_target = self.net.get_probs_atlas('target')['0'] # [B, A]
        else:
            raise ValueError
        if self.cfg.model.losses.kl_c_domain_diff.wasserstein.logratio_transform:
            # map probabilities to log-ratio space
            # see https://stats.stackexchange.com/questions/33685/what-are-some-distributions-over-the-probability-simplex
            log_source = torch.log(torch.clamp_min(probs_source, 1e-6)) # [B, A]
            log_target = torch.log(torch.clamp_min(probs_target, 1e-6)) # [B, A]
            samples_source = log_source[:, :-1] - torch.mean(log_source, dim=-1, keepdim=True) # [B, A-1]
            samples_target = log_target[:, :-1] - torch.mean(log_target, dim=-1, keepdim=True) # [B, A-1]
        else:
            samples_source = probs_source
            samples_target = probs_target

        loss = self.calc_loss_kl_c_domain_diff(samples_source, samples_target)
        return loss

    def loss_expert_usage(self, data, output):
        """
        Minimize the KL divergence between the atlas assignment probabilities and a uniform distribution.
        """
        assert not self.cfg.net.atlas.qc.calc.levelwise
        tau = self.cfg.model.losses.expert_usage.tau
        losses = []
        for domain in self.cfg.var.domains:
            if self.cfg.model.losses.c_dispersity.ws_domain[0] == 0.:
                continue
            probs = self.net.get_probs_atlas(domain)['0'] # [B, A]
            probs = torch.mean(probs, dim=0) # [A]
            loss = torch.sum(torch.max(torch.zeros_like(probs), tau - probs))
            losses.append(loss)
        if len(losses) == 1:
            return losses[0]
        else:
            assert len(losses) == 2 == len(self.cfg.var.domains)
            loss = self.calc_final_loss_by_weights_domain(torch.stack(losses), self.cfg.var.domains,
                                                          self.cfg.model.losses.expert_usage.ws_domain)
        return loss


    def loss_c_dispersity(self, data, output):
        losses = []
        for domain in self.cfg.var.domains:
            if self.cfg.model.losses.c_dispersity.ws_domain[0] == 0.:
                continue
            probs = self.net.get_probs_atlas(domain)['0'] # [B, A]
            if self.cfg.model.losses.c_dispersity.mode == 'pairwise_fisher_rao':
                idxs = torch.tensor(list(combinations(range(probs.shape[0]), 2)))
                assert len(idxs) == probs.shape[0] * (probs.shape[0] - 1) // 2
                probs1 = probs[idxs[:, 0]] # [N, A]
                probs2 = probs[idxs[:, 1]] # [N, A]
                distances = fisher_rao_distance(probs1[:, None], probs2[:, None]) # [N, 1, 1]
                assert distances.shape[1] == distances.shape[2] == 1
                distance_mean = torch.mean(distances)
                loss = -distance_mean
                losses.append(loss)
            elif self.cfg.model.losses.c_dispersity.mode == 'det_euclidean':
                probs_mean = torch.mean(probs, dim=0, keepdim=True)
                probs_normalized = probs - probs_mean # [B, A]
                cov = torch.sum(probs_normalized[:, :, None] * probs_normalized[:, None, :], dim=0) # [A, A]
                cov = cov / probs.shape[0]
                cov = cov + torch.eye(cov.shape[0], device=cov.device) * 1e-6
                loss = -torch.logdet(cov)
                losses.append(loss)
            else:
                raise NotImplementedError
        if len(losses) == 1:
            return losses[0]
        else:
            assert len(losses) == 2 == len(self.cfg.var.domains)
            loss = self.calc_final_loss_by_weights_domain(torch.stack(losses), self.cfg.var.domains,
                                                          self.cfg.model.losses.c_dispersity.ws_domain)
        return loss

    def loss_qc_diverse_repulsive(self, data, output):
        """
        Minimize the KL divergence between the atlas assignment probabilities and a uniform distribution.
        """
        assert not self.cfg.net.atlas.qc.calc.levelwise
        losses = []
        for domain in self.cfg.var.domains:
            if domain == 'source':
                if self.cfg.model.losses.qc_diverse_repulsive.ws_domain[0] == 0.:
                    continue
                probs = self.net.get_probs_atlas('source')['0'] # [B, A]
            elif domain == 'target':
                if self.cfg.model.losses.qc_diverse_repulsive.ws_domain[1] == 0.:
                    continue
                probs = self.net.get_probs_atlas('target')['0'] # [B, A]
            else:
                raise ValueError(f'unknown domain: {domain}')
            idxs = torch.tensor(list(combinations(range(probs.shape[0]), 2)))
            probs1 = probs[idxs[:, 0]] # [N, A]
            probs2 = probs[idxs[:, 1]] # [N, A]

            if self.cfg.model.losses.qc_diverse_repulsive.distance == 'fisher_rao':
                dists = fisher_rao_distance(probs1[:, None], probs2[:, None]) # [N, 1, 1]
                assert dists.shape[1] == dists.shape[2] == 1
                dists = dists[:, 0, 0] # [N]
            else:
                raise ValueError(f'unknown distance: {self.cfg.model.losses.qc_diverse_repulsive.distance}')

            if self.cfg.model.losses.qc_diverse_repulsive.kernel.name == 'gaussian':
                sigma = self.cfg.model.losses.qc_diverse_repulsive.kernel.gaussian.sigma
                loss = torch.mean(torch.exp(-dists**2 / (2 * sigma**2)))
            elif self.cfg.model.losses.qc_diverse_repulsive.kernel.name == 'coulomb':
                alpha = self.cfg.model.losses.qc_diverse_repulsive.kernel.coulomb.alpha
                loss = torch.mean(1 / (dists+1e-6)**alpha)
            elif self.cfg.model.losses.qc_diverse_repulsive.kernel.name == 'laplacian':
                sigma = self.cfg.model.losses.qc_diverse_repulsive.kernel.laplacian.sigma
                loss = torch.mean(torch.exp(-dists / sigma))
            else:
                raise ValueError(f'unknown kernel: {self.cfg.model.losses.qc_diverse_repulsive.kernel.name}')
            losses.append(loss)

        if len(losses) == 1:
            loss = losses[0]
        else:
            assert len(losses) == 2 == len(self.cfg.var.domains)
            loss = self.calc_final_loss_by_weights_domain(torch.stack(losses), self.cfg.var.domains,
                                                          self.cfg.model.losses.qc_diverse_repulsive.ws_domain)
        return loss


    def loss_v_volume_change_domain_diff(self, data, output):
        disps_source = self.net.get_disp_inv_final('source') # [BxS, D, ...]
        disps_target = self.net.get_disp_inv_final('target') # [BxS, D, ...]
        calc_jac = JacobianDeterminant(dimension=self.cfg.dataset.dim)
        det_jac_source = calc_jac(disps_source) # [BxS, ...]
        det_jac_target = calc_jac(disps_target) # [BxS, ...]

        seg_source = self.net.get_logits_seg_reg('source') # [BxS, K, ...]
        seg_target = self.net.get_logits_seg_reg('target') # [BxS, K, ...]
        if self.cfg.dataset.dim == 2:
            seg_source = seg_source[:, :, 1:-1, 1:-1]
            seg_target = seg_target[:, :, 1:-1, 1:-1]
        elif self.cfg.dataset.dim == 3:
            seg_source = seg_source[:, :, 1:-1, 1:-1, 1:-1]
            seg_target = seg_target[:, :, 1:-1, 1:-1, 1:-1]
        else:
            raise ValueError(f'unknown dim: {self.cfg.dataset.dim}')
        seg_source = torch.argmax(seg_source, dim=1, keepdim=True) # [BxS, 1, ...]
        seg_target = torch.argmax(seg_target, dim=1, keepdim=True) # [BxS, 1, ...]
        seg_source = seg2onehot(seg_source, n_classes=self.cfg.dataset.n_classes, dim_class=1) # [BxS, K, ...]
        seg_target = seg2onehot(seg_target, n_classes=self.cfg.dataset.n_classes, dim_class=1) # [BxS, K, ...]
        seg_source = seg_source[:, 1:] # [BxS, K, ...]
        seg_target = seg_target[:, 1:] # [BxS, K, ...]

        vol_change_source = torch.sum(seg_source * det_jac_source[:, None],
                                      dim=list(range(2, seg_source.dim()))) # [BxS, K]
        vol_change_target = torch.sum(seg_target * det_jac_target[:, None],
                                      dim=list(range(2, seg_target.dim()))) # [BxS, K]
        vol_source = torch.sum(seg_source, dim=list(range(2, seg_source.dim()))) # [BxS, K]
        vol_target = torch.sum(seg_target, dim=list(range(2, seg_target.dim()))) # [BxS, K]
        vol_change_source = vol_change_source / (vol_source+1e-8) # [BxS, K]
        vol_change_target = vol_change_target / (vol_target+1e-8) # [BxS, K]
        vol_change_source = torch.clamp_min(vol_change_source, 1e-8) # [BxS, K]
        vol_change_target = torch.clamp_min(vol_change_target, 1e-8) # [BxS, K]

        # pairwise segmentation similarity
        dims = list(range(3, seg_source.dim() + 1))
        numerator = 2 * torch.sum(seg_source[:, None] * seg_target[None], dim=dims) # [BxS, BxS, K]
        denominator = torch.sum(seg_source[:, None] + seg_target[None], dim=dims) # [BxS, BxS, K]
        sims = (numerator+1e-8) / (denominator+1e-8) # [BxS, BxS, K]
        mask = sims > 0.5
        sims = sims[mask] # [N]
        # pairwise volume change difference
        diff_vol_change = (torch.log(vol_change_source[:, None]) - torch.log(vol_change_target[None]))[mask].abs() # [N]

        loss = torch.mean(diff_vol_change * (sims - 0.5) * 2)
        return loss

    def loss_quantize_probs_a(self, data, output):
        assert not self.cfg.net.atlas.qc.calc.levelwise
        assert self.cfg.net.atlas.qc.calc.quantize_target
        dists = self.cfg.var.obj_model.recorder['probs_closest_dists'] # [B]
        if dists is None:
            return 0.
        return torch.mean(dists)


    def loss_diff_qc_to_mask_sim(self, data, output):
        losses = []
        for domain in self.cfg.var.domains:
            if domain == 'source':
                if self.cfg.model.losses.diff_qc_to_mask_sim.ws_domain[0] == 0.:
                    continue
                masks = self.net.get_segs_gt_reg('source') # [B, 1, ...]
                masks = seg2onehot(masks, n_classes=self.cfg.dataset.n_classes, dim_class=1) # [B, K, ...]
            elif domain == 'target':
                if self.cfg.model.losses.diff_qc_to_mask_sim.ws_domain[1] == 0.:
                    continue
                masks = self.net.get_logits_seg_reg('target') # [B, K, ...]
                masks = torch.argmax(masks, dim=1, keepdim=True) # [B, 1, ...]
                masks = seg2onehot(masks, n_classes=self.cfg.dataset.n_classes, dim_class=1) # [B, K, ...]
            else:
                raise ValueError(f'unknown domain: {domain}')
            probs_a = self.net.get_probs_atlas(domain)['0'] # [B, A]
            idxs = torch.tensor(list(combinations(range(masks.shape[0]), 2))) # [N, 2]
            masks1 = masks[idxs[:, 0]] # [N, K, ...]
            masks2 = masks[idxs[:, 1]] # [N, K, ...]
            probs1 = probs_a[idxs[:, 0]] # [N, A]
            probs2 = probs_a[idxs[:, 1]] # [N, A]

            # calculate similarity between masks
            dims = list(range(2, masks.dim()))
            if self.cfg.model.losses.diff_qc_to_mask_sim.sim_mask.metric == 'dice':
                numerator = 2 * torch.sum(masks1 * masks2, dim=dims) # [N, K]
                denominator = torch.sum(masks1 + masks2, dim=dims) # [N, K]
                sim_masks = (numerator+1e-8) / (denominator+1e-8) # [N, K]
                sim_masks = torch.mean(sim_masks, dim=1) # [N]
            elif self.cfg.model.losses.diff_qc_to_mask_sim.sim_mask.metric == 'pixel_acc':
                masks_bg = masks1[:, 0:1] * masks2[:, 0:1] # [N, 1, ...]
                masks_fg = 1 - masks_bg # [N, 1, ...]
                masks1 = torch.argmax(masks1, dim=1, keepdim=True) # [N, 1, ...]
                masks2 = torch.argmax(masks2, dim=1, keepdim=True) # [N, 1, ...]
                sim_masks = (torch.mean((masks1 == masks2).float() * masks_fg, dim=list(range(1, masks_fg.dim()))) + 1e-6) / (torch.mean(masks_fg, dim=list(range(1, masks_fg.dim()))) + 1e-6) # [N]
            else:
                raise NotImplementedError(f'unknown metric: {self.cfg.model.losses.diff_qc_to_mask_sim.sim_mask.metric}')


            # calculate similarity between atlas probabilities
            if self.cfg.model.losses.diff_qc_to_mask_sim.distance_qc == 'hellinger':
                sims_prob = 1 - hellinger_distance(probs1, probs2) # [N]
            elif self.cfg.model.losses.diff_qc_to_mask_sim.distance_qc == 'fisher_rao':
                sims_prob = fisher_rao_distance(probs1[:, None], probs2[:, None]) # [N, 1, 1]
                assert sims_prob.shape[1] == sims_prob.shape[2] == 1
                sims_prob = sims_prob[:, 0, 0] # [N]
                sims_prob = 1 - sims_prob / torch.pi
            else:
                raise ValueError(f'unknown distance_qc: {self.cfg.model.losses.diff_qc_to_mask_sim.distance_qc}')

            loss = torch.mean((sim_masks - sims_prob)**2)
            losses.append(loss)

        if len(losses) == 1:
            return losses[0]
        else:
            assert len(losses) == 2 == len(self.cfg.var.domains)
            loss = self.calc_final_loss_by_weights_domain(torch.stack(losses), self.cfg.var.domains,
                                                          self.cfg.model.losses.diff_qc_to_mask_sim.ws_domain)
        return loss

    def calc_final_loss_by_weights_domain(self, loss, domains, ws_domain):
        """
        loss: [C], C <= 2
        domains: [C], C <= 2, each value is a str ('source', 'target')
        ws_domain: [2], source and target weights for the loss
        """
        assert len(loss) == len(domains) <= 2
        assert len(ws_domain) == 2
        ws = []
        for domain in domains:
            if domain == 'source':
                idx = 0
            elif domain == 'target':
                idx = 1
            else:
                raise ValueError(f'unknown domain: {domain}')
            ws.append(ws_domain[idx])
        ws = torch.tensor(ws, device=loss.device)
        final_loss = torch.sum(loss * ws) / torch.sum(ws)
        return final_loss

    def loss_recon(self, data, output):
        def get_neg_log_prob(img, rec, scale, dims_mean, seg=None, bg_zero=False, include_bg=True):
            # img: [B, 1, 1, ...]
            # rec: [B, S, C, ...]
            assert rec.shape[2] in [1, 2]
            if bg_zero or not include_bg:
                assert seg is not None

            if seg is not None:
                seg = seg > 0
            if bg_zero:
                img = img * seg
            l1 = torch.abs(img - rec[:, :, :1]) # [B, S, 1, ...]
            if not include_bg:
                l1 = l1 * seg # [B, S, 1, ...]
                n_pixels_fg = torch.sum(seg, dim=dims_mean)

            if isinstance(scale, (float, int)):
                assert scale > 0.
                if include_bg:
                    l1 = torch.mean(l1, dim=dims_mean) / scale
                else:
                    l1 = torch.sum(l1, dim=dims_mean) / n_pixels_fg / scale
            else:
                assert scale == 'learned'
                scale = rec[:, :, 1:] # [B, S, 1, ...]
                scale = F.softplus(scale, beta=1).clamp(min=1e-8) # [B, S, 1, ...]
                l1 = l1/scale + torch.log(scale) # [B, S, 1, ...]
                if include_bg:
                    l1 = torch.mean(l1, dim=dims_mean)
                else:
                    l1 = torch.sum(l1, dim=dims_mean) / n_pixels_fg
            return l1

        l1s = []
        for domain in self.cfg.var.domains:
            scale = self.cfg.model.losses.recon.scale
            img = data[f'img_{domain}'] # [B, n_adj*2+1, ...]
            img = img[:, img.shape[1]//2:img.shape[1]//2+1] # [B, 1, ...]
            rec = self.net.get_imgs_rec_ori(domain) # [BxS, C, ...]
            S = int(rec.shape[0] / img.shape[0])
            rec = rearrange(rec, '(B S) ... -> B S ...', S=S) # [B, S, C, ...]
            img = img[:, None] # [B, 1, 1, ...]

            i_domain = ['source', 'target'].index(domain)
            if ((not self.cfg.model.losses.recon.include_bgs[i_domain])
                    or self.cfg.model.losses.recon.bg_zeros[i_domain]):
                if domain == 'source':
                    seg = data['seg_source'][:, None] # [B, 1, 1, ...]
                elif domain == 'target':
                    seg = self.net.get_logits_seg('target') # [B, K, ...]
                    seg = torch.argmax(seg, dim=1, keepdim=True)[:, None] # [B, 1, 1, ...]
                else:
                    raise ValueError(f'unknown domain: {domain}')
            else:
                seg = None

            if self.cfg.net.atlas.soft_idx:
                assert img.shape[1] == rec.shape[1] == 1 # S==1
                l1 = get_neg_log_prob(img, rec, scale, dims_mean=list(range(img.dim())), seg=seg,
                                      bg_zero=self.cfg.model.losses.recon.bg_zeros[i_domain],
                                      include_bg=self.cfg.model.losses.recon.include_bgs[i_domain])
            else:
                if self.cfg.net.atlas.n_samples == 'deterministic':
                    assert not self.cfg.net.atlas.qc.calc.levelwise
                    prob_a = self.net.get_probs_atlas(domain)['0'] # [B, S=A]
                    l1 = get_neg_log_prob(img, rec, scale, dims_mean=list(range(2, img.dim())), seg=seg,
                                          bg_zero=self.cfg.model.losses.recon.bg_zeros[i_domain],
                                          include_bg=self.cfg.model.losses.recon.include_bgs[i_domain]) # [B, S]
                    l1 = torch.sum(l1 * prob_a, dim=1) # [B]
                    l1 = torch.mean(l1)
                else:
                    l1 = get_neg_log_prob(img, rec, scale, dims_mean=list(range(img.dim())), seg=seg,
                                          bg_zero=self.cfg.model.losses.recon.bg_zeros[i_domain],
                                          include_bg=self.cfg.model.losses.recon.include_bgs[i_domain])

            l1s.append(l1)
        l1 = self.calc_final_loss_by_weights_domain(torch.stack(l1s), self.cfg.var.domains,
                                                    self.cfg.model.losses.recon.ws_domain)
        return l1

    def loss_recon_ncc(self, data, output):
        l1s = []
        for domain in self.cfg.var.domains:
            img = data[f'img_{domain}'] # [B, 1, ...]
            rec = self.net.get_imgs_rec_ori(domain) # [BxS, 1 or 2, ...]
            rec = rec[:, :1]
            S = int(rec.shape[0] / img.shape[0])
            rec = rearrange(rec, '(B S) ... -> B S ...', S=S) # [B, S, 1, ...]
            img = img[:, None] # [B, 1, 1, ...]
            img = rearrange(img, 'B S ... -> (B S) ...') # [BxS, C, ...]
            rec = rearrange(rec, 'B S ... -> (B S) ...') # [BxS, C, ...]
            if self.cfg.net.atlas.soft_idx:
                assert img.shape[1] == rec.shape[1] == 1
                l1 = self.calc_loss_recon_ncc.loss(img, rec)
            else:
                if self.cfg.net.atlas.n_samples == 'deterministic':
                    assert not self.cfg.net.atlas.qc.calc.levelwise
                    prob_a = self.net.get_probs_atlas(domain)['0'] # [B, S=A]
                    l1 = self.calc_loss_recon_ncc.loss(img, rec, mean_dims=list(range(1, img.dim()))) # [BxS]
                    l1 = rearrange(l1, '(B S) -> B S', S=S) # [B, S]
                    l1 = torch.sum(l1 * prob_a, dim=1) # [B]
                    l1 = torch.mean(l1)
                else:
                    l1 = self.calc_loss_recon_ncc.loss(img, rec)
            l1s.append(l1)
        l1 = self.calc_final_loss_by_weights_domain(torch.stack(l1s), self.cfg.var.domains,
                                                    self.cfg.model.losses.recon.ws_domain)
        return l1

    def loss_kl_v(self, data, output):
        losses_domain = []
        for domain in self.cfg.var.domains:
            loss_domain = 0.
            lv2name2param, _, _ = self.net.get_disps_levelwise(domain) # [BxS, D, ...]
            for l, name2param in lv2name2param.items():
                if self.cfg.model.losses.kl_v.remove_mean:
                    mean = name2param['mean'] # [BxS, D, ...]
                    mean = mean - torch.mean(mean, dim=list(range(2, mean.ndim)), keepdim=True)
                else:
                    mean = None
                kl = self.calc_loss_kl_v(name2param, level=int(l), mean=mean) # [BxS]
                if self.cfg.net.atlas.soft_idx:
                    assert kl.shape[0] == data[f'img_{domain}'].shape[0]
                    kl = torch.mean(kl)
                else:
                    if self.cfg.net.atlas.n_samples == 'deterministic':
                        assert not self.cfg.net.atlas.qc.calc.levelwise
                        probs_a = self.net.get_probs_atlas(domain)['0'] # [B, S=A]
                        kl = rearrange(kl, '(B S) -> B S', B=probs_a.shape[0]) # [B, S]
                        kl = torch.sum(kl * probs_a, dim=1) # [B]
                        kl = torch.mean(kl)
                    else:
                        kl = torch.mean(kl)
                loss_domain += kl
            losses_domain.append(loss_domain)
        loss = self.calc_final_loss_by_weights_domain(torch.stack(losses_domain), self.cfg.var.domains,
                                                      self.cfg.model.losses.kl_v.ws_domain)
        assert len(lv2name2param) == len(self.cfg.net.reg.levels)
        loss = loss / len(self.cfg.net.reg.levels)
        return loss

    def loss_v_domain_diff_region_wise(self, data, output):
        disp_source = self.net.get_disp_final('source') # [BxS, D, ...]
        disp_target = self.net.get_disp_final('target') # [BxS, D, ...]
        seg_source = self.net.get_logits_seg('source') # [BxS, K, ...]
        seg_target = self.net.get_logits_seg('target') # [BxS, K, ...]
        K = seg_source.shape[1]
        assert K == self.cfg.dataset.n_classes
        seg_source = torch.argmax(seg_source, dim=1) # [BxS, ...]
        seg_target = torch.argmax(seg_target, dim=1) # [BxS, ...]
        losses = []
        for k in range(1, K):
            fg_source = seg_source == k # [BxS, ...]
            fg_target = seg_target == k # [BxS, ...]
            disp_source_fg = rearrange(disp_source, 'B D ... -> B ... D')[fg_source] # [N, D]
            disp_target_fg = rearrange(disp_target, 'B D ... -> B ... D')[fg_target] # [N, D]
            if disp_source_fg.shape[0] == 0 or disp_target_fg.shape[0] == 0:
                continue
            if self.cfg.model.losses.v_domain_diff_region_wise.subtract_mean:
                disp_source_fg = disp_source_fg - torch.mean(disp_source_fg, dim=0, keepdim=True)
                disp_target_fg = disp_target_fg - torch.mean(disp_target_fg, dim=0, keepdim=True)
            print(
                f'Domain diff of disps: k: {k}, n_source: {disp_source_fg.shape[0]}, n_target: {disp_target_fg.shape[0]}'
            )
            loss = self.calc_loss_v_domain_diff_region_wise(disp_source_fg, disp_target_fg)
            losses.append(loss)
        loss = sum(losses) / len(losses)
        return loss

    def loss_kl_z(self, data, output):
        losses_domain = []
        for domain in self.cfg.var.domains:
            loss_domain = 0.
            lv2name2param_pzc = self.net.get_lv2name2param_pzc(domain) # [BxS, C/2, ...]
            lv2name2param_qzxv = self.net.get_lv2name2param_qzxv(domain) # [BxS, C/2, ...]
            for lv in lv2name2param_pzc.keys():
                name2param_pzc = lv2name2param_pzc[lv] # [BxS, C/2, ...]
                name2param_qzxv = lv2name2param_qzxv[lv] # [BxS, C/2, ...]
                try:
                    dist_pzc = tcdist.Normal(name2param_pzc['mean'], name2param_pzc['std']) # [BxS, C/2, ...]
                    dist_qzxv = tcdist.Normal(name2param_qzxv['mean'], name2param_qzxv['std']) # [BxS, C/2, ...]
                except Exception as e:
                    print(type(e), e)
                    quit()
                dist_pzc = tcdist.Independent(dist_pzc, self.cfg.dataset.dim + 1)
                dist_qzxv = tcdist.Independent(dist_qzxv, self.cfg.dataset.dim + 1)
                kl = tcdist.kl_divergence(dist_qzxv, dist_pzc) # [BxS]
                if self.cfg.net.atlas.soft_idx:
                    assert kl.shape[0] == data[f'img_{domain}'].shape[0]
                    kl = torch.mean(kl)
                else:
                    if self.cfg.net.atlas.n_samples == 'deterministic':
                        assert not self.cfg.net.atlas.qc.calc.levelwise
                        probs_a = self.net.get_probs_atlas(domain)['0'] # [B, S=A]
                        kl = rearrange(kl, '(B S) -> B S', B=probs_a.shape[0]) # [B, S]
                        kl = torch.sum(kl * probs_a, dim=1) # [B]
                        kl = torch.mean(kl)
                    else:
                        kl = torch.mean(kl)
                kl /= np.prod(data[f'img_{domain}'].shape[2:])
                loss_domain += kl
            losses_domain.append(loss_domain)
        loss = self.calc_final_loss_by_weights_domain(torch.stack(losses_domain), self.cfg.var.domains,
                                                      self.cfg.model.losses.kl_z.ws_domain)
        assert len(lv2name2param_pzc) == self.cfg.net.n_levels
        loss = loss / self.cfg.net.n_levels
        return loss

    def loss_kl_qc_to_standard_dirichlet(self, data, output):
        assert self.cfg.net.atlas.qc.mode.name == 'dirichlet'
        assert not self.cfg.net.atlas.qc.calc.levelwise
        alphas = self.net.get_params_dist_prob_atlas('source')['0'] # [B, A]
        dist1 = tcdist.Dirichlet(alphas)
        dist2 = tcdist.Dirichlet(torch.ones_like(alphas))
        kl = tcdist.kl_divergence(dist1, dist2) # [B]
        kl = torch.mean(kl)
        return kl

    def loss_kl_mix_dirichlet_neg_ent(self, data, output):
        assert self.cfg.net.atlas.qc.mode.name == 'dirichlet'
        assert not self.cfg.net.atlas.qc.calc.levelwise
        losses = []
        for domain in self.cfg.var.domains:
            log_prob = self.net.get_log_prob_samples_mix_qc_same_domain(domain) # [M]
            neg_ent = torch.mean(log_prob)
            losses.append(neg_ent)
        loss = self.calc_final_loss_by_weights_domain(torch.stack(losses), self.cfg.var.domains,
                                                        self.cfg.model.losses.kl_mix_dirichlet_neg_ent.ws_domain)
        return loss

    def loss_kl_dirichlet_domain_diff(self, data, output):
        assert self.cfg.net.atlas.qc.mode.name == 'dirichlet'
        assert not self.cfg.net.atlas.qc.calc.levelwise
        assert self.cfg.model.losses.kl_dirichlet_domain_diff.estimation_mode == 'monte_carlo'
        log_prob1 = self.net.get_log_prob_samples_mix_qc_same_domain('source') # [M]
        log_prob2 = self.net.get_log_prob_samples_mix_qc_diff_domain('source') # [M]
        kl_s2t = torch.mean(log_prob1 - log_prob2)

        log_prob1 = self.net.get_log_prob_samples_mix_qc_same_domain('target') # [M]
        log_prob2 = self.net.get_log_prob_samples_mix_qc_diff_domain('target') # [M]
        kl_t2s = torch.mean(log_prob2 - log_prob1)

        loss = (kl_s2t + kl_t2s) / 2
        return loss

    def loss_kl_atlas(self, data, output):
        kls = []
        for lv in range(self.cfg.net.n_levels):
            atlas = getattr(self.net, f'atlas_lv{lv}') # [A, C, ...]
            mean, logits = atlas.chunk(2, dim=1) # [A, C/2, ...]
            var = self.net.poe.logits2var(logits) # [A, C/2, ...]
            std = torch.sqrt(var) # [A, C/2, ...]
            dist1 = tcdist.Normal(mean, std) # [A, C/2, ...]
            dist2 = tcdist.Normal(torch.zeros_like(mean), torch.ones_like(std)) # [A, C/2, ...]
            kl = tcdist.kl_divergence(dist1, dist2) # [A, C/2, ...]
            kl = torch.mean(kl)
            kls.append(kl)
        loss = sum(kls) / len(kls)
        return loss

    def loss_kl_atlas_to_geomean(self, data, output):
        kls = []
        for lv in range(self.cfg.net.n_levels):
            atlas = getattr(self.net, f'atlas_lv{lv}') # [A, C, ...]
            mean, logits = atlas.chunk(2, dim=1) # [A, C/2, ...]
            var = self.net.poe.logits2var(logits) # [A, C/2, ...]
            std = torch.sqrt(var) # [A, C/2, ...]
            dist1 = tcdist.Normal(mean, std) # [A, C/2, ...]

            name2param = {'mean': mean[None], 'var': var[None]} # [1, A, C/2, ...]
            ws = torch.ones(mean.shape[0], device=mean.device) / mean.shape[0] # [A]
            ws = ws[None, ..., *[None] * (mean.dim() - 1)] # [1, A, 1, ...]
            name2param_geomean = self.net.poe.get_poe(name2param, ws=ws, params_returned=['mean', 'std'])
            dist2 = tcdist.Normal(name2param_geomean['mean'], name2param_geomean['std']) # [1, C/2, ...]

            kl = tcdist.kl_divergence(dist1, dist2) # [A, C/2, ...]
            kl = torch.mean(kl)
            kls.append(kl)
        loss = sum(kls) / len(kls)
        return loss

    def loss_kl_c_sample_unif(self, data, output):
        loss = 0.
        losses_domain = []
        for domain in self.cfg.var.domains:
            assert not self.cfg.net.atlas.qc.calc.levelwise
            probs_a = self.net.get_probs_atlas(domain)['0'] # [B, A]
            dist_probs = tcdist.Categorical(probs=probs_a)
            dist_uni = tcdist.Categorical(probs=torch.ones_like(probs_a) / probs_a.shape[1])
            kl = tcdist.kl_divergence(dist_probs, dist_uni) # [B]
            assert kl.shape[0] == probs_a.shape[0]
            assert len(kl.shape) == 1
            losses_domain.append(torch.mean(kl))
        loss = self.calc_final_loss_by_weights_domain(torch.stack(losses_domain), self.cfg.var.domains,
                                                      self.cfg.model.losses.kl_c.ws_domain)
        return loss

    def loss_c_neg_variance(self, data, output):
        losses_domain = []
        for domain in self.cfg.var.domains:
            assert not self.cfg.net.atlas.qc.calc.levelwise
            probs_a = self.net.get_probs_atlas(domain)['0'] # [B, A]
            var = torch.var(probs_a, dim=0) # [A]
            var = torch.mean(var)
            losses_domain.append(-var)
        loss = sum(losses_domain) / len(losses_domain)
        return loss

    def loss_kl_c_batch_unif(self, data, output):
        # mean atlas probablities across the batch should be uniform
        losses_domain = []
        for domain in self.cfg.var.domains:
            assert not self.cfg.net.atlas.qc.calc.levelwise
            probs_a = self.net.get_probs_atlas(domain)['0'] # [B, A]
            probs_a = torch.mean(probs_a, dim=0) # [A]
            dist_probs = tcdist.Categorical(probs=probs_a)
            dist_uni = tcdist.Categorical(probs=torch.ones_like(probs_a) / probs_a.shape[0])
            kl = tcdist.kl_divergence(dist_probs, dist_uni)
            losses_domain.append(kl)
        loss = self.calc_final_loss_by_weights_domain(torch.stack(losses_domain), self.cfg.var.domains,
                                                      self.cfg.model.losses.kl_c_batch_unif.ws_domain)
        return loss

    def loss_kl_c_sample_ent(self, data, output):
        # entropy of the atlas probabilities should be minimized
        losses_domain = []
        for domain in self.cfg.var.domains:
            assert not self.cfg.net.atlas.qc.calc.levelwise
            probs_a = self.net.get_probs_atlas(domain)['0'] # [B, A]
            dist_probs = tcdist.Categorical(probs=probs_a)
            kl = dist_probs.entropy() # [B]
            assert kl.shape[0] == probs_a.shape[0]
            losses_domain.append(torch.mean(kl))
        loss = self.calc_final_loss_by_weights_domain(torch.stack(losses_domain), self.cfg.var.domains,
                                                      self.cfg.model.losses.kl_c_sample_ent.ws_domain)
        return loss

    def loss_c_domain_diff_batch_mean(self, data, output):
        probs_source = self.net.get_probs_atlas('source')['0'] # [B, A]
        probs_target = self.net.get_probs_atlas('target')['0'] # [B, A]
        mean_source = torch.mean(probs_source, dim=0) # [A]
        mean_target = torch.mean(probs_target, dim=0) # [A]
        loss = torch.norm(mean_source - mean_target, p=2)
        return loss

    def loss_c_domain_diff_batch_covar(self, data, output):
        probs_source = self.net.get_probs_atlas('source')['0'] # [B, A]
        probs_target = self.net.get_probs_atlas('target')['0'] # [B, A]
        B = probs_source.shape[0]
        assert B == probs_target.shape[0]
        probs_source = probs_source - torch.mean(probs_source, dim=0, keepdim=True) # [B, A]
        probs_target = probs_target - torch.mean(probs_target, dim=0, keepdim=True) # [B, A]
        covar_source = probs_source.T @ probs_source # [A, A]
        covar_target = probs_target.T @ probs_target # [A, A]
        covar_source = covar_source / (B-1)
        covar_target = covar_target / (B-1)
        loss = torch.norm(covar_source - covar_target, p='fro')**2
        return loss

    def loss_distance_qz(self, data, output):
        calc_loss = GaussianDistance()
        lvs = self.cfg.model.losses.distance_qz.levels
        lvs = [str(lv) for lv in lvs]
        losses = []
        if self.cfg.model.losses.distance_qz.based_on == 'feat':
            lv2feat_source = self.net.get_lv2feat('source') # [B, C, ...]
            lv2feat_target = self.net.get_lv2feat('target') # [B, C, ...]
            for lv in lvs:
                mean_source, logits_source = lv2feat_source[lv].chunk(2, dim=1)
                var_source = self.net.poe.logits2var(logits_source)
                log_var_source = torch.log(var_source)
                mean_target, logits_target = lv2feat_target[lv].chunk(2, dim=1)
                var_target = self.net.poe.logits2var(logits_target)
                log_var_target = torch.log(var_target)
                loss = calc_loss(mean_source, log_var_source, mean_target, log_var_target)
                losses.append(loss)
        elif self.cfg.model.losses.distance_qz.based_on == 'atlas':
            assert self.cfg.net.atlas.soft_idx
            lv2name2param_source = self.net.get_lv2name2param_atlas('source') # [BxS, C/2, ...]
            lv2name2param_target = self.net.get_lv2name2param_atlas('target') # [BxS, C/2, ...]
            for lv in lvs:
                mean_source = lv2name2param_source[lv]['mean']
                std_source = lv2name2param_source[lv]['std']
                log_var_source = torch.log(std_source) * 2
                mean_target = lv2name2param_target[lv]['mean']
                std_target = lv2name2param_target[lv]['std']
                log_var_target = torch.log(std_target) * 2
                loss = calc_loss(mean_source, log_var_source, mean_target, log_var_target)
                losses.append(loss)
        else:
            raise ValueError(f'unknown based_on: {self.cfg.model.losses.distance_qz.based_on}')
        loss = sum(losses) / len(losses)
        return loss

    def loss_pos_slice(self, data, output):
        losses_domain = []
        for domain in self.cfg.var.domains:
            poss_pred = self.net.get_poss_pred(domain) # [B]
            poss_gt = data[f'pos_slice_{domain}'] # [B]
            loss = F.mse_loss(poss_pred, poss_gt)
            losses_domain.append(loss)
        loss = self.calc_final_loss_by_weights_domain(torch.stack(losses_domain), self.cfg.var.domains,
                                                      self.cfg.model.losses.pos_slice.ws_domain)
        return loss

    def loss_pos_slice_soft_contrastive(self, data, output):
        if hasattr(self, 'queue_qc_source'):
            probs_source = self.queue_qc_source # [B1, A]
            assert hasattr(self, 'queue_poss_source')
            poss_source = self.queue_poss_source # [B1]
        else:
            probs_source = self.net.get_probs_atlas('source')['0'] # [B1, A]
            poss_source = data['pos_slice_source'] # [B1]

        probs_target = self.net.get_probs_atlas('target')['0'] # [B2, A]
        poss_target = data['pos_slice_target'] # [B2]

        sim_pos = 1 - torch.abs(poss_source[:, None] - poss_target[None]) # [B1, B2]
        dist_prob = fisher_rao_distance(probs_source[None], probs_target[None]) # [1, B1, B2]
        assert dist_prob.shape[0] == 1
        dist_prob = dist_prob[0] / np.pi # [B1, B2], in [0, 1]

        margin = torch.quantile(dist_prob, 0.3, dim=0, keepdim=True) # [1, B2]
        loss = sim_pos * dist_prob + (1 - sim_pos) * torch.clamp(margin - dist_prob, min=0)
        loss = torch.mean(loss)
        return loss

    def loss_same_qc_rand_bias_field(self, data, output):
        cfg_this = self.cfg.model.losses.same_qc_rand_bias_field
        losses = []
        for domain in self.cfg.var.domains:
            if cfg_this.ws_domain[['source', 'target'].index(domain)] == 0.:
                continue
            assert not self.cfg.net.atlas.qc.calc.levelwise
            probs_1 = self.net.get_probs_atlas(domain)['0'] # [B0, A]
            probs_1 = probs_1[data[f'idxs_img_aug_{domain}']] # [B, A]
            probs_2 = self.net.get_probs_atlas(f'{domain}_aug')['1'] # [B, A]
            if cfg_this.metric == 'fisher_rao':
                dist = fisher_rao_distance(probs_1[:, None], probs_2[:, None]) # [B, 1, 1]
                assert dist.shape[1] == dist.shape[2] == 1
                dist = dist[:, 0, 0] # [B]
                loss = torch.mean(dist)
            else:
                raise NotImplementedError(f'unknown metric: {cfg_this.metric}')
            losses.append(loss)
        if len(losses) == 1:
            loss = losses[0]
        else:
            assert len(losses) == 2
            loss = self.calc_final_loss_by_weights_domain(torch.stack(losses), self.cfg.var.domains,
                                                            cfg_this.ws_domain)
        return loss

    def loss_same_disp_rand_bias_field(self, data, output):
        cfg_this = self.cfg.model.losses.same_disp_rand_bias_field
        losses = []
        for domain in self.cfg.var.domains:
            if cfg_this.ws_domain[['source', 'target'].index(domain)] == 0.:
                continue
            assert not self.cfg.net.atlas.qc.calc.levelwise
            disp_1 = self.net.get_disp_final(domain) # [B0, D, ...]
            disp_1 = disp_1[data[f'idxs_img_aug_{domain}']] # [B, D, ...]
            disp_2 = self.net.get_disp_final(f'{domain}_aug') # [B, D, ...]
            if cfg_this.metric == 'mse':
                loss = torch.mean((disp_1 - disp_2)**2)
            else:
                raise NotImplementedError(f'unknown metric: {cfg_this.metric}')
            losses.append(loss)
        if len(losses) == 1:
            loss = losses[0]
        else:
            assert len(losses) == 2
            loss = self.calc_final_loss_by_weights_domain(torch.stack(losses), self.cfg.var.domains,
                                                            cfg_this.ws_domain)
        return loss

    def loss_translation(self, data, output):
        """Translation disp (source) must match reverse-mapping convention: new_locs = grid + flow.
        So we sample source at (output_pos + flow). To have label-1 centroid at image center:
        center + flow = centroid => flow = centroid - center. Target = centroid - image_center.
        """
        if not self.cfg.net.reg_affine.enable or not self.cfg.net.reg_affine.translation.enable:
            raise ValueError('loss_translation requires net.reg_affine.enable=True and net.reg_affine.translation.enable=True')
        seg = data['seg_source']  # [B, 1, H, W] or [B, 1, D, H, W]
        if seg.dim() != 4:
            raise NotImplementedError('loss_translation only implemented for 2D (B, 1, H, W)')
        B, _, H, W = seg.shape
        mask = (seg == 1).float()  # [B, 1, H, W]
        n_b = mask.sum(dim=(1, 2, 3))  # [B]
        valid = n_b > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=seg.device)
        device = seg.device
        dtype = seg.dtype
        i = torch.arange(H, device=device, dtype=dtype)
        j = torch.arange(W, device=device, dtype=dtype)
        I = repeat(i, 'h -> h w', w=W)   # [H, W]
        J = repeat(j, 'w -> h w', h=H)   # [H, W]
        centroid_i = (mask * I).sum(dim=(1, 2, 3)) / n_b.clamp_min(1)
        centroid_j = (mask * J).sum(dim=(1, 2, 3)) / n_b.clamp_min(1)
        image_center_i = (H - 1) / 2.0
        image_center_j = (W - 1) / 2.0
        # Reverse mapping: flow = centroid - center so that center + flow = centroid
        target_di = centroid_i - image_center_i  # [B]
        target_dj = centroid_j - image_center_j  # [B]
        target = torch.stack([target_di, target_dj], dim=1)  # [B, 2]
        pred = self.net.get_params_affine_translation('source')  # [B, 2] at level_predict
        level_predict = self.cfg.net.reg_affine.translation.level_predict
        pred_img = pred * (2.0 ** level_predict)
        loss_per_sample = ((pred_img - target) ** 2).sum(dim=1).sqrt() / H
        loss = (loss_per_sample * valid.float()).sum() / valid.float().sum().clamp_min(1)

        vals = torch.cat([target, pred_img], dim=1).detach().cpu().numpy()
        formatted = [[f'{x:.4f}' for x in row] for row in vals]
        print(f'tgt_trans vs pred_trans: {formatted}')
        return loss

    def loss_zoom(self, data, output):
        """Target zoom from foreground (seg>=1) equivalent radius.
        Zoom is the linear scale factor (output vs input); target = reference_radius / radius.
        Equivalent radius r = sqrt(area/pi). Set cfg.model.losses.zoom.reference_radius.
        """
        if not self.cfg.net.reg_affine.enable or not self.cfg.net.reg_affine.zoom.enable:
            raise ValueError('loss_zoom requires net.reg_affine.enable=True and net.reg_affine.zoom.enable=True')
        if not self.cfg.net.reg_affine.zoom.isotropic:
            raise ValueError('loss_zoom only supports isotropic zoom (one scale); set net.reg_affine.zoom.isotropic=True')
        seg = data['seg_source']  # [B, 1, H, W] or [B, 1, D, H, W]
        if seg.dim() != 4:
            raise NotImplementedError('loss_zoom only implemented for 2D (B, 1, H, W)')
        mask = (seg >= 1).float()  # foreground
        area = mask.sum(dim=(1, 2, 3))  # [B], pixel count
        valid = area > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=seg.device)
        reference_radius = float(self.cfg.model.losses.zoom.reference_radius)
        radius = (area.clamp_min(1e-6) / np.pi).sqrt()  # [B], equivalent radius
        target_scale = radius / reference_radius  # [B]
        pred_zoom = self.net.get_params_affine_zoom('source')  # [B, n_zoom]
        assert pred_zoom.shape[1] == 1
        target_scale = target_scale.unsqueeze(1) # [B, 1]
        loss_per_sample = ((pred_zoom.log() - target_scale.log()) ** 2).sum(dim=1)
        loss = (loss_per_sample * valid.float()).sum() / valid.float().sum().clamp_min(1)
        
        vals = torch.cat([target_scale, pred_zoom], dim=1).detach().cpu().numpy()
        formatted = [[f'{x:.4f}' for x in row] for row in vals]
        print(f'tgt_zoom vs pred_zoom: {formatted}')
        return loss

    def before_epoch(self, mode='train', i_repeat=0):
        super().before_epoch(mode, i_repeat)
        if mode == 'train':
            if i_repeat == 0:
                if self.cfg.model.source_free_stage_2.enable:
                    assert self.cfg.exp.train.path_model_trained is not None
                    cfg_trainable = self.cfg.model.source_free_stage_2.trainable
                    if not cfg_trainable.anchor:
                        for l in range(self.cfg.net.n_levels):
                            atlas = getattr(self.net, f'atlas_lv{l}') # [A, C, ...]
                            # freeze the atlas, which is a nn.Parameter
                            assert hasattr(atlas, 'requires_grad')
                            atlas.requires_grad = False
                    if not cfg_trainable.encoder_style:
                        for param in self.net.encoder_style.parameters():
                            param.requires_grad = False

                    if not cfg_trainable.encoder_content.shallow:
                        if cfg_trainable.encoder_content.deep:
                            raise NotImplementedError
                        for param in self.net.domain2encoder['source'].parameters():
                            param.requires_grad = False
                    else:
                        if not cfg_trainable.encoder_content.deep:
                            raise NotImplementedError
                    self.net.domain2encoder['target'] = self.net.domain2encoder['source']
                        
                    if not cfg_trainable.net_c:
                        assert not self.cfg.net.atlas.qc.calc.levelwise
                        for param in self.net.domain2net_c['source'].parameters():
                            param.requires_grad = False
                    self.net.domain2net_c['target'] = self.net.domain2net_c['source']

                    if not cfg_trainable.reg:
                        for l in self.cfg.net.reg.levels:
                            for param in self.net.domain_lv2reg[f'source_{l}'].parameters():
                                param.requires_grad = False
                            self.net.domain_lv2reg[f'target_{l}'] = self.net.domain_lv2reg[f'source_{l}']
                    
                    if not cfg_trainable.decoder_rec:
                        for param in self.net.domain2decoder_rec['source'].parameters():
                            param.requires_grad = False
                    self.net.domain2decoder_rec['target'] = self.net.domain2decoder_rec['source']
                    
                    assert not cfg_trainable.decoder_seg
                    if not cfg_trainable.decoder_seg:
                        for param in self.net.decoder_seg.parameters():
                            param.requires_grad = False
                    print('Freezed some parameters for source-free stage 2 training')


            if self.cfg.net.atlas.freeze_grad.enable:
                assert self.cfg.net.atlas.freeze_grad.mode == 'periodic'
                if self.current_epoch % self.cfg.net.atlas.freeze_grad.periodic.period == 0:
                    for lv in range(self.cfg.net.n_levels):
                        atlas = getattr(self.net, f'atlas_lv{lv}')
                        atlas.requires_grad = not atlas.requires_grad

            for name_loss in ['kl_c_domain_diff', 'kl_atlas']:
                schedule = self.cfg.model.losses[name_loss].get('schedule', {})
                start, end = schedule.get('start_end_epochs', [0, 0])
                assert -1 <= start <= end
                if end == 0:
                    continue
                type_sch = schedule.get('type', 'linear')
                w_end = self.cfg.model.ws_loss[name_loss]
                if self.current_epoch <= start:
                    w = 0.
                elif self.current_epoch >= end:
                    w = w_end
                elif type_sch == 'linear':
                    w = (self.current_epoch - start) / (end-start) * w_end
                else:
                    raise ValueError(f'unknown type of schedule: {type_sch}')
                if not hasattr(self, 'name_loss2w_current_epoch'):
                    self.name_loss2w_current_epoch = {}
                self.name_loss2w_current_epoch[name_loss] = w
                print('(loss weight scheduler) ', 'loss:', name_loss, 'weight:', w)

    def get_metrics(self, data, output, mode='train'):
        super().get_metrics(data, output, mode)

        with torch.no_grad():
            if mode in ['val', 'test']:
                # Check if reconstruction metrics are needed
                needs_recon_metrics = False
                if mode == 'val':
                    if hasattr(self.cfg.exp.val, 'metrics'):
                        needs_recon_metrics = (getattr(self.cfg.exp.val.metrics, 'psnr', False)
                                               or getattr(self.cfg.exp.val.metrics, 'ssim', False)
                                               or getattr(self.cfg.exp.val.metrics, 'l1', False))
                elif mode == 'test':
                    if hasattr(self.cfg.exp.test, 'metrics'):
                        needs_recon_metrics = (getattr(self.cfg.exp.test.metrics, 'psnr', False)
                                               or getattr(self.cfg.exp.test.metrics, 'ssim', False)
                                               or getattr(self.cfg.exp.test.metrics, 'l1', False))
                
                if not self.test_by_sliding_win:
                    for domain in self.cfg.var.domains:
                        if self.cfg.net.atlas.qc.mode.name == 'deterministic':
                            probs_atlas = self.net.get_probs_atlas(domain)['0'] # [B, A]
                        elif self.cfg.net.atlas.qc.mode.name == 'dirichlet':
                            probs_atlas = self.net.get_params_dist_prob_atlas(domain)['0'] # [B, A]
                        else:
                            raise ValueError(f'unknown qc mode: {self.cfg.net.qc.mode.name}')
                        idxs_assign_atlas = self.net.get_idxs_atlas(domain)['0'] # [B, S=1, A]
                        assert idxs_assign_atlas.shape[1] == 1
                        idxs_assign_atlas = torch.argmax(idxs_assign_atlas[:, 0], dim=-1) # [B]
                        assert len(data[f'patient_slice_{domain}']) == idxs_assign_atlas.shape[0]
                        for b, patient_slice in enumerate(data[f'patient_slice_{domain}']):
                            self.metric_store.add_other(f'patient_slice_{domain}', patient_slice)
                            self.metric_store.add_other(f'atlas_assigned_{domain}', idxs_assign_atlas[b].item())
                            for a in range(probs_atlas.shape[1]):
                                self.metric_store.add_other(f'prob_atlas{a}_{domain}', probs_atlas[b, a].item())
                        # Store gt_reg for dice similarity calculation into metric_store to align with df/patient_slice
                        if mode in ['val', 'test'] and not self.test_by_sliding_win:
                            gt_reg_all = self.net.get_segs_gt_reg(domain)  # [BxS, 1, ...]
                            patient_slices = data[f'patient_slice_{domain}']
                            B = len(patient_slices)
                            S = gt_reg_all.shape[0] // B
                            assert S == 1
                            for i, patient_slice in enumerate(patient_slices):
                                gt_reg = gt_reg_all[i].cpu()  # [1, ...]
                                self.metric_store.add_other(f'gt_reg_{domain}', gt_reg)

                        # Store reconstruction images for reconstruction metrics (for all patients)
                        if needs_recon_metrics and not self.test_by_sliding_win:
                            try:
                                rec_all = self.net.get_imgs_rec_ori(domain)  # [BxS, C, ...]
                                patient_slices = data[f'patient_slice_{domain}']
                                B = len(patient_slices)
                                S = rec_all.shape[0] // B if rec_all.shape[0] >= B else 1
                                
                                for i, patient_slice in enumerate(patient_slices):
                                    patient = patient_slice.split('_')[0]
                                    
                                    # Store reconstruction
                                    if S > 1:
                                        # If S > 1, take mean across samples for this batch item
                                        rec = rec_all[i * S:(i + 1) * S, :1].mean(dim=0, keepdim=True)  # [1, ...]
                                    else:
                                        # If S == 1, directly index
                                        rec = rec_all[i, :1]  # [1, ...]
                                    rec = rec.cpu()
                                    self.patient_store.add_tensor(f'{domain}_patient_{patient}', f'imgs_rec_{domain}', rec)
                            except Exception as e:
                                # If reconstruction cannot be obtained, skip (will be caught by assert in after_epoch)
                                pass

                        # visualization of example cases
                        if self.cfg.exp.mode != 'train' and mode != 'train':
                            patients_vis = self.cfg.exp[mode].patients_vis[domain]
                            patient_slices = data[f'patient_slice_{domain}']
                            patients = [patient_slice.split('_')[0] for patient_slice in patient_slices]
                            slices = [int(patient_slice.split('_')[1]) for patient_slice in patient_slices]
                            ns_slice = [int(v) for v in data[f'n_slices_patient_{domain}']]
                            n_slices_vis = 15
                            for b, patient in enumerate(patients):
                                if patient not in patients_vis:
                                    continue
                                n_slice = ns_slice[b]
                                if n_slice > n_slices_vis:
                                    slices_vis = np.linspace(0, n_slice-1, n_slices_vis, dtype=int)
                                    if slices[b] not in slices_vis:
                                        continue
                                self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'vis_img_ori', data[f'img_{domain}'][b].cpu()) # [1, ...]
                                self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'vis_img_reg', self.net.get_imgs_reg(domain)[b].cpu()) # [1, ...]
                                self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'vis_rec_ori', self.net.get_imgs_rec_ori(domain)[b, :1].cpu()) # [1, ...]
                                self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'vis_rec_reg', self.net.get_imgs_rec_reg(domain)[b, :1].cpu()) # [1, ...]
                                seg_ori = self.net.get_logits_seg(domain) # [B, K, ...]
                                self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'vis_seg_ori', torch.argmax(seg_ori[b], dim=0, keepdim=True).cpu()) # [1, ...]
                                seg_reg = self.net.get_logits_seg_reg(domain) # [B, K, ...]
                                self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'vis_seg_reg', torch.argmax(seg_reg[b], dim=0, keepdim=True).cpu()) # [1, ...]
                                self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'vis_seg_gt', data[f'seg_{domain}'][b].cpu()) # [1, ...]
                                disp = self.net.get_disp_final(domain) # [B, D, ...]
                                grid = self.get_grid_img(disp) # [B, 1, ...]
                                self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'vis_disp', self.net.lv2stn['0'](grid[[b]], disp[[b]])[0]) # [1, ...]

                        # save results if needed
                        if self.cfg.exp.mode == 'test':
                            patients_vis = self.cfg.exp[mode].patients_vis[domain]
                            cfg_save = self.cfg.exp.test.save
                            for b, patient in enumerate(patients):
                                if cfg_save.seg_reg:
                                    seg_reg = self.net.get_logits_seg_reg(domain) # [B, K, ...]
                                    self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'seg_reg', torch.argmax(seg_reg[b], dim=0, keepdim=True).cpu()) # [1, ...]
                                if cfg_save.rec_reg:
                                    self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'rec_reg', self.net.get_imgs_rec_reg(domain)[b, :1].cpu()) # [1, ...]
                                if cfg_save.rec_ori:
                                    self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'rec_ori', self.net.get_imgs_rec_ori(domain)[b, :1].cpu())
                                if cfg_save.disp:
                                    disp = self.net.get_disp_final(domain) # [B, D, ...]
                                    self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'disp', disp[b].cpu()) # [D, ...]
                                    disp_velocity = self.net.get_disp_composed_by_velocity(domain)
                                    self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'disp_velocity', disp_velocity[b].cpu())
                                    if self.cfg.net.reg_affine.enable:
                                        lv2disps_affine = self.net.get_disps_affine(domain)
                                        disps_affine = lv2disps_affine['0']
                                        if self.cfg.net.reg_affine.zoom.enable:
                                            self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'disp_zoom', disps_affine[0][b].cpu())
                                        if self.cfg.net.reg_affine.translation.enable:
                                            idx_trans = 1 if self.cfg.net.reg_affine.zoom.enable else 0
                                            self.patient_store.add_tensor(f'{domain}_patient_{patient}', 'disp_trans', disps_affine[idx_trans][b].cpu())

                        if self.cfg.model.ws_loss.adv_enc_feat > 0.:
                            lv2out_discriminator_enc_feat = self.net.get_output_discriminator_enc_feat(domain) # [B, 1]
                            for lv in lv2out_discriminator_enc_feat.keys():
                                out = lv2out_discriminator_enc_feat[lv]
                                assert out.shape[1] == 1
                                out = out[:, 0] # [B]
                                if domain == 'source':
                                    correct = out >= 0.5
                                elif domain == 'target':
                                    correct = out < 0.5
                                else:
                                    raise ValueError(f'unknown domain: {domain}')
                                correct = correct.float().cpu().numpy().tolist()
                                self.metric_store.add_metric(f'acc_discrim_enc_feat_{domain}_{lv}', correct)
                                self.metric_store.add_metric(f'acc_discrim_enc_feat', correct)

        return self.metrics_iter

    def after_epoch(self, mode='train'):
        with torch.no_grad():
            for k, v in self.metrics_epoch.items():
                self.metrics_epoch[k] = v / len(getattr(self.cfg.var.obj_operator, f'{mode}_set'))


            if mode in ['val', 'test']:
                names_metric = ['dice']
                if self.cfg.exp.mode == 'test' and ('harborview' not in self.cfg.dataset.name and 'mocha' not in self.cfg.dataset.name):
                    names_metric.append('assd')
                
                # Add reconstruction metrics if configured
                if mode == 'val':
                    # Validation metrics
                    if hasattr(self.cfg.exp.val, 'metrics'):
                        if getattr(self.cfg.exp.val.metrics, 'psnr', False):
                            names_metric.append('psnr')
                        if getattr(self.cfg.exp.val.metrics, 'ssim', False):
                            names_metric.append('ssim')
                        if getattr(self.cfg.exp.val.metrics, 'l1', False):
                            names_metric.append('l1')
                elif mode == 'test':
                    # Test metrics
                    if hasattr(self.cfg.exp.test, 'metrics'):
                        if getattr(self.cfg.exp.test.metrics, 'psnr', False):
                            names_metric.append('psnr')
                        if getattr(self.cfg.exp.test.metrics, 'ssim', False):
                            names_metric.append('ssim')
                        if getattr(self.cfg.exp.test.metrics, 'l1', False):
                            names_metric.append('l1')
                
                self.patient_store.aggregate_tensors()

                for domain_patient_idx, name2tensors in self.patient_store.patient2name2tensors.items():
                    domain, _, patient = domain_patient_idx.split('_')
                    gt = name2tensors[f'gt_{domain}']
                    logits = name2tensors[f'logits_{domain}']

                    ds = getattr(self.cfg.var.obj_operator, f'{mode}_set')
                    if hasattr(ds, 'domain2idx2spacing'):
                        try:
                            spacing = getattr(self.cfg.var.obj_operator,
                                              f'{mode}_set').domain2idx2spacing[domain][int(patient)]
                        except:
                            spacing = getattr(self.cfg.var.obj_operator,
                                              f'{mode}_set').domain2idx2spacing[domain][patient]
                    else:
                        spacing = None
                    
                    # Get img and rec for reconstruction metrics
                    img = name2tensors.get(f'img_{domain}', None)
                    rec = name2tensors.get(f'imgs_rec_{domain}', None)
                    
                    name2value_metric = self.metric_calculator(gt, logits, 'seg', 'logits', dim_class=1,
                                                               ignore_bg=False, names_metric=names_metric,
                                                               spacing=spacing, img=img, rec=rec) # [K-1]
                    for name_metric, values in name2value_metric.items():
                        # Segmentation metrics return [K], reconstruction metrics return scalar
                        if name_metric in ['psnr', 'ssim', 'l1']:
                            # Reconstruction metrics: scalar value
                            self.metric_store.add_metric(f'{name_metric}_{domain}', values.item())
                        else:
                            # Segmentation metrics: [K] array
                            values = values[1:]
                            if isinstance(values, np.ndarray):
                                values = torch.from_numpy(values)
                            for i, value in enumerate(values):
                                name_class = self.cfg.dataset.idx2name_class[str(i + 1)]
                                self.metric_store.add_metric(f'{name_metric}_{name_class}_{domain}', value.item())
                            self.metric_store.add_metric(f'{name_metric}_mean_{domain}', torch.mean(values).item())
                    self.metric_store.add_other(f'idx_patient_{domain}', patient)

                if not self.test_by_sliding_win:
                    """
                    ################################ log atlas probs ################################
                    """
                    if self.cfg.exp.source_only:
                        domains = ['source']
                    else:
                        if self.cfg.dataset.name == 'harborview':
                            domains = ['target']
                        else:
                            domains = ['source', 'target']
                    for domain in domains:
                        col = self.metric_store.name2value_other[f'atlas_assigned_{domain}']
                        col = [None, None] + col
                        name2values = {'atlas_assigned': col}
                        for a in range(self.cfg.net.atlas.num):
                            col = self.metric_store.name2value_other[f'prob_atlas{a}_{domain}'] # [B]
                            col = [np.mean(np.asarray(name2values['atlas_assigned'][2:]) == a), np.mean(col)] + col
                            name2values[f'atlas_{a}'] = col

                        idx_col = self.metric_store.name2value_other[f'patient_slice_{domain}']
                        idx_col = ['atlas_freq', 'mean_prob'] + idx_col
                        df = pd.DataFrame(name2values, index=idx_col)
                        df = df.round(2)
                        df.index.name = 'patient_slice'
                        print(df)


                        ws = df.iloc[2:, 1:].to_numpy() # [N, A]
                        w_bar = ws.mean(axis=0) # [A]
                        if self.cfg.exp[mode].metrics.w_bar:
                            for j, w in enumerate(w_bar):
                                self.metrics_epoch[f'w_bar_{j}_{domain}'] = w
                        if self.cfg.exp[mode].metrics.usage_entropy:
                            usage_entropy = -np.sum(w_bar * np.log(w_bar + 1e-10))
                            N_eff = np.exp(usage_entropy)
                            self.metrics_epoch[f'usage_entropy_{domain}'] = usage_entropy
                            self.metrics_epoch[f'N_eff_{domain}'] = N_eff


                        if self.cfg.exp[mode].metrics.logdet_covar_w:
                            w_centered = ws - w_bar # [N, A]
                            U = self.U_for_logdet_covar_w
                            Z = w_centered @ U # [N, A-1]
                            Sigma = (Z.T @ Z) / (Z.shape[0] - 1) # [A-1, A-1]
                            Sigma_reg = Sigma + 1e-6 * np.eye(Sigma.shape[0])
                            Q_value = np.linalg.slogdet(Sigma_reg)[1]
                            self.metrics_epoch[f'logdet_covar_w_{domain}'] = Q_value

                        def fisher_rao_pairwise(ws: np.ndarray, eps: float = 1e-12):
                            """
                            ws: (N, A), rows sum to 1, nonnegative
                            return: (N, N) Fisher–Rao distance matrix
                            """
                            ws = np.asarray(ws, dtype=np.float64)
                            ws = np.clip(ws, 0.0, None)
                            S = np.sqrt(ws + eps)        # [N, A]
                            inner = S @ S.T # [N, N]
                            inner = np.clip(inner, -1.0, 1.0)
                            D = np.arccos(inner)
                            return D

                        def spearman_from_pairwise(D_fr: np.ndarray, D_dice: np.ndarray):
                            """
                            D_fr, D_dice: (N, N) pairwise distance/dissimilarity matrices
                            returns: (r_s, p_value)
                            """
                            assert D_fr.shape == D_dice.shape and D_fr.ndim == 2 and D_fr.shape[0] == D_fr.shape[1]
                            N = D_fr.shape[0]
                            iu = np.triu_indices(N, k=1)
                            x = D_fr[iu]     # shape: (N*(N-1)/2,)
                            y = D_dice[iu]
                            r_s, p_value = spearmanr(x, y)
                            return r_s, p_value

                        # Calculate D_dice: pairwise dice similarity matrix for gt_reg masks
                        def dice_pairwise(masks_onehot: torch.Tensor):
                            """
                            masks_onehot: (N, K, ...), onehot encoded masks
                            return: (N, N) dice similarity matrix
                            """
                            masks1 = masks_onehot.unsqueeze(1)  # [N, 1, K, ...]
                            masks2 = masks_onehot.unsqueeze(0)  # [1, N, K, ...]
                            dims_spatial = list(range(3, masks1.dim()))  # spatial dims (exclude class)
                            numerator = 2 * torch.sum(masks1 * masks2, dim=dims_spatial)  # [N, N, K]
                            denominator = torch.sum(masks1 + masks2, dim=dims_spatial)  # [N, N, K]
                            dice = (numerator + 1e-8) / (denominator + 1e-8)  # [N, N, K]
                            D_dice = dice.mean(dim=2)  # [N, N]
                            return D_dice

                        if self.cfg.exp[mode].metrics.spearman_r_dice_vs_fr:
                            D_fr = fisher_rao_pairwise(ws) # [N, N]
                            gt_reg_key = f'gt_reg_{domain}'
                            gt_regs = self.metric_store.name2value_other[gt_reg_key]  # list of tensors
                            masks = torch.stack([torch.as_tensor(m) for m in gt_regs])  # [N, 1, ...]
                            masks_onehot = seg2onehot(masks, n_classes=self.cfg.dataset.n_classes, dim_class=1)  # [N, K, ...]
                            D_dice = dice_pairwise(masks_onehot).cpu().numpy()  # [N, N]
                            spearman_r, p_value = spearman_from_pairwise(D_fr, 1-D_dice)
                            if np.isnan(spearman_r):
                                spearman_r = -2
                            self.metrics_epoch[f'spearman_r_dice_vs_fr_{domain}'] = spearman_r


                for domain_patient_idx, name2tensors in self.patient_store.patient2name2tensors.items():
                    domain, _, patient = domain_patient_idx.split('_')
                    gt = name2tensors[f'gt_{domain}']
                    logits = name2tensors[f'logits_{domain}']
                    if self.cfg.exp.mode == 'test':
                        cfg_save = self.cfg.exp.test.save
                        path_folder = os.path.join(self.cfg.var.obj_operator.path_exp, 'vis_seg', self.cfg.dataset.name,
                                                   self.cfg.model.name)
                        os.makedirs(path_folder, exist_ok=True)
                        if self.cfg.dataset.name == 'harborview':
                            pass
                        else:
                            try:
                                patient = f'{int(patient):03d}'
                            except:
                                pass

                        if cfg_save.img_ori:
                            img = name2tensors[f'img_{domain}']
                            img = img.squeeze().cpu().numpy()
                            nii = sitk.GetImageFromArray(img)
                            sitk.WriteImage(nii, os.path.join(path_folder, f'{domain}_{patient}_img_ori.nii.gz'))
                        if cfg_save.seg_gt:
                            gt = gt.squeeze().cpu().numpy()
                            nii = sitk.GetImageFromArray(gt)
                            sitk.WriteImage(nii, os.path.join(path_folder, f'{domain}_{patient}_seg_gt.nii.gz'))
                        if cfg_save.seg_pred:
                            pred = logits.argmax(dim=1).squeeze().cpu().numpy()
                            nii = sitk.GetImageFromArray(pred)
                            sitk.WriteImage(nii, os.path.join(path_folder, f'{domain}_{patient}_seg_pred.nii.gz'))
                        if cfg_save.seg_reg:
                            seg_reg = name2tensors['seg_reg'].squeeze().cpu().numpy() # [B, H, W]
                            nii = sitk.GetImageFromArray(seg_reg)
                            sitk.WriteImage(nii, os.path.join(path_folder, f'{domain}_{patient}_seg_reg.nii.gz'))
                        if cfg_save.rec_reg:
                            rec_reg = name2tensors['rec_reg'].squeeze().cpu().numpy() # [B, H, W]
                            nii = sitk.GetImageFromArray(rec_reg)
                            sitk.WriteImage(nii, os.path.join(path_folder, f'{domain}_{patient}_rec_reg.nii.gz'))
                        if cfg_save.rec_ori:
                            rec_ori = name2tensors['rec_ori'].squeeze().cpu().numpy() # [B, H, W]
                            nii = sitk.GetImageFromArray(rec_ori)
                            sitk.WriteImage(nii, os.path.join(path_folder, f'{domain}_{patient}_rec_ori.nii.gz'))
                        if cfg_save.disp:
                            def _save_disp_nii(disp_tensor, path):
                                disp = disp_tensor.cpu().numpy()
                                assert len(disp.shape) == 4 and disp.shape[1] == 2
                                disp = np.swapaxes(disp, 0, 1)
                                zeros = np.zeros((1, *disp.shape[1:]), dtype=disp.dtype)
                                disp = np.concatenate([disp, zeros], axis=0)
                                disp = np.moveaxis(disp, 0, -1)
                                nii = sitk.GetImageFromArray(disp, isVector=True)
                                sitk.WriteImage(nii, path)

                            disp = name2tensors['disp'] # [B, D, H, W]
                            _save_disp_nii(disp, os.path.join(path_folder, f'{domain}_{patient}_disp.nii.gz'))

                            if 'disp_velocity' in name2tensors:
                                _save_disp_nii(name2tensors['disp_velocity'], os.path.join(path_folder, f'{domain}_{patient}_disp_velocity.nii.gz'))
                            if 'disp_zoom' in name2tensors:
                                _save_disp_nii(name2tensors['disp_zoom'], os.path.join(path_folder, f'{domain}_{patient}_disp_zoom.nii.gz'))
                            if 'disp_trans' in name2tensors:
                                _save_disp_nii(name2tensors['disp_trans'], os.path.join(path_folder, f'{domain}_{patient}_disp_trans.nii.gz'))
                        if cfg_save.w:
                            ws_target = []
                            ws_source = []
                            for a in range(self.cfg.net.atlas.num):
                                col = self.metric_store.name2value_other[f'prob_atlas{a}_target'] # [B]
                                ws_target.append(col)
                                col = self.metric_store.name2value_other[f'prob_atlas{a}_source'] # [B]
                                ws_source.append(col)
                            ws_target = np.asarray(ws_target).T # [B, A]
                            ws_source = np.asarray(ws_source).T # [B, A]
                            path_folder_w = os.path.join(self.cfg.var.obj_operator.path_exp, 'w', self.cfg.dataset.name, self.cfg.model.name)
                            os.makedirs(path_folder_w, exist_ok=True)
                            df = pd.DataFrame(ws_source, columns=[f'atlas_{a}' for a in range(ws_source.shape[1])])
                            df.to_csv(os.path.join(path_folder_w, f'source_test.csv'))
                            df = pd.DataFrame(ws_target, columns=[f'atlas_{a}' for a in range(ws_target.shape[1])])
                            df.to_csv(os.path.join(path_folder_w, f'target_test.csv'))

                        if cfg_save.geodesic_seg.enable:
                            n_seg = cfg_save.geodesic_seg.n_seg
                            end_pairs = cfg_save.geodesic_seg.end_pairs # set of w pairs
                            end_sets = cfg_save.geodesic_seg.end_sets # set of w
                            if len(end_sets) > 0:
                                assert len(end_pairs) == 0
                                pairs_w = list(itertools.combinations(end_sets, 2))
                            else:
                                assert len(end_pairs) > 0
                                pairs_w = end_pairs
                            for end_pair in pairs_w:
                                end1, end2 = end_pair # [M] x 2
                                end1, end2 = torch.tensor(end1).float(), torch.tensor(end2).float() # [M]
                                end1, end2 = torch.sqrt(end1), torch.sqrt(end2) # [M]
                                end1, end2 = end1 / torch.norm(end1), end2 / torch.norm(end2)
                                theta = torch.arccos(torch.clamp(torch.sum(end1 * end2), -1, 1))
                                t_values = torch.linspace(0, 1, n_seg)[:, None] # [N, 1]
                                geodesic_points = ((torch.sin((1 - t_values) * theta) * end1 
                                                    + torch.sin(t_values * theta) * end2) / torch.sin(theta)) # [N, M]
                                ws = geodesic_points ** 2 # [N, M]
                                segs = self.net.calc_seg_by_ws(ws).cpu().numpy() # [N, H, W]
                                nii = sitk.GetImageFromArray(segs)
                                end1 = '_'.join([str(v) for v in end1.numpy()])
                                end2 = '_'.join([str(v) for v in end2.numpy()])
                                sitk.WriteImage(nii, os.path.join(path_folder, f'geodesic_seg_{end_pair[0]}__{end_pair[1]}.nii.gz'))

                for k, v in self.metric_store.name2metric.items():
                    self.metrics_epoch[k] = np.mean(v)
                if self.cfg.exp.source_only:
                    self.metrics_epoch['metric_final'] = self.metrics_epoch['dice_mean_source']
                else:
                    self.metrics_epoch['metric_final'] = self.metrics_epoch['dice_mean_target']

                if self.cfg.exp.source_only:
                    domains = ['source']
                else:
                    if self.cfg.dataset.name == 'harborview':
                        domains = ['target']
                    else:
                        domains = ['source', 'target']
                for domain in domains:
                    metrics = {
                        k: v
                        for k, v in self.metric_store.name2metric.items()
                        if domain in k and not k.startswith('acc_discrim')
                    }
                    df = pd.DataFrame(metrics, index=self.metric_store.name2value_other[f'idx_patient_{domain}'])
                    df = df.round(4)
                    df.index.name = 'patient_idx'
                    print(df)


    def get_grid_img(self, disps):
        # disps: [B, D, ...]
        img_grid = get_img_grid(n_imgs=disps.shape[0], size_img=disps.shape[2:], padding_min=4, thickness=1,
                                spacing=9).to(disps.device)
        return img_grid # [B, 1, ...]

    @staticmethod
    def display_images(images, rows, cols):
        fig = plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
        plt.tight_layout()

    def vis(self, writer, global_step, data, output, mode, in_epoch):
        super().vis(writer, global_step, data, output, mode, in_epoch)
        if self.cfg.exp.mode == 'train':
            return
        if self.test_by_sliding_win:
            return
        if mode == 'train':
            return
        if in_epoch:
            return

        label2color = {
            1: (0, 167, 234),
            2: (255, 42, 105),
            3: (0, 175, 0), 
            4: (240, 163, 9),
        }
        with torch.no_grad():
            for domain in ['source', 'target']:
                patients_vis = self.cfg.exp[mode].patients_vis[domain]
                for patient in patients_vis:
                    imgs_show = {k[4:]: v for k, v in self.patient_store.patient2name2tensors[patient].items() if k.startswith('vis')}
                    # add gt seg contours
                    keys1 = ['img_ori', 'rec_ori', 'seg_ori', 'seg_reg', 'seg_gt']
                    # keys2 = ['rec_source_atlas', 'rec_target_atlas']
                    keys2 = []
                    for key in keys1 + keys2:
                        imgs_show[key] = torch.concat([imgs_show[key]] * 3, dim=1) # [B, 3, ...]
                        if key.startswith('seg'):
                            imgs_show[key] = torch.moveaxis(imgs_show[key], 1, -1) # [B, ..., 3]
                            for label, color in label2color.items():
                                seg = imgs_show[key][..., 0] == label # [B, ...]
                                imgs_show[key][seg] = torch.Tensor(color, device=seg.device).to(imgs_show[key].dtype)
                            imgs_show[key] = torch.moveaxis(imgs_show[key], -1, 1) # [B, 3, ...]

                    seg_gt = imgs_show['seg_gt']
                    B = seg_gt.shape[0]

                    if self.cfg.dataset.name == 'amos22':
                        h = 7
                        w = B * 5 / 4 * h * 0.5
                        fig, axes = plt.subplots(4, 2, figsize=(w, h), dpi=200)
                    elif self.cfg.dataset.name == 'mscmr':
                        h = 8.75
                        w = B * 2 / 4 * h * 0.5
                        fig, axes = plt.subplots(4, 2, figsize=(w, h), dpi=200)
                    elif self.cfg.dataset.name == 'harborview':
                        h = 8.75
                        w = B * 2 / 4 * h * 0.5
                        fig, axes = plt.subplots(4, 2, figsize=(w, h), dpi=200)
                    else:
                        raise ValueError
                    for ax, key in zip(axes.reshape(-1), imgs_show.keys()):
                        ax.axis('off'), ax.set_xticks([]), ax.set_yticks([])
                        if key == 'disp':
                            pad_value = 0.
                        else:
                            pad_value = 1.
                        if key.startswith('seg'):
                            # imgs_show[key] = imgs_show[key] / (self.cfg.dataset.n_classes - 1)
                            imgs_show[key] = imgs_show[key] / 255
                        if imgs_show[key].shape[1] == 1:
                            cmap = 'gray'
                            image_grid = make_grid(imgs_show[key], nrow=B, pad_value=pad_value).cpu().numpy()[0]
                        else:
                            cmap = None
                            image_grid = make_grid(imgs_show[key], nrow=B, pad_value=pad_value).cpu().numpy()
                            image_grid = np.transpose(image_grid, (1, 2, 0))
                        ax.imshow(image_grid, cmap=cmap)
                        ax.set_title(key, fontsize=12)

                    title = f'{mode}, {domain}, patient {patient}'
                    fig.suptitle(title, fontsize=14)
                    fig.tight_layout(pad=0.01)
                    writer.add_figure(f'{mode}_{domain}_{patient}', fig, global_step)
                    self.clear_nested_dict_or_list(imgs_show)
