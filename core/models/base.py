import torch
import torch.nn as nn
from omegaconf import OmegaConf as Ocfg
from collections import OrderedDict
import numpy as np
from core.utils.seg import logits2seg, seg2onehot
import monai
import pandas as pd
import SimpleITK as sitk
import os


def get_assd(pred, gt, spacing, n_classes):
    # pred, gt: [...], 0, ..., n_classes-1
    groundtruth = gt
    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1, n_classes, 5))
    surface_distance_results = np.zeros((1, n_classes, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(n_classes):
        pred_i = (pred == i).astype(np.float32)
        if np.sum(pred_i) == 0:
            overlap_results[0, i, :] = 0
            surface_distance_results[0, i, :] = 0
        else:
            # Overlap measures
            overlap_measures_filter.Execute(ITKTrue == i, ITKPred == i)
            overlap_results[0, i, 0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0, i, 1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0, i, 2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0, i, 3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0, i, 4] = overlap_measures_filter.GetFalsePositiveError()
            # Hausdorff distance
            hausdorff_distance_filter.Execute(ITKTrue == i, ITKPred == i)

            surface_distance_results[0, i, 0] = hausdorff_distance_filter.GetHausdorffDistance()
            # Symmetric surface distance measures

            reference_distance_map = sitk.Abs(
                sitk.SignedMaurerDistanceMap(ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(
                sitk.SignedMaurerDistanceMap(ITKPred == i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred == i)
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            # Multiply the binary surface segmentations with the distance maps. The resulting distance
            # maps contain non-zero values only on the surface (they can also contain zero on the surface)
            seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

            # Get all non-zero distances and then add zero distances if required.
            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
            # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
            # segmentations, though in our case it is. More on this below.
            surface_distance_results[0, i, 1] = np.mean(all_surface_distances)

    values = surface_distance_results[0, :, 1]
    return values


class MetricCalculator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, gt, pred, gt_type, pred_type, dim_class, ignore_bg, names_metric, spacing=None, img=None,
                rec=None):
        # gt, pred: [..., K, ...], K may be 1
        # types: logits, seg, onehot
        # img, rec: [..., C, ...] for reconstruction metrics (C is typically 1)

        # convert to onehot
        if gt_type == 'logits':
            gt = logits2seg(gt, dim_class)
            gt_type = 'seg'
        if gt_type == 'seg':
            gt = seg2onehot(gt, self.n_classes, dim_class)
        if pred_type == 'logits':
            pred = logits2seg(pred, dim_class)
            pred_type = 'seg'
        if pred_type == 'seg':
            pred = seg2onehot(pred, self.n_classes, dim_class)

        if ignore_bg:
            assert gt.shape[dim_class] == self.n_classes
            assert pred.shape[dim_class] == self.n_classes
            gt = torch.narrow(gt, dim=dim_class, start=1, length=self.n_classes - 1)
            pred = torch.narrow(pred, dim=dim_class, start=1, length=self.n_classes - 1)
            assert gt.shape[dim_class] == self.n_classes - 1
            assert pred.shape[dim_class] == self.n_classes - 1

        name2value_metric = {}
        for name_metric in names_metric:
            if name_metric == 'dice':
                values = self.calc_dice(gt, pred, dim_class)
            elif name_metric == 'assd':
                values = self.calc_assd(gt, pred, dim_class, spacing)
            elif name_metric == 'psnr':
                assert img is not None and rec is not None, "img and rec are required for PSNR metric"
                values = self.calc_psnr(img, rec)
            elif name_metric == 'ssim':
                assert img is not None and rec is not None, "img and rec are required for SSIM metric"
                values = self.calc_ssim(img, rec)
            elif name_metric == 'l1':
                assert img is not None and rec is not None, "img and rec are required for L1 metric"
                values = self.calc_l1(img, rec)
            else:
                raise ValueError(f'Unknown name_metric {name_metric}')
            name2value_metric[name_metric] = values # [K] for segmentation metrics, scalar for reconstruction metrics

        return name2value_metric

    def calc_dice(self, gt, pred, dim_class):
        # gt, pred: [..., K, ...], onehot
        # return: [K]
        dims_sum = [i for i in range(len(gt.shape)) if i != dim_class]
        numerator = 2 * torch.sum(gt * pred, dim=dims_sum) # [K]
        denominator = torch.sum(gt + pred, dim=dims_sum) # [K]
        dice = (numerator+1e-8) / (denominator+1e-8) # [K]
        return dice

    def calc_assd(self, gt, pred, dim_class, spacing):
        # gt, pred: [..., K, ...], onehot
        # return: [K]
        assert dim_class == 1
        n_classes = gt.shape[dim_class]
        gt = gt.float()
        pred = pred.float()

        gt = torch.movedim(gt, dim_class, 0) # [K, ...]
        pred = torch.movedim(pred, dim_class, 0) # [K, ...]
        gt = gt[None] # [1, K, ...]
        pred = pred[None] # [1, K, ...]
        # print(spacing, gt.shape, pred.shape)

        gt = torch.argmax(gt, dim=1)[0].cpu().numpy() # [...]
        pred = torch.argmax(pred, dim=1)[0].cpu().numpy()
        groundtruth = gt
        spacing = spacing[::-1]
        ITKPred = sitk.GetImageFromArray(pred, isVector=False)
        ITKPred.SetSpacing(spacing)
        ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
        ITKTrue.SetSpacing(spacing)

        overlap_results = np.zeros((1, n_classes, 5))
        surface_distance_results = np.zeros((1, n_classes, 5))

        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

        for i in range(n_classes):
            pred_i = (pred == i).astype(np.float32)
            if np.sum(pred_i) == 0:
                overlap_results[0, i, :] = 0
                surface_distance_results[0, i, :] = 0
            else:
                # Overlap measures
                overlap_measures_filter.Execute(ITKTrue == i, ITKPred == i)
                overlap_results[0, i, 0] = overlap_measures_filter.GetJaccardCoefficient()
                overlap_results[0, i, 1] = overlap_measures_filter.GetDiceCoefficient()
                overlap_results[0, i, 2] = overlap_measures_filter.GetVolumeSimilarity()
                overlap_results[0, i, 3] = overlap_measures_filter.GetFalseNegativeError()
                overlap_results[0, i, 4] = overlap_measures_filter.GetFalsePositiveError()
                # Hausdorff distance
                hausdorff_distance_filter.Execute(ITKTrue == i, ITKPred == i)

                surface_distance_results[0, i, 0] = hausdorff_distance_filter.GetHausdorffDistance()
                # Symmetric surface distance measures

                reference_distance_map = sitk.Abs(
                    sitk.SignedMaurerDistanceMap(ITKTrue == i, squaredDistance=False, useImageSpacing=True))
                reference_surface = sitk.LabelContour(ITKTrue == i)
                statistics_image_filter = sitk.StatisticsImageFilter()
                # Get the number of pixels in the reference surface by counting all pixels that are 1.
                statistics_image_filter.Execute(reference_surface)
                num_reference_surface_pixels = int(statistics_image_filter.GetSum())

                segmented_distance_map = sitk.Abs(
                    sitk.SignedMaurerDistanceMap(ITKPred == i, squaredDistance=False, useImageSpacing=True))
                segmented_surface = sitk.LabelContour(ITKPred == i)
                # Get the number of pixels in the reference surface by counting all pixels that are 1.
                statistics_image_filter.Execute(segmented_surface)
                num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

                # Multiply the binary surface segmentations with the distance maps. The resulting distance
                # maps contain non-zero values only on the surface (they can also contain zero on the surface)
                seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
                ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

                # Get all non-zero distances and then add zero distances if required.
                seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
                seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
                seg2ref_distances = seg2ref_distances + \
                                    list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
                ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
                ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
                ref2seg_distances = ref2seg_distances + \
                                    list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

                all_surface_distances = seg2ref_distances + ref2seg_distances

                # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
                # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
                # segmentations, though in our case it is. More on this below.
                surface_distance_results[0, i, 1] = np.mean(all_surface_distances)

        values = surface_distance_results[0, :, 1] # [K]

        # values = monai.metrics.compute_average_surface_distance(pred, gt, include_background=True, symmetric=True,
        #                                                         spacing=spacing) # [K]
        # print(values)
        # assert values.shape[0] == 1
        # values = values[0] # [K]
        return values

    def calc_psnr(self, img, rec):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            img: [..., C, ...] original image (expected to be in [0, 1] range)
            rec: [..., C, ...] reconstructed image (will be clipped to [0, 1])
        
        Returns:
            psnr: scalar tensor
        """
        try:
            from torchmetrics.functional import peak_signal_noise_ratio
        except ImportError:
            raise ImportError(
                "torchmetrics is required for PSNR calculation. Please install it with: pip install torchmetrics")

        # Check if original image is in [0, 1] range
        img_min = img.min().item()
        img_max = img.max().item()
        if img_min < 0.0 or img_max > 1.0:
            # Original image should be in [0, 1] range
            raise ValueError(
                f"Original image is not in [0, 1] range (min={img_min:.4f}, max={img_max:.4f}). PSNR calculation requires images in [0, 1] range."
            )
        # Original image is in [0, 1] range, use data_range=1.0
        data_range = 1.0
        img_norm = img

        # Clip reconstruction to [0, 1] range (model output may be outside this range)
        rec_clipped = torch.clamp(rec, 0.0, 1.0)

        # Calculate PSNR
        psnr = peak_signal_noise_ratio(rec_clipped, img_norm, data_range=data_range)
        return psnr

    def calc_ssim(self, img, rec):
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            img: [..., C, ...] original image (expected to be in [0, 1] range)
            rec: [..., C, ...] reconstructed image (will be clipped to [0, 1])
        
        Returns:
            ssim: scalar tensor
        """
        try:
            from torchmetrics.functional import structural_similarity_index_measure
        except ImportError:
            raise ImportError(
                "torchmetrics is required for SSIM calculation. Please install it with: pip install torchmetrics")

        # Check if original image is in [0, 1] range
        img_min = img.min().item()
        img_max = img.max().item()
        if img_min < 0.0 or img_max > 1.0:
            # Original image should be in [0, 1] range
            raise ValueError(
                f"Original image is not in [0, 1] range (min={img_min:.4f}, max={img_max:.4f}). SSIM calculation requires images in [0, 1] range."
            )
        # Original image is in [0, 1] range, keep as is
        img_norm = img

        # Clip reconstruction to [0, 1] range (model output may be outside this range)
        rec_clipped = torch.clamp(rec, 0.0, 1.0)

        # SSIM expects images in [0, 1] range
        ssim = structural_similarity_index_measure(rec_clipped, img_norm)
        return ssim

    def calc_l1(self, img, rec):
        """
        Calculate L1 distance (Mean Absolute Error).
        
        Args:
            img: [..., C, ...] original image (expected to be in [0, 1] range)
            rec: [..., C, ...] reconstructed image (will be clipped to [0, 1])
        
        Returns:
            l1: scalar tensor
        """
        # Clip reconstruction to [0, 1] range (model output may be outside this range)
        # Original image is kept as is (should already be in [0, 1] range)
        rec_clipped = torch.clamp(rec, 0.0, 1.0)
        l1 = torch.mean(torch.abs(img - rec_clipped))
        return l1


def clear_nested_dict_or_list(d):
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
            clear_nested_dict_or_list(v)
        else:
            try:
                assert type(v) is torch.Tensor or type(v) is str or type(v) is np.ndarray, type(v)
            except Exception as e:
                breakpoint()
        del d[k]


class MetricStore(nn.Module):
    def __init__(self):
        super().__init__()
        self.name2metric = OrderedDict()
        self.name2value_other = OrderedDict()

    def add_metric(self, name, value):
        if name not in self.name2metric:
            self.name2metric[name] = []
        if isinstance(value, list):
            self.name2metric[name].extend(value)
        else:
            self.name2metric[name].append(value)

    def add_other(self, name, value):
        if name not in self.name2value_other:
            self.name2value_other[name] = []
        self.name2value_other[name].append(value)


class PatientStore:
    def __init__(self):
        self.patient2name2tensors = {}

    def add_tensor(self, patient, name, tensor):
        self.patient2name2tensors.setdefault(patient, {}).setdefault(name, []).append(tensor)

    def aggregate_tensors(self):
        for patient, name2tensors in self.patient2name2tensors.items():
            for name, tensors in name2tensors.items():
                self.patient2name2tensors[patient][name] = torch.stack(tensors)
                # if len(tensors) > 1:
                #     self.patient2name2tensors[patient][name] = torch.stack(tensors)
                # else:
                #     self.patient2name2tensors[patient][name] = tensors[0]

    def clear(self):
        clear_nested_dict_or_list(self.patient2name2tensors)
        del self.patient2name2tensors
        self.patient2name2tensors = {}


class Base(nn.Module):
    def __init__(self, cfg: Ocfg):
        super().__init__()
        self.cfg = cfg
        self.cfg.var.obj_model = self

        # common losses
        self.calc_loss_dice = monai.losses.DiceLoss(include_background=True, softmax=True, squared_pred=False,
                                                    jaccard=False, reduction='mean', smooth_nr=1e-05, smooth_dr=1e-05,
                                                    batch=False, weight=None)

        if self.cfg.model.losses.ce.ws_class is not None:
            ws = torch.tensor(self.cfg.model.losses.ce.ws_class)
            assert ws[0] == 1, 'background class should always have weight 1; ths weights are pre-normalized'
            ws = ws / ws.sum()
        else:
            ws = None
        self.calc_loss_ce = torch.nn.CrossEntropyLoss(weight=ws, ignore_index=-100, reduction='mean',
                                                      label_smoothing=0.0)

        self.patient_store = PatientStore()

    def forward(self, data):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.cfg.var.obj_operator.device)
        domains = []
        for domain in ['source', 'target']:
            if f'img_{domain}' in data:
                if self.cfg.exp.source_only and domain == 'target':
                    del data[f'img_target']
                    del data[f'seg_target']
                else:
                    domains.append(domain)
        assert len(domains) > 0
        self.cfg.var.domains = domains

        for domain in domains:
            self.cfg.var.n_samples = data[f'img_{domain}'].shape[1]

        return data

    def loss_dice_ce(self, data, output):
        loss_dice = self.loss_dice(data, output)
        loss_ce = self.loss_ce(data, output)
        ws = self.cfg.model.losses.dice_ce.ws
        return loss_dice * ws[0] + loss_ce * ws[1]

    def loss_dice(self, data, output):
        input = output['logits_source'] # [B, K, ...], not normalized
        target = data['seg_source'] # [B, 1, ...], values in [0, ..., n_classes-1]
        target = seg2onehot(target, n_classes=input.shape[1], dim_class=1) # [B, K, ...]
        loss = self.calc_loss_dice(input, target)
        return loss

    def loss_ce(self, data, output):
        input = output['logits_source'] # [B, K, ...], not normalized
        target = data['seg_source'] # [B, 1, ...], values in [0, ..., n_classes-1]
        target = target.squeeze(1).long() # [B, ...]
        loss = self.calc_loss_ce(input, target)
        return loss

    def before_epoch(self, mode='train', i_repeat=0):
        self.mode = mode
        self.metrics_epoch = OrderedDict()
        if mode == 'train':
            self.current_epoch = i_repeat

        if mode in ['val', 'test']:

            self.metric_calculator = MetricCalculator(self.cfg.dataset.n_classes)
            self.metric_store = MetricStore()
            self.patient_store.clear()

    def after_epoch(self, mode='train'):
        with torch.no_grad():
            for k, v in self.metrics_epoch.items():
                self.metrics_epoch[k] = v / len(getattr(self.cfg.var.obj_operator, f'{mode}_set'))

            if mode in ['val', 'test']:
                names_metric = ['dice']
                if self.cfg.exp.mode == 'test' and self.cfg.dataset.name != 'harborview':
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
                    # for domain in ['source', 'target']:
                    # if f'gt_{domain}' not in name2tensors:
                    # continue
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
                                                               spacing=spacing, img=img, rec=rec)
                    for name_metric, values in name2value_metric.items():
                        # Segmentation metrics return [K], reconstruction metrics return scalar
                        if name_metric in ['psnr', 'ssim', 'l1']:
                            # Reconstruction metrics: scalar value
                            self.metric_store.add_metric(f'{name_metric}_{domain}', values.item())
                        else:
                            # Segmentation metrics: [K] array
                            values = values[1:] # Skip background class
                            if isinstance(values, np.ndarray):
                                values = torch.from_numpy(values)
                            for i, value in enumerate(values):
                                name_class = self.cfg.dataset.idx2name_class[str(i + 1)]
                                self.metric_store.add_metric(f'{name_metric}_{name_class}_{domain}', value.item())
                            self.metric_store.add_metric(f'{name_metric}_mean_{domain}', torch.mean(values).item())
                    self.metric_store.add_other(f'idx_patient_{domain}', patient)

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

    def get_metrics(self, data, output, mode='train'):
        """
            loss_final: used for backward training
            metric_final: used to select best model (higher is better)
            other metrics: for visualization
        """
        self.metrics_iter = OrderedDict()
        if self.training:
            self.metrics_iter['loss_final'] = 0.
            for name_loss, w in self.cfg.model.ws_loss.items():
                if w > 0.:
                    w_current_epoch = getattr(self, 'name_loss2w_current_epoch', {}).get(name_loss, w)
                    loss = getattr(self, f'loss_{name_loss}')(data, output)
                    self.metrics_iter[f'loss_{name_loss}'] = loss.item()
                    self.metrics_iter['loss_final'] += w_current_epoch * loss

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

                for domain in self.cfg.var.domains:
                    patient_slices = data[f'patient_slice_{domain}']
                    logits = output[f'logits_{domain}'] # [B, K, ...]
                    gts = data[f'seg_{domain}'] # [B, 1, ...]
                    imgs = data[f'img_{domain}'] # [B, n_adj*2+1, ...]
                    assert len(patient_slices) == logits.shape[0] == gts.shape[0]
                    for i, (patient_slice, gt, logit) in enumerate(zip(patient_slices, gts, logits)):
                        patient = patient_slice.split('_')[0]
                        if mode == 'test':
                            gt = gt.cpu()
                            logit = logit.cpu()
                        self.patient_store.add_tensor(f'{domain}_patient_{patient}', f'gt_{domain}', gt)
                        self.patient_store.add_tensor(f'{domain}_patient_{patient}', f'logits_{domain}', logit)

                        # Store image and reconstruction for reconstruction metrics
                        needs_img_save = (mode == 'test' and hasattr(self.cfg.exp.test, 'save')
                                          and getattr(self.cfg.exp.test.save, 'img_ori', False))
                        if needs_recon_metrics or needs_img_save:
                            # Store image (for reconstruction metrics or saving)
                            img = imgs[i] # [n_adj*2+1, ...]
                            img_center = img[img.shape[0] // 2:img.shape[0] // 2 + 1] # [1, ...]
                            img_center = img_center.cpu()
                            self.patient_store.add_tensor(f'{domain}_patient_{patient}', f'img_{domain}', img_center)

                            # Store reconstruction if available and needed for metrics
                            if needs_recon_metrics and f'imgs_rec_{domain}' in output:
                                rec = output[f'imgs_rec_{domain}'][i] # [C, ...]
                                # Extract first channel if multiple channels (for learned scale, second channel is scale)
                                if rec.shape[0] > 1:
                                    rec = rec[:1] # [1, ...]
                                rec = rec.cpu()
                                self.patient_store.add_tensor(f'{domain}_patient_{patient}', f'imgs_rec_{domain}', rec)

            for k, v in self.metrics_iter.items():
                self.metrics_epoch[k] = self.metrics_epoch.get(k, 0.) + float(v) * self.cfg.var.n_samples
        return self.metrics_iter

    def vis(self, writer, global_step, data, output, mode, in_epoch):
        if self.cfg.exp.mode != 'test':
            return
        if in_epoch:
            return
