import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import os
import glob
import pandas as pd
import monai.transforms as mt
from torchvision.transforms.functional import equalize


class MSCMRData(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.domain2idx2img = {} # domain -> idx -> [N, H, W]
        self.domain2n_slices = {} # domain -> N
        self.domain2idx2seg = {} # domain -> idx -> [N, H, W]
        self.domain2i_slice_all2idx_i_slice = {} # domain -> i_slice_all -> (idx, i_slice)
        self.domain2idxs = {} # domain -> [idx]
        self.domain2idx2spacing = {} # domain -> idx -> [3]

        assert str(cfg.dataset.version) == '1.1'
        path_folder = './data/mscmr'
        self.dataset_ver = '1.1'
        self.dataset_id = 'ffed4437a23247db90e1284378bee7cc'
        print(f'Dataset ID: {self.dataset_id}, version: {self.dataset_ver}')
        self.path_folder = path_folder

        domains = ['source', 'target']

        for domain in domains:
            idxs = eval(cfg.dataset[domain].range_idx[mode])
            mod = cfg.dataset[domain].mod
            idx2img, idx2spacing = self.read_imgs(self.path_folder, mod, idxs, is_label=False)
            if cfg.dataset.preprocess.equalize:
                for idx, img in idx2img.items():
                    img = img[:, None] # [N, 1, H, W]
                    img = (img - img.min()) / (img.max() - img.min()) * 255
                    img = torch.tensor(img, dtype=torch.uint8)
                    img = equalize(img).numpy().astype(np.float32)
                    idx2img[idx] = img[:, 0]
            idx2img = self.normalize_imgs(idx2img, mod)
            idx2seg, _ = self.read_imgs(self.path_folder, cfg.dataset[domain].mod, idxs, is_label=True)

            self.domain2idx2img[domain] = idx2img
            self.domain2idx2seg[domain] = idx2seg
            self.domain2n_slices[domain] = np.sum([img.shape[0] for img in idx2img.values()])
            self.domain2idxs[domain] = list(idx2img.keys())
            self.domain2idx2spacing[domain] = idx2spacing

        assert self.cfg.dataset.hw_img == 192
        for domain in domains:
            i_slice_all = 0
            self.domain2i_slice_all2idx_i_slice[domain] = {}
            for idx, img in self.domain2idx2img[domain].items():
                for i_slice in range(img.shape[0]):
                    self.domain2i_slice_all2idx_i_slice[domain][i_slice_all] = (idx, i_slice)
                    i_slice_all += 1

        self.init_transform()

    def init_transform(self):
        if self.mode == 'train':
            trans = []
            cfg = self.cfg.dataset.aug
            if isinstance(cfg.crop, int):
                crop = cfg.crop
            else:
                assert len(cfg.crop) == 2
                assert cfg.crop[0] == cfg.crop[1]
                crop = cfg.crop[0]

            if crop > 0:
                trans += [
                    mt.RandSpatialCropd(keys=['img_seg_source'], roi_size=[crop, crop], random_center=True,
                                        random_size=False, allow_missing_keys=True),
                ]
                if cfg.crop_target:
                    trans += [
                        mt.RandSpatialCropd(keys=['img_seg_target'], roi_size=[crop, crop], random_center=True,
                                            random_size=False, allow_missing_keys=True),
                    ]
                else:
                    trans += [
                        mt.CenterSpatialCropd(keys=['img_seg_target'], roi_size=[crop, crop], allow_missing_keys=True),
                    ]

            trans += [mt.SplitDimd(keys=['img_seg_source'], allow_missing_keys=True)]
            trans += [mt.SplitDimd(keys=['img_seg_target'], allow_missing_keys=True)]

            self.transform = mt.Compose(trans)
        else:
            cfg = self.cfg.dataset.aug
            if isinstance(cfg.center_crop, int):
                crop = cfg.center_crop
            else:
                assert len(cfg.center_crop) == 2
                assert cfg.center_crop[0] == cfg.center_crop[1]
                crop = cfg.center_crop[0]
            self.transform = mt.CenterSpatialCropd(keys=['img_source', 'img_target', 'seg_source', 'seg_target'],
                                                   roi_size=[crop, crop], allow_missing_keys=True)

    def read_imgs(self, path_folder, mod, idxs, is_label):
        if idxs == 'all':
            paths = glob.glob(os.path.join(path_folder, mod, f'patient*.nii'))
            idxs = set()
            for path in paths:
                name_nii = os.path.basename(path)
                idx = int(name_nii.split('_')[0].split('patient')[1])
                idxs.add(idx)
            idxs = list(idxs)
            idxs.sort()

        idx2arr = {}
        idx2spacing = {}
        for idx in idxs:
            path = glob.glob(os.path.join(path_folder, mod, f'patient{idx}_*.nii'))
            assert len(path) == 2, (path_folder, mod, idx, path)
            if 'manual' in path[0]:
                path = [path[1], path[0]]
            assert 'manual' not in path[0]
            assert 'manual' in path[1]
            if is_label:
                path = path[1]
            else:
                path = path[0]
            nii = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(nii).astype(np.float32) # [N, H, W]
            assert arr.ndim == 3
            assert arr.shape[1] == arr.shape[2] == self.cfg.dataset.hw_img

            if is_label:
                arr[arr == 200] = 1
                arr[arr == 500] = 2
                arr[arr == 600] = 3
                assert np.allclose(np.sort(np.unique(arr)), [0, 1, 2, 3]), np.unique(arr)
            else:
                spacing = nii.GetSpacing()[::-1]
                assert len(spacing) == 3
                assert np.allclose(spacing[1:], [0.76, 0.76])
                idx2spacing[idx] = spacing
            idx2arr[idx] = arr
        return idx2arr, idx2spacing

    def normalize_imgs(self, idx2img, mod):
        def normalize(array, mode, min=None, max=None, mean=None, std=None):
            if mode == 'min-max':
                if min is None:
                    min = array.min()
                if max is None:
                    max = array.max()
                array = (array-min) / (max-min)
            elif mode == 'z-score':
                if mean is None:
                    mean = array.mean()
                if std is None:
                    std = array.std()
                array = (array-mean) / std
            elif mode == '0-max':
                if max is None:
                    max = array.max()
                array = np.clip(array, 0, None) / max
            else:
                assert min is not None
                assert max is not None
                percentiles = mode.split('-')
                assert len(percentiles) == 2
                for percentile in percentiles:
                    assert 0 <= int(percentile) <= 100
                array = np.clip(array, min, max).astype(np.float32)
                array = (array-min) / (max-min)
                array = array.astype(np.float32)
            return array

        cfg_norm = self.cfg.dataset.preprocess.normalize
        assert isinstance(cfg_norm.enable, bool)
        if not cfg_norm.enable:
            return idx2img

        if cfg_norm.wise in ['subject', 'dataset']:
            df = self.get_stats(mod)
        else:
            assert cfg_norm.wise == 'slice'
            df = None

        idx2img_norm = {}
        for idx, img in idx2img.items():
            if cfg_norm.wise == 'subject':
                min, max, mean, std = (df.loc[f'patient_{idx}', col] for col in ['min', 'max', 'mean', 'std'])
                if cfg_norm.mode not in ['min-max', 'z-score', '0-max']:
                    percentiles = cfg_norm.mode.split('-')
                    assert len(percentiles) == 2
                    percentiles = [int(percentile) for percentile in percentiles]
                    min = np.percentile(img, percentiles[0])
                    max = np.percentile(img, percentiles[1])
            elif cfg_norm.wise == 'dataset':
                min, max, mean, std = (df.loc['all_patients', col] for col in ['min', 'max', 'mean', 'std'])
                assert cfg_norm.mode in ['min-max', 'z-score', '0-max']
            elif cfg_norm.wise == 'slice':
                assert img.ndim == 3
                assert img.shape[1] == img.shape[2]
                if cfg_norm.mode in ['min-max', 'z-score', '0-max']:
                    min = np.min(img, axis=(1, 2), keepdims=True)
                    max = np.max(img, axis=(1, 2), keepdims=True)
                else:
                    percentiles = cfg_norm.mode.split('-')
                    assert len(percentiles) == 2
                    percentiles = [int(percentile) for percentile in percentiles]
                    min = np.percentile(img, percentiles[0], axis=(1, 2), keepdims=True)
                    max = np.percentile(img, percentiles[1], axis=(1, 2), keepdims=True)
                mean = np.mean(img, axis=(1, 2), keepdims=True)
                std = np.std(img, axis=(1, 2), keepdims=True)
            else:
                raise ValueError(cfg_norm.wise)
            idx2img_norm[idx] = normalize(img, cfg_norm.mode, min, max, mean, std)
        return idx2img_norm

    def get_stats(self, mod):
        ver = self.dataset_ver
        id = self.dataset_id
        path_csv = f'./data/mscmr/{ver}_{id}/stats_{mod}.csv'
        if not os.path.exists(path_csv):
            self.extract_stats(mod)
        df = pd.read_csv(path_csv, index_col=0)
        return df

    def extract_stats(self, mod):
        def get_stats_by_img(idx2img):
            cols = {
                'mean': [],
                'std': [],
                'min': [],
                'max': [],
            }
            rows = ['all_patients']
            imgs_all = np.concatenate(list(idx2img.values()))
            cols['mean'].append(imgs_all.mean())
            cols['std'].append(imgs_all.std())
            cols['min'].append(imgs_all.min())
            cols['max'].append(imgs_all.max())

            for idx, img in idx2img.items():
                rows.append(f'patient_{idx}')
                cols['mean'].append(img.mean())
                cols['std'].append(img.std())
                cols['min'].append(img.min())
                cols['max'].append(img.max())

            df = pd.DataFrame(cols, index=rows)
            return df

        idx2img = self.read_imgs(self.path_folder, mod, 'all', is_label=False) # idx -> [P, H, W]
        df_stats = get_stats_by_img(idx2img)
        ver = self.dataset_ver
        id = self.dataset_id
        if not os.path.exists(f'./data/mscmr/{ver}_{id}'):
            os.makedirs(f'./data/mscmr/{ver}_{id}')
        df_stats.to_csv(f'./data/mscmr/{ver}_{id}/stats_{mod}.csv')

    def __len__(self):
        if self.cfg.exp.source_only:
            return self.domain2n_slices['source']
        else:
            if self.mode == 'train':
                return self.domain2n_slices['target']
            else:
                return self.domain2n_slices['target'] + self.domain2n_slices.get('source', 0)

    def __getitem__(self, idx):
        """
        Returns: data (dict)
            when train:
                img_source: [1, H, W]
                seg_source: [1, H, W]
                img_target: [1, H, W]
                seg_target: [1, H, W]
                patient_slice_source: '{idx}_{i_slice}'
                patient_slice_target: '{idx}_{i_slice}'
            when val or test:
                img_{domain}: [N, H, W] # domain = 'source' or 'target'
                patient: '{idx}'
        """

        if self.mode == 'train':
            idx_, i_slice = self.domain2i_slice_all2idx_i_slice['target'][idx]
            img_target = self.domain2idx2img['target'][idx_][i_slice] # [H, W]
            seg_target = self.domain2idx2seg['target'][idx_][i_slice] # [H, W]
            img_seg_target = np.stack([img_target, seg_target]) # [2, H, W]
            patient_slice_target = f'{idx_}_{i_slice}'
            pos_slice_target = i_slice / (self.domain2idx2img['target'][idx_].shape[0] - 1)

            # idxs_to_sample_source = getattr(self.cfg.var.obj_operator, 'idxs_to_sample_source', None)
            # if idxs_to_sample_source is None or len(idxs_to_sample_source) == 0:
            #     idxs_to_sample_source = np.arange(self.domain2n_slices['source']).tolist()
            #     np.random.shuffle(idxs_to_sample_source)
            #     self.cfg.var.obj_operator.idxs_to_sample_source = idxs_to_sample_source
            # idx_source = self.cfg.var.obj_operator.idxs_to_sample_source.pop()
            idx_source = np.random.randint(self.domain2n_slices['source'])
            idx_, i_slice = self.domain2i_slice_all2idx_i_slice['source'][idx_source]
            img_source = self.domain2idx2img['source'][idx_][i_slice] # [H, W]
            seg_source = self.domain2idx2seg['source'][idx_][i_slice] # [H, W]
            img_seg_source = np.stack([img_source, seg_source]) # [2, H, W]
            patient_slice_source = f'{idx_}_{i_slice}'
            pos_slice_source = i_slice / (self.domain2idx2img['source'][idx_].shape[0] - 1)

            data = {
                'img_seg_source': img_seg_source, # [2, H, W]
                'img_seg_target': img_seg_target, # [2, H, W]
                'patient_slice_source': patient_slice_source, # str
                'patient_slice_target': patient_slice_target, # str
            }

            if getattr(self, 'transform', None) is not None:
                data = self.transform(data)
            data_ = {}
            for key, value in data.items():
                if key == 'img_seg_source_0':
                    data_['img_source'] = value
                elif key == 'img_seg_source_1':
                    data_['seg_source'] = value
                elif key == 'img_seg_target_0':
                    data_['img_target'] = value
                elif key == 'img_seg_target_1':
                    data_['seg_target'] = value
                elif key.startswith('patient'):
                    data_[key] = value
            data = data_

            data['pos_slice_source'] = torch.tensor(pos_slice_source, dtype=torch.float32)
            data['pos_slice_target'] = torch.tensor(pos_slice_target, dtype=torch.float32)

        else:
            if self.cfg.exp.source_only:
                domain = 'source'
            else:
                if idx < self.domain2n_slices['target']:
                    domain = 'target'
                else:
                    domain = 'source'
                    idx = idx - self.domain2n_slices['target']

            idx_, i_slice = self.domain2i_slice_all2idx_i_slice[domain][idx]
            img = self.domain2idx2img[domain][idx_][i_slice][None] # [1, H, W]
            seg = self.domain2idx2seg[domain][idx_][i_slice][None] # [1, H, W]
            data = {
                f'img_{domain}': img, # [1, H, W]
                f'seg_{domain}': seg, # [1, H, W]
                f'patient_slice_{domain}': f'{idx_}_{i_slice}', # str
                f'n_slices_patient_{domain}': str(self.domain2idx2img[domain][idx_].shape[0]), # str
            }
            if getattr(self, 'transform', None) is not None:
                data = self.transform(data)

            poss = i_slice / (self.domain2idx2img[domain][idx_].shape[0] - 1)
            data[f'pos_slice_{domain}'] = poss

        return data

    def get_batch(self, samples):
        data = {}
        keys_to_stack = set()
        for sample in samples:
            for k, v in sample.items():
                data.setdefault(k, []).append(v)
                if isinstance(v, torch.Tensor):
                    keys_to_stack.add(k)
                elif isinstance(v, (float, int)):
                    keys_to_stack.add(k)
                    data[k][-1] = torch.tensor(v, dtype=torch.float32)
                else:
                    assert isinstance(v, str), (k, type(v))

        for k in keys_to_stack:
            data[k] = torch.stack(data[k])

        return data


from monai.data import ThreadDataLoader
from monai.data import Dataset as MonaiDataset


class MSCMR(Dataset):
    def __new__(cls, cfg, mode):
        dataset = MSCMRData(cfg, mode)

        if mode == 'train':
            num_workers = 8
            shuffle = True
        elif mode == 'val':
            num_workers = 6
            shuffle = False
        elif mode == 'test':
            num_workers = 6
            shuffle = False
        else:
            raise ValueError

        data_loader = ThreadDataLoader(dataset, num_workers=num_workers, batch_size=cfg.exp[mode].batch_size,
                                       shuffle=shuffle, multiprocessing_context='spawn', use_thread_workers=True,
                                       collate_fn=dataset.get_batch)
        return dataset, data_loader
