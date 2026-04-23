import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import os
import glob
import pandas as pd
import monai.transforms as mt
from torchvision.transforms.functional import equalize
import yaml


class AMOS22Data(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.domain2idx2img = {} # domain -> idx -> [N, H, W]
        self.domain2idx2box = {} # domain -> idx -> [N, H, W]
        self.domain2idx2idx_patient = {} # domain -> idx -> [N, H, W]
        self.domain2n_slices = {} # domain -> N
        self.domain2idx2seg = {} # domain -> idx -> [N, H, W]
        self.domain2i_slice_all2idx_i_slice = {} # domain -> i_slice_all -> (idx, i_slice)
        self.domain2idxs = {} # domain -> [idx]
        self.domain2idx2spacing = {}

        assert str(cfg.dataset.version) == '1.1'
        path_folder = './data/amos22'
        self.dataset_ver = '1.1'
        self.dataset_id = 'c8aecc8318994a398eb8f51904838f27'
        print(f'Dataset ID: {self.dataset_id}, version: {self.dataset_ver}')
        self.path_folder = path_folder
        with open(os.path.join(self.path_folder, 'info.yaml'), 'r') as f:
            self.info = yaml.load(f, Loader=yaml.FullLoader)

        domains = ['source', 'target']

        for domain in domains:
            idxs = eval(cfg.dataset[domain].range_idx[mode])
            mod = cfg.dataset[domain].mod
            idx2img, idx2box, idx2idx_patient = self.read_imgs(self.path_folder, mod, idxs, is_label=False)
            if cfg.dataset.preprocess.equalize:
                for idx, img in idx2img.items():
                    img = img[:, None] # [N, 1, H, W]
                    img = (img - img.min()) / (img.max() - img.min()) * 255
                    img = torch.tensor(img, dtype=torch.uint8)
                    img = equalize(img).numpy().astype(np.float32)
                    idx2img[idx] = img[:, 0]
            idx2img = self.normalize_imgs(idx2img, mod)
            idx2seg, _, _ = self.read_imgs(self.path_folder, mod, idxs, is_label=True)

            self.domain2idx2img[domain] = idx2img
            self.domain2idx2box[domain] = idx2box
            self.domain2idx2idx_patient[domain] = idx2idx_patient
            self.domain2idx2seg[domain] = idx2seg
            self.domain2n_slices[domain] = np.sum([img.shape[0] for img in idx2img.values()])
            self.domain2idxs[domain] = list(idx2img.keys())
            self.domain2idx2spacing[domain] = {
                idx2idx_patient[idx]: self.info['idx_patient2spacing'][idx2idx_patient[idx]]
                for idx in idxs
            } # this is patient idx

            boxes = np.array(list(idx2box.values())) # [N, 3, 2]
            hws = boxes[:, 1:, 1] - boxes[:, 1:, 0] # [N, 2]
            print(f'box ({domain}, {mode}): ', np.min(hws, axis=0), np.max(hws, axis=0))

            print(mode, domain, 'idxs: ', [self.domain2idx2idx_patient[domain][i] for i in self.domain2idxs[domain]])

        for domain in domains:
            i_slice_all = 0
            self.domain2i_slice_all2idx_i_slice[domain] = {}
            for idx, img in self.domain2idx2img[domain].items():
                for i_slice in range(img.shape[0]):
                    self.domain2i_slice_all2idx_i_slice[domain][i_slice_all] = (idx, i_slice)
                    i_slice_all += 1
        print('domain2n_slices:', self.domain2n_slices)

        self.init_transform()

    def init_transform(self):
        if self.mode == 'train':
            trans = []
            cfg = self.cfg.dataset.aug

            if cfg.crop != -1:
                trans += [
                    mt.RandSpatialCropd(keys=['img_seg_source'], roi_size=[cfg.crop[0], cfg.crop[1]],
                                        random_center=True, random_size=False, allow_missing_keys=True),
                ]
                if not self.cfg.exp.source_only:
                    if cfg.crop_target:
                        trans += [
                            mt.RandSpatialCropd(keys=['img_seg_target'], roi_size=[cfg.crop[0], cfg.crop[1]],
                                                random_center=True, random_size=False, allow_missing_keys=True),
                        ]
                    else:
                        trans += [
                            mt.CenterSpatialCropd(keys=['img_seg_target'], roi_size=[cfg.crop[0], cfg.crop[1]],
                                                  allow_missing_keys=True),
                        ]
                trans += [
                    mt.ResizeWithPadOrCropd(keys=['img_seg_source',
                                                  'img_seg_target'], spatial_size=[cfg.crop[0], cfg.crop[1]],
                                            mode='constant', value=0., allow_missing_keys=True),
                ]
            self.transform = mt.Compose(trans)
        else:
            cfg = self.cfg.dataset.aug
            self.transform = mt.Compose([
                mt.ResizeWithPadOrCropd(keys=['img_source', 'img_target', 'seg_source',
                                              'seg_target'], spatial_size=[cfg.crop[0], cfg.crop[1]], mode='constant',
                                        value=0., allow_missing_keys=True),
            ])

    def read_imgs(self, path_folder, mod, idxs, is_label):
        n_select = self.cfg.dataset.mods[mod].select_patients.num
        mod2idxs_patient = {
            'mr': [548, 559, 552, 532, 583, 572, 592, 517, 593, 540, 510, 585, 555, 556, 584, 541, 573, 563, 550, 595, 508, 549, 553, 575, 507, 561, 551, 589, 554, 558],
            'ct': [47, 401, 162, 149, 370, 1, 136, 377, 287, 8, 217, 180, 89, 171, 191, 383, 286, 356, 192, 207, 404, 281, 193, 316, 41, 390, 198, 98, 50, 297, 282, 396, 293, 60, 153],
        }
        assert len(mod2idxs_patient[mod]) == n_select, (mod, len(mod2idxs_patient[mod]), n_select)
        idxs_patient = mod2idxs_patient[mod]
        paths_img = [glob.glob(os.path.join(path_folder, 'imgs_npy', f'{mod}*{idx:04d}.npy'))[0] for idx in idxs_patient]

        if idxs == 'all':
            idxs = np.arange(len(paths_img))

        idx2arr = {}
        idx2box = {}
        idx2idx_patient = {}
        for idx in idxs:
            path = paths_img[idx]
            if is_label:
                path = path.replace('imgs_npy', 'segs_npy')
            arr = np.load(path).astype(np.float32) # [D, H, W]
            assert arr.ndim == 3
            idx2arr[idx] = arr

            box = self.info['idx_patient2box'][idxs_patient[idx]]
            idx2box[idx] = box
            idx2idx_patient[idx] = idxs_patient[idx]
        return idx2arr, idx2box, idx2idx_patient

    def normalize_imgs(self, idx2img, mod):
        cfg_norm = self.cfg.dataset.preprocess.normalize[mod]
        assert isinstance(cfg_norm.enable, bool)
        if not cfg_norm.enable:
            return idx2img

        idx2img_norm = {}
        for idx, img in idx2img.items():
            assert img.ndim == 3
            if cfg_norm.wise == 'subject':
                if cfg_norm.clip_percent is not None:
                    values_percent = np.percentile(img, cfg_norm.clip_percent)
                    img = np.clip(img, values_percent[0], values_percent[1])
                elif cfg_norm.clip_value is not None:
                    img = np.clip(img, cfg_norm.clip_value[0], cfg_norm.clip_value[1])

                if cfg_norm.mode == 'min-max':
                    min_ = img.min()
                    max_ = img.max()
                    img = (img-min_) / (max_-min_)
                elif cfg_norm.mode == 'z-score':
                    img = (img - img.mean()) / img.std()
                else:
                    raise ValueError
            elif cfg_norm.wise == 'slice':
                if cfg_norm.clip_percent is not None:
                    values_percent = np.percentile(img, cfg_norm.clip_percent, axis=(1, 2),
                                                   keepdims=True) # [2, D, 1, 1]
                    img = np.clip(img, values_percent[0], values_percent[1])
                elif cfg_norm.clip_value is not None:
                    img = np.clip(img, cfg_norm.clip_value[0], cfg_norm.clip_value[1])

                if cfg_norm.mode == 'min-max':
                    min_ = img.min(axis=(1, 2), keepdims=True)
                    max_ = img.max(axis=(1, 2), keepdims=True)
                    img = (img-min_) / (max_-min_)
                elif cfg_norm.mode == 'z-score':
                    img = (img - img.mean(axis=(1, 2), keepdims=True)) / img.std(axis=(1, 2), keepdims=True)
                else:
                    raise ValueError
            else:
                raise ValueError(cfg_norm.wise)
            idx2img_norm[idx] = img.astype(np.float32)
        return idx2img_norm

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
                img_source: [C, H, W]
                seg_source: [C, H, W]
                img_target: [C, H, W]
                seg_target: [C, H, W]
                patient_slice_source: '{idx}_{i_slice}'
                patient_slice_target: '{idx}_{i_slice}'
            when val or test:
                img_{domain}: [N, H, W] # domain = 'source' or 'target'
                patient: '{idx}'
        """

        n_adj = self.cfg.dataset.n_adj
        if self.mode == 'train':
            if not self.cfg.exp.source_only:
                idx_, i_slice = self.domain2i_slice_all2idx_i_slice['target'][idx]
                i_slices = np.arange(i_slice - n_adj, i_slice + n_adj + 1)
                i_slices = np.clip(i_slices, 0, self.domain2idx2img['target'][idx_].shape[0] - 1)
                img_target = self.domain2idx2img['target'][idx_][i_slices] # [C, H, W]
                seg_target = self.domain2idx2seg['target'][idx_][[i_slice]] # [1, H, W]
                img_seg_target = np.concatenate([img_target, seg_target]) # [C+1, H, W]
                patient_slice_target = f'{idx_}_{i_slice}'
                pos_slice_target = i_slice / (self.domain2idx2img['target'][idx_].shape[0] - 1)

            # idxs_to_sample_source = getattr(self.cfg.var.obj_operator, 'idxs_to_sample_source', None)
            # if idxs_to_sample_source is None or len(idxs_to_sample_source) == 0:
            #     idxs_to_sample_source = np.arange(self.domain2n_slices['source']).tolist()
            #     np.random.shuffle(idxs_to_sample_source)
            #     self.cfg.var.obj_operator.idxs_to_sample_source = idxs_to_sample_source
            # idx_source = self.cfg.var.obj_operator.idxs_to_sample_source.pop()

            if self.cfg.exp.source_only:
                idx_source = idx
            else:
                idx_source = np.random.randint(self.domain2n_slices['source'])
            idx_, i_slice = self.domain2i_slice_all2idx_i_slice['source'][idx_source]
            i_slices = np.arange(i_slice - n_adj, i_slice + n_adj + 1)
            i_slices = np.clip(i_slices, 0, self.domain2idx2img['source'][idx_].shape[0] - 1)
            img_source = self.domain2idx2img['source'][idx_][i_slices] # [C, H, W]
            seg_source = self.domain2idx2seg['source'][idx_][[i_slice]] # [1, H, W]
            img_seg_source = np.concatenate([img_source, seg_source]) # [C+1, H, W]
            patient_slice_source = f'{idx_}_{i_slice}'
            pos_slice_source = i_slice / (self.domain2idx2img['source'][idx_].shape[0] - 1)

            data = {
                'img_seg_source': img_seg_source, # [2*C, H, W]
                # 'img_seg_target': img_seg_target, # [2*C, H, W]
                'patient_slice_source': patient_slice_source, # str
                # 'patient_slice_target': patient_slice_target, # str
            }
            if not self.cfg.exp.source_only:
                data['img_seg_target'] = img_seg_target # [2*C, H, W]
                data['patient_slice_target'] = patient_slice_target # str

            if getattr(self, 'transform', None) is not None:
                data = self.transform(data)
            data_ = {}
            for key, value in data.items():
                if key == 'img_seg_source':
                    data_['img_source'] = value[:-1] # [C, H, W]
                    data_['seg_source'] = value[-1:] # [C, H, W]
                elif key == 'img_seg_target':
                    data_['img_target'] = value[:-1] # [C, H, W]
                    data_['seg_target'] = value[-1:] # [C, H, W]
                elif key.startswith('patient'):
                    data_[key] = value
            data = data_

            data['pos_slice_source'] = torch.tensor(pos_slice_source, dtype=torch.float32)
            if not self.cfg.exp.source_only:
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
            i_slices = np.arange(i_slice - n_adj, i_slice + n_adj + 1)
            i_slices = np.clip(i_slices, 0, self.domain2idx2img[domain][idx_].shape[0] - 1)
            img = self.domain2idx2img[domain][idx_][i_slices] # [C, H, W]
            seg = self.domain2idx2seg[domain][idx_][[i_slice]] # [1, H, W]
            patient = self.domain2idx2idx_patient[domain][idx_]
            data = {
                f'img_{domain}': img, # [C, H, W]
                f'seg_{domain}': seg, # [C, H, W]
                f'patient_slice_{domain}': f'{patient}_{i_slice}', # str
                f'n_slices_patient_{domain}': str(self.domain2idx2img[domain][idx_].shape[0]), # str
                # f'idx': f'{idx_}', # str
            }
            box = np.array(self.domain2idx2box[domain][idx_]) # [3, 2]
            center = (box[1:, 0] + box[1:, 1]) / 2 # [2]
            crop = self.cfg.dataset.aug.center_crop # [2]
            h_min = max(0, int(center[0] - crop[0] // 2))
            h_max = min(img.shape[1], int(center[0] - crop[0] // 2 + crop[0]))
            w_min = max(0, int(center[1] - crop[1] // 2))
            w_max = min(img.shape[2], int(center[1] - crop[1] // 2 + crop[1]))
            img = img[:, h_min:h_max, w_min:w_max]
            seg = seg[:, h_min:h_max, w_min:w_max]
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


class AMOS22(Dataset):
    def __new__(cls, cfg, mode):
        dataset = AMOS22Data(cfg, mode)

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
