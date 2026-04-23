import torch
import torch.nn as nn
import torch.nn.functional as F
from core.networks.basic_nets.att_unet import MyAttentionUnet
from core.networks.blocks.decoder import MyDecoder
from core.networks.blocks.reg import Reg
from core.networks.blocks.stn import SpatialTransformer
from core.networks.blocks.poe import ProductofExperts
from core.networks.blocks.vel2disp import VectorIntegration
from einops import rearrange, repeat
import numpy as np
from core.utils.cache import cache_res, cache_res_by_domain, cache_res_by_domain_lv
from reinmax import reinmax
from core.networks.basic_nets.discriminator import ConvDomainDiscriminator, SegmentationDiscriminator
from core.utils.gradient_reversal import GradientReversal
from core.networks.basic_nets.style_encoder import StyleEncoder
from core.losses.probability_distance import hellinger_distance, fisher_rao_distance
import torch.distributions as tcdist
import monai.transforms as mt
import omegaconf


class Ours2Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if isinstance(cfg.net.encoder.channels, list):
            assert len(cfg.net.encoder.channels) == cfg.net.n_levels
            channels = cfg.net.encoder.channels
        elif isinstance(cfg.net.encoder.channels, int):
            channels = [cfg.net.encoder.channels * 2**l for l in range(cfg.net.n_levels)]
        else:
            raise ValueError(f'Unsupported type for channels: {cfg.net.channels}')
        strides = [2 for _ in range(cfg.net.n_levels - 1)]

        domains = ['source', 'target']
        self.poe = ProductofExperts()
        self.domain2encoder = nn.ModuleDict()
        self.domain2decoder_rec = nn.ModuleDict()
        self.domain_lv2reg = nn.ModuleDict()
        self.lv2stn = nn.ModuleDict()
        self.lv2vel2disp = nn.ModuleDict()
        self.domain2net_c = nn.ModuleDict()
        for domain in domains:
            self.domain2encoder[domain] = MyAttentionUnet(spatial_dims=cfg.dataset.dim,
                                                          in_channels=cfg.dataset.n_adj * 2 + 1,
                                                          out_channels=cfg.dataset.n_classes, channels=channels,
                                                          strides=strides, kernel_size=3, up_kernel_size=3, dropout=0.0,
                                                          norm=cfg.net.encoder.norm_type)
            if cfg.model.losses.recon.scale == 'learned':
                out_channels = 2
            else:
                assert isinstance(cfg.model.losses.recon.scale, (float, int))
                out_channels = 1
            self.domain2decoder_rec[domain] = MyDecoder(cfg, role='decoder_rec', spatial_dims=cfg.dataset.dim,
                                                        channels=[int(c // 2) for c in channels],
                                                        out_channels=out_channels)

            for i, l in enumerate(cfg.net.reg.levels):
                self.domain_lv2reg[f'{domain}_{l}'] = Reg(spatial_dims=cfg.dataset.dim, in_channels=channels[l],
                                                          n_blocks=cfg.net.reg.n_blocks[i])

            if self.cfg.net.atlas.qc.calc.mode.name == 'net':
                if self.cfg.net.atlas.qc.calc.levelwise:
                    if self.cfg.net.atlas.qc.calc.input_includes_atlas:
                        raise NotImplementedError
                    self.domain2net_c[domain] = nn.ModuleDict()
                    for l in range(cfg.net.n_levels):
                        self.domain2net_c[domain][f'{l}'] = nn.Sequential(
                            nn.Linear(channels[l], channels[l] // 2), nn.LeakyReLU(),
                            nn.Linear(channels[l] // 2, cfg.net.atlas.num))
                else:
                    channels_concat = [channels[l] for l in cfg.net.atlas.qc.calc.levels]
                    if self.cfg.net.atlas.qc.calc.input_includes_atlas:
                        channels_concat += [channels[l] * cfg.net.atlas.num for l in cfg.net.atlas.qc.calc.levels]
                    channels_concat = sum(channels_concat)
                    nets = [
                        nn.Linear(channels_concat, channels_concat),
                        nn.LeakyReLU(),
                    ]
                    if self.cfg.net.net_c.dropout > 0.:
                        print(f'Using dropout in net_c: {self.cfg.net.net_c.dropout}')
                        nets.append(nn.Dropout(self.cfg.net.net_c.dropout))
                    nets.append(nn.Linear(channels_concat, cfg.net.atlas.num))
                    self.domain2net_c[domain] = nn.Sequential(*nets)

        if self.cfg.model.ws_loss.adv_enc_feat > 0.:
            if self.cfg.model.losses.adv_enc_feat.mode == 'levelwise':
                self.lv2discriminator_enc_feat = nn.ModuleDict()
                for lv in self.cfg.model.losses.adv_enc_feat.levels:
                    self.lv2discriminator_enc_feat[f'{lv}'] = ConvDomainDiscriminator(channels[lv])
            else:
                raise NotImplementedError

        if self.cfg.model.ws_loss.adv_seg > 0.:
            self.discriminator_seg = SegmentationDiscriminator(cfg.dataset.n_classes)

        if self.cfg.net.style_code.enable:
            if self.cfg.net.style_code.adain.classwise:
                n_classes = cfg.dataset.n_classes
            else:
                n_classes = None
            self.encoder_style = StyleEncoder(spatial_dims=cfg.dataset.dim, in_channels=cfg.dataset.n_adj * 2 + 1,
                                              style_dim=cfg.net.style_code.adain.style_dim, n_classes=n_classes)

        if self.cfg.model.ws_loss.pos_slice > 0:
            self.domain2net_pos_slice = nn.ModuleDict()
            self.domain2net_pos_slice['source'] = nn.Linear(channels[-1], 1)
            self.domain2net_pos_slice['target'] = self.domain2net_pos_slice['source']

        self.decoder_seg = MyDecoder(cfg, role='decoder_seg', spatial_dims=cfg.dataset.dim,
                                     channels=[int(c // 2) for c in channels], out_channels=cfg.dataset.n_classes)
        for l in cfg.net.reg.levels:
            hw_img = cfg.dataset.aug.get('crop', cfg.dataset.get('hw_img', None))
            if isinstance(hw_img, (list, tuple, omegaconf.listconfig.ListConfig)):
                assert len(hw_img) == cfg.dataset.dim
                size = list(np.array(hw_img) // 2**l)
            elif isinstance(hw_img, int):
                size = [hw_img // 2**l] * cfg.dataset.dim
            else:
                raise ValueError(f'Unsupported type for hw_img: {cfg.dataset.hw_img}')

            interp_mode = 'bilinear' if cfg.dataset.dim == 2 else 'trilinear'
            self.lv2stn[f'{l}'] = SpatialTransformer(size=size, interp_mode=interp_mode)
            self.lv2vel2disp[f'{l}'] = VectorIntegration(size=size)

        for l in range(cfg.net.n_levels):
            hw_img = cfg.dataset.aug.get('crop', cfg.dataset.get('hw_img', None))
            if isinstance(hw_img, (list, tuple, omegaconf.listconfig.ListConfig)):
                assert len(hw_img) == cfg.dataset.dim
                size = list(np.array(hw_img) // 2**l)
            elif isinstance(hw_img, int):
                size = [hw_img // 2**l] * cfg.dataset.dim
            else:
                raise ValueError(f'Unsupported type for hw_img: {cfg.dataset.hw_img}')
            atlas = torch.randn(cfg.net.atlas.num, channels[l], *size) # [A, C, ...]
            self.register_parameter(f'atlas_lv{l}', nn.Parameter(atlas))

        if cfg.net.encoder.between_domains == 'same':
            del self.domain2encoder['target']
            self.domain2encoder['target'] = self.domain2encoder['source']
        elif cfg.net.encoder.between_domains == 'conv_shared':
            self.make_convs_shared(self.domain2encoder['source'], self.domain2encoder['target'])
            self.make_convs_shared(self.domain2decoder_rec['source'], self.domain2decoder_rec['target'])
        elif cfg.net.encoder.between_domains == 'separate':
            pass
        if cfg.net.net_c.between_domains == 'same':
            del self.domain2net_c['target']
            self.domain2net_c['target'] = self.domain2net_c['source']
        elif cfg.net.net_c.between_domains == 'separate':
            pass
        else:
            raise ValueError(f'Unsupported value for cfg.net.net_c.between_domains: {cfg.net.net_c.between_domains}')

        if cfg.net.reg_affine.enable:
            D = cfg.dataset.dim
            if cfg.net.reg_affine.translation.enable:
                conv_cls = getattr(nn, f'Conv{D}d')
                ch = channels[cfg.net.reg_affine.translation.level_predict]
                self.reg_affine_translation = nn.Sequential(
                    conv_cls(ch, ch, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    conv_cls(ch, 1, kernel_size=1, stride=1, padding=0),
                )
            else:
                self.reg_affine_translation = None
            if cfg.net.reg_affine.zoom.enable:
                n_zoom = 1 if cfg.net.reg_affine.zoom.isotropic else D
                ch = channels[cfg.net.reg_affine.zoom.level_predict]
                hidden_dim = ch
                self.reg_affine_zoom = nn.Sequential(
                    nn.Linear(ch, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, n_zoom),
                )
            else:
                self.reg_affine_zoom = None
            assert (
                self.reg_affine_translation is not None or self.reg_affine_zoom is not None
            ), 'net.reg_affine.enable=True but no translation or zoom enabled'
        else:
            self.reg_affine_translation = None
            self.reg_affine_zoom = None

        if cfg.net.style_code.enable:
            assert cfg.net.decoder_rec.between_domains == 'same'
        if cfg.net.decoder_rec.between_domains == 'same':
            del self.domain2decoder_rec['target']
            self.domain2decoder_rec['target'] = self.domain2decoder_rec['source']
        elif cfg.net.decoder_rec.between_domains == 'conv_shared':
            self.make_convs_shared(self.domain2decoder_rec['source'], self.domain2decoder_rec['target'])
        elif cfg.net.decoder_rec.between_domains == 'separate':
            pass

        for l in cfg.net.reg.levels:
            if cfg.net.reg.between_domains == 'same':
                del self.domain_lv2reg[f'target_{l}']
                self.domain_lv2reg[f'target_{l}'] = self.domain_lv2reg[f'source_{l}']
            elif cfg.net.reg.between_domains == 'conv_shared':
                self.make_convs_shared(self.domain_lv2reg[f'source_{l}'], self.domain_lv2reg[f'target_{l}'])
            elif cfg.net.reg.between_domains == 'separate':
                pass
            else:
                raise ValueError(f'Unsupported value for cfg.reg.between_domains: {cfg.net.reg.between_domains}')

        # print(self.domain2encoder['source'])

    def make_convs_shared(self, net1, net2):
        # shared convs, i.e., domain-specific batch norms
        for name, m in net1.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                m2 = net2.get_submodule(name)
                m2.weight = m.weight
                m2.bias = m.bias # type: ignore

    @cache_res_by_domain('lv2feat')
    def get_lv2feat(self, domain):
        lv2feat = self.domain2encoder[domain.split('_')[0]](self.data[f'img_{domain}']) # l -> [B, C, ...]
        # '{l}' -> [B, C, ...]
        return lv2feat

    @cache_res_by_domain('style_code')
    def get_style_code(self, domain):
        style_code = self.encoder_style(self.data[f'img_{domain}']) # [B, C] or [B, K, C]
        if self.cfg.net.style_code.adain.classwise:
            assert style_code.dim() == 3
            assert style_code.shape[1] == self.cfg.dataset.n_classes
            mask = self.get_logits_seg_reg(domain) # [BxS, K, ...]
            mask = torch.softmax(mask, dim=1) # [BxS, K, ...]
            style_code = {'mask': mask, 'code': style_code}
        return style_code

    @cache_res_by_domain('output_discriminator_enc_feat')
    def get_output_discriminator_enc_feat(self, domain):
        alpha = self.cfg.model.losses.adv_enc_feat.alpha
        if self.cfg.model.losses.adv_enc_feat.mode == 'levelwise':
            lv2feat = self.get_lv2feat(domain)
            lv2out = {}
            for lv in self.cfg.model.losses.adv_enc_feat.levels:
                feat = GradientReversal(alpha=alpha)(lv2feat[f'{lv}'])
                out = self.lv2discriminator_enc_feat[f'{lv}'](feat) # [B, 1]
                lv2out[f'{lv}'] = out # [B, 1]
        else:
            raise NotImplementedError
        return lv2out

    @cache_res_by_domain('output_discriminator_seg')
    def get_output_discriminator_seg(self, domain):
        seg = self.get_logits_seg(domain) # [BxS, K, ...]
        seg = torch.softmax(seg, dim=1) # [BxS, K, ...]
        seg = GradientReversal(alpha=self.cfg.model.losses.adv_seg.alpha)(seg) # [BxS, K, ...]
        out = self.discriminator_seg(seg) # [BxS, 1, ...]
        return out

    @cache_res_by_domain_lv('atlas_rotated')
    def get_atlas_rotated(self, domain, lv, a=None):
        atlas = getattr(self, f'atlas_lv{lv}') # [A, C, ...]
        B = self.data[f'img_{domain}'].shape[0]

        # Apply rotations to atlas if rotation angles exist
        if self.training and (self.cfg.dataset.aug.rand_rotation or self.cfg.dataset.aug.rand_flip):
            assert f'rotation_{domain}' in self.data or f'flip_{domain}' in self.data
            angles = self.data.get(f'rotation_{domain}', [0] * B) # [B]
            flips = self.data.get(f'flip_{domain}', [0] * B) # [B]
            # Create a list of rotated atlases
            auged_atlases = []
            for angle, flip in zip(angles, flips):
                rotated_atlas = atlas
                if angle != 0:
                    rotated_atlas = torch.rot90(rotated_atlas, k=angle // 90, dims=[-2, -1]) # [A, C, ...]
                if flip == 1:
                    rotated_atlas = torch.flip(rotated_atlas, dims=[-1])
                elif flip == 2:
                    rotated_atlas = torch.flip(rotated_atlas, dims=[-2])
                auged_atlases.append(rotated_atlas)
            # Stack the rotated atlases
            atlas = torch.stack(auged_atlases) # [B, A, C, ...]
        else:
            atlas = repeat(atlas, 'A C ... -> B A C ...', B=B) # [B, A, C, ...]

        if a is not None:
            atlas = atlas[:, a] # [B, C, ...]
        return atlas

    @cache_res_by_domain('samples_mix_qc')
    def get_samples_mix_qc(self, domain):
        assert self.cfg.net.atlas.qc.mode.name == 'dirichlet'
        assert not self.cfg.net.atlas.qc.calc.levelwise
        alphas = self.get_params_dist_prob_atlas(domain)['0'] # [B, A]
        dist = tcdist.Dirichlet(alphas) # [B, A]
        N = self.cfg.model.losses.kl_dirichlet_domain_diff.monte_carlo.n_samples_per_dirichlet
        samples = dist.sample((N, )) # [N, B, A]
        samples = rearrange(samples, 'N B A -> (N B) A') # [M, A], M samples from Dirichlet mixture
        return samples

    @cache_res_by_domain('log_prob_samples_mix_qc_same_domain')
    def get_log_prob_samples_mix_qc_same_domain(self, domain):
        assert self.cfg.net.atlas.qc.mode.name == 'dirichlet'
        assert not self.cfg.net.atlas.qc.calc.levelwise
        alphas = self.get_params_dist_prob_atlas(domain)['0'] # [B, A]
        B = alphas.shape[0]
        dist = tcdist.Dirichlet(alphas) # [B, A]
        samples = self.get_samples_mix_qc(domain) # [M, A]
        samples = samples[:, None] # [M, 1, A]
        log_prob = dist.log_prob(samples) # [M, B]
        log_prob = log_prob.logsumexp(dim=1) - np.log(B) # [M]
        return log_prob

    @cache_res_by_domain('log_prob_samples_mix_qc_diff_domain')
    def get_log_prob_samples_mix_qc_diff_domain(self, domain):
        # samples from Dirichlet mixture of domain
        # log prob under Dirichlet mixture of the other domain
        assert self.cfg.net.atlas.qc.mode.name == 'dirichlet'
        assert not self.cfg.net.atlas.qc.calc.levelwise
        if domain == 'source':
            domain_diff = 'target'
        elif domain == 'target':
            domain_diff = 'source'
        else:
            raise ValueError(f'Unsupported domain: {domain}')

        alphas = self.get_params_dist_prob_atlas(domain_diff)['0'] # [B, A]
        B = alphas.shape[0]
        dist = tcdist.Dirichlet(alphas) # [B, A]
        samples = self.get_samples_mix_qc(domain) # [M, A]
        samples = samples[:, None] # [M, 1, A]
        log_prob = dist.log_prob(samples) # [M, B]
        log_prob = log_prob.logsumexp(dim=1) - np.log(B) # [M]
        return log_prob

    @cache_res_by_domain('lv2gap')
    def get_gap_for_params_dist_prob_atlas(self, domain):
        """Compute global average pool (GAP) on encoder features only.
        Returns lv2gap: dict key -> gap tensor [B, C]. Keys are level indices f'{lv}'.
        Atlas GAP is not computed here; callers that need it should compute it themselves.
        """
        lv2feat = self.get_lv2feat(domain)
        if self.cfg.net.atlas.qc.calc.mode.name != 'net':
            raise ValueError(
                f'Unsupported value for cfg.net.atlas.qc.calc.mode.name: {self.cfg.net.atlas.qc.calc.mode.name}')
        if self.cfg.net.atlas.qc.calc.levelwise and self.cfg.net.atlas.qc.calc.input_includes_atlas:
            raise NotImplementedError
        levels = (
            list(range(self.cfg.net.n_levels))
            if self.cfg.net.atlas.qc.calc.levelwise
            else self.cfg.net.atlas.qc.calc.levels
        )
        lv2gap = {}
        for lv in levels:
            out_enc = lv2feat[f'{lv}']  # [B, C, ...]
            gap_out_enc = torch.mean(out_enc, dim=list(range(2, out_enc.dim())))  # [B, C]
            lv2gap[f'{lv}'] = gap_out_enc
        return lv2gap

    @cache_res_by_domain('params_affine_zoom')
    def get_params_affine_zoom(self, domain):
        """Predict zoom parameters from last-stage (channels[-1]) GAP via get_gap_for_params_dist_prob_atlas, then reg_affine_zoom.
        Returns [B, n_zoom] when reg_affine_zoom is enabled, else None.
        """
        if self.reg_affine_zoom is None:
            raise ValueError("net.reg_affine.zoom.enable=False")
        lv2gap = self.get_gap_for_params_dist_prob_atlas(domain)
        lv = self.cfg.net.reg_affine.zoom.level_predict
        gap = lv2gap[f'{lv}']  # [B, C]
        zoom_params = self.reg_affine_zoom(gap)  # [B, n_zoom]

        s_max = 2.4
        log_smax = zoom_params.new_tensor(float(s_max)).log()
        dlog_s = log_smax * torch.tanh(zoom_params)
        zoom = torch.exp(dlog_s).clamp_min(1e-8) # [B, n_zoom], values in [1/s_max, s_max]
        return zoom

    def get_disp_affine_zoom(self, zoom, spatial_shape):
        # zoom: [B, n_zoom]
        H, W = spatial_shape
        B = zoom.shape[0]
        device = zoom.device
        dtype = zoom.dtype

        if zoom.shape[1] == 1:
            si = zoom[:, 0]                      # [B]
            sj = zoom[:, 0]                      # [B]
        elif zoom.shape[1] == 2:
            si = zoom[:, 0]                      # [B]
            sj = zoom[:, 1]                      # [B]
        else:
            raise ValueError(f"Unsupported n_zoom: {zoom.shape[1]}")

        si = rearrange(si, 'b -> b 1 1 1')           # [B,1, 1, 1]
        sj = rearrange(sj, 'b -> b 1 1 1')           # [B,1, 1, 1]

        j = torch.arange(W, device=device, dtype=dtype)
        i = torch.arange(H, device=device, dtype=dtype)
        J = repeat(j, 'w -> h w', h=H)           # [H, W] (j index)
        I = repeat(i, 'h -> h w', w=W)           # [H, W] (i index)

        ci = (H - 1) / 2.0
        cj = (W - 1) / 2.0

        I_out = I                                # [H, W]
        J_out = J                                # [H, W]

        I_in = ci + (I_out - ci) / si   # [B, 1, H, W]
        J_in = cj + (J_out - cj) / sj   # [B, 1, H, W]

        disp_i = I_in - I_out                              # [B, 1, H, W]
        disp_j = J_in - J_out                              # [B, 1, H, W]
        disp_zoom = torch.cat([disp_i, disp_j], dim=1)     # [B,2,H,W] = (di,dj)
        return disp_zoom


    @cache_res_by_domain('params_affine_translation')
    def get_params_affine_translation(self, domain):
        """Predict translation from second-to-last stage feature via reg_affine_translation.
        Returns [B, 1, H, W] (2D) or [B, 1, D, H, W] (3D) when enabled, else None.
        """
        if self.reg_affine_translation is None:
            raise ValueError("net.reg_affine.translation.enable=False")
        lv2feat = self.get_lv2feat(domain)
        lv = self.cfg.net.reg_affine.translation.level_predict
        feat = lv2feat[f'{lv}']  # [B, C, ...]
        out = self.reg_affine_translation(feat)  # [B, 1, ...]


        B = out.shape[0]
        assert out.dim() == 4 or out.dim() == 5, f'out.shape: {out.shape}'
        device = out.device
        dtype = out.dtype

        tau = 1.0 

        if out.dim() == 4:
            # out: [B, 1, H, W]
            B, _, H, W = out.shape
            device = out.device
            dtype = out.dtype

            tau = 1.0
            eps = 1e-8
            assert out.shape[1] == 1, f'out.shape: {out.shape}'
            logits = rearrange(out / tau, 'b c h w -> b (c h w)')
            P = torch.softmax(logits, dim=1).clamp_min(eps)          # [B, H*W]
            P = rearrange(P, 'b (h w) -> b 1 h w', h=H, w=W)        # [B, 1, H, W]
            j = torch.arange(W, device=device, dtype=dtype)
            i = torch.arange(H, device=device, dtype=dtype)

            J = repeat(j, 'w -> h w', h=H)                           # [H, W]
            I = repeat(i, 'h -> h w', w=W)                           # [H, W]

            grid = torch.stack([I, J], dim=0)                        # [2, H, W]
            grid = rearrange(grid, 'c h w -> 1 c h w')               # [1, 2, H, W]
            Eij = (P * grid).sum(dim=[2, 3])                         # [B, 2]
            Ei, Ej = Eij.split(1, dim=1) # [B, 1], [B, 1]
            di = Ei - (H - 1) / 2.0
            dj = Ej - (W - 1) / 2.0
            dij = torch.cat([di, dj], dim=1) # [B, 2]
        else:
            raise NotImplementedError(f'Unsupported out.dim(): {out.dim()}')
        return dij

    @cache_res_by_domain('params_dist_prob_atlas')
    def get_params_dist_prob_atlas(self, domain):
        def convert_logits_to_param_dist(logits):
            if self.cfg.net.atlas.qc.mode.name == 'deterministic':
                return logits
            elif self.cfg.net.atlas.qc.mode.name == 'dirichlet':
                return F.softplus(logits, beta=1) + 1.
            else:
                raise NotImplementedError(
                    f'Unsupported value for cfg.net.atlas.qc.mode.name: {self.cfg.net.atlas.qc.mode.name}')

        lv2gap = self.get_gap_for_params_dist_prob_atlas(domain)
        if self.cfg.net.atlas.qc.calc.mode.name == 'net':
            if self.cfg.net.atlas.qc.calc.levelwise:
                lv2param_dist = {}
                for lv in range(self.cfg.net.n_levels):
                    logits = self.domain2net_c[domain.split('_')[0]][f'{lv}'](lv2gap[f'{lv}'])  # [B, A]
                    lv2param_dist[f'{lv}'] = convert_logits_to_param_dist(logits) # [B, A]
            else:
                gaps = []
                for lv in self.cfg.net.atlas.qc.calc.levels:
                    gaps.append(lv2gap[f'{lv}'])
                    if self.cfg.net.atlas.qc.calc.input_includes_atlas:
                        atlas = self.get_atlas_rotated(domain, lv) # [B, A, C, ...]
                        gap_atlas = torch.mean(atlas, dim=list(range(3, atlas.dim()))) # [B, A, C]
                        gap_atlas = gap_atlas.reshape(gap_atlas.shape[0], -1) # [B, A*C]
                        gaps.append(gap_atlas)
                gap_concat = torch.cat(gaps, dim=1)  # [B, C]
                logits = self.domain2net_c[domain.split('_')[0]](gap_concat)  # [B, A]
                param_dist = convert_logits_to_param_dist(logits)  # [B, A]
                lv2param_dist = {f'{lv}': param_dist for lv in range(self.cfg.net.n_levels)}
        else:
            raise ValueError(
                f'Unsupported value for cfg.net.atlas.qc.calc.mode.name: {self.cfg.net.atlas.qc.calc.mode.name}')
        return lv2param_dist

    @cache_res_by_domain('probs_a_before_post_process')
    def get_probs_atlas_before_post_process(self, domain):
        lv2param_dist = self.get_params_dist_prob_atlas(domain) # [B, A]
        lvs_all = list(lv2param_dist.keys())
        if self.cfg.net.atlas.qc.calc.levelwise:
            lvs_calc = lvs_all
        else:
            lvs_calc = [lvs_all[0]]
        lv2prob = {}
        for lv in lvs_calc:
            param_dist = lv2param_dist[lv] # [B, A]
            if self.cfg.net.atlas.qc.mode.name == 'deterministic':
                # dist is just the logits output by the network
                prob = torch.softmax(param_dist, dim=1) # [B, A]
            elif self.cfg.net.atlas.qc.mode.name == 'dirichlet':
                # dist is parameters of Dirichlet distributions
                dist = tcdist.Dirichlet(param_dist) # [B, A]
                if self.training:
                    prob = dist.rsample() # [B, A]
                else:
                    prob = dist.mean # [B, A]
            else:
                raise NotImplementedError(
                    f'Unsupported value for cfg.net.atlas.qc.mode.name: {self.cfg.net.atlas.qc.mode.name}')
            lv2prob[lv] = prob

        if not self.cfg.net.atlas.qc.calc.levelwise:
            for lv in lvs_all[1:]:
                lv2prob[lv] = lv2prob[lvs_calc[0]]

        # [B, A]
        return lv2prob

    @cache_res_by_domain('probs_a')
    def get_probs_atlas(self, domain):
        def post_process_probs(probs):
            probs_source_closest = None
            dists = None
            if self.training and self.cfg.net.atlas.soft_idx_add_noise.enable:
                std = self.cfg.net.atlas.soft_idx_add_noise.std
                std = 0.02
                noise = torch.randn_like(probs) * std
                probs = probs + noise
                probs = probs / probs.sum(dim=1, keepdim=True)
            probs = torch.clamp(probs, min=0., max=1.)

            if self.cfg.net.atlas.qc.calc.quantize_target and domain == 'target':
                assert self.cfg.model.losses.kl_c_domain_diff.queue_size > 0
                probs_source = self.cfg.var.obj_model.queue_qc_source # [N, A]
                if probs_source.shape[0] > 0:
                    dist = fisher_rao_distance(probs[None], probs_source[None]) # [1, B, N]
                    assert dist.shape[0] == 1
                    dist = dist[0] # [B, N]
                    idxs = dist.argmin(dim=1) # [B]
                    dists = dist[torch.arange(dist.shape[0]), idxs] # [B]
                    # straight-through estimator
                    probs_source_closest = probs_source[idxs] # [B, A]
                    probs = (probs_source_closest - probs).detach() + probs # [B, A]
            return probs, probs_source_closest, dists

        lv2prob = self.get_probs_atlas_before_post_process(domain) # [B, A]
        lvs_all = list(lv2prob.keys())
        if self.cfg.net.atlas.qc.calc.levelwise:
            lvs_calc = lvs_all
        else:
            lvs_calc = [lvs_all[0]]
        lv2prob_post = {}
        for lv in lvs_calc:
            lv2prob_post[lv], probs_source_closest, dists = post_process_probs(lv2prob[lv])

        if not self.cfg.net.atlas.qc.calc.levelwise:
            for lv in lvs_all[1:]:
                lv2prob_post[lv] = lv2prob_post[lvs_all[0]]

        assert not self.cfg.net.atlas.qc.calc.levelwise
        self.cfg.var.obj_model.recorder['probs_source_closest'] = probs_source_closest
        self.cfg.var.obj_model.recorder['probs_closest_dists'] = dists

        return lv2prob_post # [B, A]

    @cache_res_by_domain('idxs_a')
    def get_idxs_atlas(self, domain):
        if self.cfg.net.atlas.soft_idx:
            lv2probs_a = self.get_probs_atlas(domain) # [B, A]
            lv2idxs_a = {k: v[:, None] for k, v in lv2probs_a.items()} # [B, S=1, A]
            return lv2idxs_a
        else:
            if not self.training:
                lv2probs_a = self.get_probs_atlas(domain) # [B, A]
                lv2idxs_a = {}
                for lv in lv2probs_a:
                    probs_a = lv2probs_a[lv] # [B, A]
                    idxs_a = probs_a.argmax(dim=1) # [B]
                    idxs_a = F.one_hot(idxs_a, probs_a.shape[1]) # [B, A]
                    idxs_a = idxs_a[:, None] # [B, S=1, A]
                    lv2idxs_a[lv] = idxs_a
                return lv2idxs_a

            if self.cfg.net.atlas.n_samples == 'deterministic':
                return None
            else:
                assert isinstance(self.cfg.net.atlas.n_samples, int)
                lv2probs_a = self.get_probs_atlas(domain) # [B, A]
                lv2idxs_a = {}
                for lv in lv2probs_a:
                    probs_a = lv2probs_a[lv] # [B, A]
                    probs_a = repeat(probs_a, 'B A -> B S A', S=self.cfg.net.atlas.n_samples) # [B, S, A]
                    idxs_a, _ = reinmax(probs_a, tau=3) # [B, S, A], using hard output (one-hot in the last dim)
                    lv2idxs_a[lv] = idxs_a
            return lv2idxs_a

    @cache_res_by_domain('poss_pred')
    def get_poss_pred(self, domain):
        lv2feat = self.get_lv2feat(domain)
        lv = self.cfg.net.n_levels - 1
        out_enc = lv2feat[f'{lv}'] # [B, C, ...]
        gap_out_enc = torch.mean(out_enc, dim=list(range(2, out_enc.dim()))) # [B, C]
        logits = self.domain2net_pos_slice[domain.split('_')[0]](gap_out_enc) # [B, 1]
        assert logits.shape[1] == 1
        assert logits.dim() == 2
        logits = logits.squeeze(1) # [B]
        poss_pred = torch.sigmoid(logits) # [B]
        return poss_pred

    def get_name2param_by_dist_vel(self, dist_vel, names):
        # dist_vel: [B, C, ...]
        # logits -> var -> std
        #               -> log_var
        assert all(name in ['mean', 'std', 'var', 'log_var'] for name in names)
        name2param = {}
        mean, logits = dist_vel.chunk(2, dim=1)
        if 'mean' in names:
            name2param['mean'] = mean
        if 'std' or 'log_var' in names:
            var = F.softplus(logits, beta=1).clamp(min=1e-5)
            if 'var' in names:
                name2param['var'] = var
        if 'std' in names:
            name2param['std'] = torch.sqrt(var)
        if 'log_var' in names:
            name2param['log_var'] = torch.log(var)
        return name2param

    @cache_res_by_domain('lv2name2param_atlas') # lv -> name -> [BxS, C/2, ...]
    def get_lv2name2param_atlas(self, domain):
        lv2feat = self.get_lv2feat(domain) # [B, C, ...]
        B = lv2feat['0'].shape[0]
        if self.cfg.net.atlas.n_samples == 'deterministic' and self.training and not self.cfg.net.atlas.soft_idx:
            lv2name2param = {}
            for l in reversed(range(self.cfg.net.n_levels)):
                raise NotImplementedError('have not changed get_atlas() to get_atlas_rotated()')
        else:
            # geometric mean of atlases for differentiable calculation
            lv2idxs_a = self.get_idxs_atlas(domain) # [B, S, A]
            lv2name2param = {}
            for l in reversed(range(self.cfg.net.n_levels)):
                idxs_a = lv2idxs_a[f'{l}'] # [B, S, A]
                atlas = self.get_atlas_rotated(domain, l) # [B, A, C, ...]
                atlas = repeat(atlas, 'B A C ... -> (B S) A C ...', S=idxs_a.shape[1]) # [BxS, A, C, ...]
                mean, logits = atlas.chunk(2, dim=2) # each [BxS, A, C/2, ...]
                ws = rearrange(idxs_a, 'B S A -> (B S) A') # [BxS, A]
                ws = ws[..., *[None] * (mean.dim() - 2)] # [BxS, A, 1, ...]
                if self.cfg.net.atlas.soft_idx_cfgs.mode == 'avg_sample':
                    name2param_atlas = {'mean': mean, 'logits': logits}
                    if self.cfg.net.atlas.soft_idx_cfgs.average == 'geometric':
                        name2param = self.poe.get_poe(name2param_atlas, ws=ws,
                                                      params_returned=['mean', 'std']) # [BxS, C/2, ...]
                    else:
                        raise NotImplementedError
                elif self.cfg.net.atlas.soft_idx_cfgs.mode == 'sample_avg':
                    var = self.poe.logits2var(logits) # [BxS, A, C/2, ...]
                    std = torch.sqrt(var) # [BxS, A, C/2, ...]
                    sample_atlas = torch.randn_like(mean) * std + mean # [BxS, A, C/2, ...]
                    if self.cfg.net.atlas.soft_idx_cfgs.average == 'arithmetic':
                        name2param = {'mean': torch.sum(sample_atlas * ws, dim=1)} # [BxS, C/2, ...]
                    elif self.cfg.net.atlas.soft_idx_cfgs.average == 'geometric':
                        name2param = {'mean': torch.prod(sample_atlas**ws, dim=1)} # [BxS, C/2, ...]
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                lv2name2param[f'{l}'] = name2param
        return lv2name2param

    def calc_seg_by_ws(self, ws):
        # ws: [B, A]
        assert self.cfg.net.atlas.soft_idx_cfgs.mode == 'avg_sample'
        lv2z = {}
        for l in reversed(range(self.cfg.net.n_levels)):
            atlas = getattr(self, f'atlas_lv{l}') # [A, C, ...]
            mean, logits = atlas.chunk(2, dim=1) # [A, C/2, ...]
            ws = ws.to(mean.device)
            mean, logits = mean[None], logits[None] # [1, A, C/2, ...]
            if ws.ndim != mean.ndim:
                ws = ws[..., *[None] * (mean.dim() - 2)] # [B, A, 1, ...]
            name2param_atlas = {'mean': mean, 'logits': logits}
            name2param_z = self.poe.get_poe(name2param_atlas, ws=ws, params_returned=['mean']) # [B, C/2, ...]
            lv2z[f'{l}'] = name2param_z['mean'] # [B, C/2, ...]

        logits = self.decoder_seg(lv2z) # [B, K, ...]
        seg = torch.argmax(logits, dim=1) # [B, ...]
        return seg

    @cache_res_by_domain('lv2name2param_pzc') # lv -> name -> [BxS, C/2, ...]
    def get_lv2name2param_pzc(self, domain):
        if self.cfg.model.qzxv.based_on == 'feat_reg':
            return self.get_lv2name2param_atlas(domain)
        elif self.cfg.model.qzxv.based_on == 'atlas':
            return self.get_lv2name2param_feat_reg(domain)
        else:
            raise ValueError(f'Unsupported value for cfg.model.qzxv.based_on: {self.cfg.model.qzxv.based_on}')

    @cache_res_by_domain('disps_affine')
    def get_disps_affine(self, domain):
        lv2disps_affine = {}
        lv2feat = self.get_lv2feat(domain) # [B, C, ...]
        for l in reversed(self.cfg.net.reg.levels):
            feat = lv2feat[f'{l}'] # [B, C, ...]
            disps_affine = []

            if self.cfg.net.reg_affine.enable:
                if self.cfg.net.reg_affine.zoom.enable:
                    zoom = self.get_params_affine_zoom(domain) # [B, n_zoom]
                    disp_zoom = self.get_disp_affine_zoom(zoom, feat.shape[2:]) # [B, D, ...]
                    disps_affine.append(disp_zoom)

                if self.cfg.net.reg_affine.translation.enable:
                    dij = self.get_params_affine_translation(domain) # [B, D]
                    dij = dij * 2**(self.cfg.net.reg_affine.translation.level_predict - l) # [B, D]
                    disp_trans = repeat(dij, 'b d -> b d h w', h=feat.shape[-2], w=feat.shape[-1])
                    disps_affine.append(disp_trans)

            lv2disps_affine[f'{l}'] = disps_affine
        return lv2disps_affine

    @cache_res_by_domain('disps_affine_inv')
    def get_disps_affine_inv(self, domain):
        lv2disps_affine = {}
        for l in reversed(self.cfg.net.reg.levels):
            lv2feat = self.get_lv2feat(domain) # [B, C, ...]
            feat = lv2feat[f'{l}'] # [B, C, ...]
            disps_affine = []

            if self.cfg.net.reg_affine.enable:
                if self.cfg.net.reg_affine.translation.enable:
                    dij = -self.get_params_affine_translation(domain) # [B, D]
                    dij = dij * 2**(self.cfg.net.reg_affine.translation.level_predict - l) # [B, D]
                    disp_trans = repeat(dij, 'b d -> b d h w', h=feat.shape[-2], w=feat.shape[-1])
                    disps_affine.append(disp_trans)

                if self.cfg.net.reg_affine.zoom.enable:
                    zoom = self.get_params_affine_zoom(domain) # [B, n_zoom]
                    disp_zoom = self.get_disp_affine_zoom(1/zoom, feat.shape[2:]) # [B, D, ...]
                    disps_affine.append(disp_zoom)

            lv2disps_affine[f'{l}'] = disps_affine
        return lv2disps_affine

    @cache_res_by_domain('lv2name2param_dist_vel', 'lv_ori_resize2vel', 'lv_ori_resize2disp')
    def get_disps_levelwise(self, domain):
        lv2feat = self.get_lv2feat(domain) # [B, C, ...]
        if self.cfg.net.reg.input_from_atlas == 'mean':
            lv2name2param_atlas = self.get_lv2name2param_atlas(domain) # [BxS, C/2, ...]
        elif self.cfg.net.reg.input_from_atlas == 'sample':
            lv2z = self.get_lv2z(domain) # [BxS, C/2, ...]
        else:
            raise ValueError(f'Unsupported value for cfg.net.reg.input_from_atlas: {self.cfg.net.reg.input_from_atlas}')
        lv2name2param_dist_vel = {} # [BxS, D, ...]
        lv_ori_resize2vel = {} # [BxS, D, ...]
        lv_ori_resize2disp = {} # [BxS, D, ...]
        lv2disps_affine = self.get_disps_affine(domain) # [BxS, D, ...]
        for l in reversed(self.cfg.net.reg.levels):
            if self.cfg.net.reg.input_from_atlas == 'mean':
                mean_atlas = lv2name2param_atlas[f'{l}']['mean'] # [BxS, C/2, ...]
            elif self.cfg.net.reg.input_from_atlas == 'sample':
                mean_atlas = lv2z[f'{l}'] # [BxS, C/2, ...]
            else:
                raise ValueError(
                    f'Unsupported value for cfg.net.reg.input_from_atlas: {self.cfg.net.reg.input_from_atlas}')
            mean_feats = lv2feat[f'{l}'].chunk(2, dim=1)[0] # [B, C/2, ...]
            S = int(mean_atlas.shape[0] // mean_feats.shape[0])
            mean_feats = repeat(mean_feats, 'B C ... -> (B S) C ...', S=S) # [BxS, C/2, ...]
            disps_affine = lv2disps_affine[f'{l}'] # [BxS, D, ...]
            if l != self.cfg.net.reg.levels[-1]:
                disps = [
                    lv_ori_resize2disp[f'{j}_{l}'] for j in range(l + 1, self.cfg.net.n_levels)
                    if j in self.cfg.net.reg.levels
                ] # [BxS, D, ...]
                disps.extend(disps_affine)
            else:
                disps = disps_affine
            if len(disps) > 0:
                mean_feats = self.lv2stn[f'{l}'](mean_feats, disps) # [BxS, C, ...]

            input_reg = torch.cat([mean_atlas, mean_feats], dim=1) # [BxS, C, ...]
            dist_vel = self.domain_lv2reg[f'{domain.split("_")[0]}_{l}'](input_reg) # [BxS, 2D, ...]
            if self.training:
                name2param = self.get_name2param_by_dist_vel(dist_vel, ['mean', 'std', 'var']) # [BxS, D, ...]
                mean = name2param['mean'] # [BxS, D, ...]
                if self.cfg.net.reg_affine.enable and self.cfg.net.reg_affine.translation.enable and self.cfg.net.reg_affine.translation.remove_mean:
                    mean = mean - torch.mean(mean, dim=list(range(2, mean.ndim)), keepdim=True)
                    name2param['mean'] = mean
                vel = torch.randn_like(name2param['mean']) * name2param['std'] + name2param['mean'] # [BxS, D, ...]
                name2param.pop('std')
            else:
                name2param = self.get_name2param_by_dist_vel(dist_vel, ['mean']) # [BxS, D, ...]
                mean = name2param['mean'] # [BxS, D, ...]
                if self.cfg.net.reg_affine.enable and self.cfg.net.reg_affine.translation.enable and self.cfg.net.reg_affine.translation.remove_mean:
                    mean = mean - torch.mean(mean, dim=list(range(2, mean.ndim)), keepdim=True)
                    name2param['mean'] = mean
                vel = name2param['mean'] # [BxS, D, ...]
            lv2name2param_dist_vel[f'{l}'] = name2param

            for i in range(l + 1):
                if i not in self.cfg.net.reg.levels:
                    continue
                vel_resized = self.resize_vec(vel, 2**(l - i)) # [BxS, D, ...]
                disp_resized = self.lv2vel2disp[f'{i}'](vel_resized) # [BxS, D, ...]
                lv_ori_resize2disp[f'{l}_{i}'] = disp_resized # [BxS, D, ...]
                if i == 0:
                    lv_ori_resize2vel[f'{l}_0'] = vel_resized # [BxS, D, ...]

        return lv2name2param_dist_vel, lv_ori_resize2vel, lv_ori_resize2disp

    @cache_res_by_domain('disp_composed_by_velocity')
    def get_disp_composed_by_velocity(self, domain):
        """Compose only multilevel velocity-integrated disps (no affine). For saving velocity-only disp."""
        _, _, lv_ori_resize2disp = self.get_disps_levelwise(domain)  # [BxS, D, ...]
        disps = [lv_ori_resize2disp[f'{l}_0'] for l in self.cfg.net.reg.levels]
        return self.lv2stn['0'].compose_flows(flows=disps)  # [BxS, D, ...]

    @cache_res_by_domain('disp_final')
    def get_disp_final(self, domain):
        _, _, lv_ori_resize2disp = self.get_disps_levelwise(domain) # [BxS, D, ...]
        disps = [lv_ori_resize2disp[f'{l}_0'] for l in self.cfg.net.reg.levels] # each [BxS, D, ...]
        disps_affine = self.get_disps_affine(domain).get(f'{self.cfg.net.reg.levels[0]}', [])
        disps.extend(disps_affine)
        disp_final = self.lv2stn['0'].compose_flows(flows=disps) # [BxS, D, ...]
        return disp_final # [BxS, D, ...]

    @cache_res_by_domain('imgs_reg')
    def get_imgs_reg(self, domain):
        disp_final = self.get_disp_final(domain) # [BxS, D, ...]
        img = self.data[f'img_{domain}'] # [B, 1, ...]
        S = int(disp_final.shape[0] // img.shape[0])
        img = repeat(img, 'B ... -> (B S) ...', S=S) # [BxS, 1, ...]
        img_reg = self.lv2stn['0'](img, disp_final) # [BxS, 1, ...]
        return img_reg

    @cache_res_by_domain('segs_gt_reg')
    def get_segs_gt_reg(self, domain):
        disp_final = self.get_disp_final(domain) # [BxS, D, ...]
        seg_gt = self.data[f'seg_{domain}'] # [B, 1, ...]
        S = int(disp_final.shape[0] // seg_gt.shape[0])
        seg_gt = repeat(seg_gt, 'B ... -> (B S) ...', S=S) # [BxS, 1, ...]
        seg_gt_reg = self.lv2stn['0'](seg_gt, disp_final, interp_mode='nearest')
        return seg_gt_reg

    @cache_res_by_domain('lv2feat_reg')
    def get_lv2feat_reg(self, domain):
        img_reg = self.get_imgs_reg(domain)
        lv2feat_reg = self.domain2encoder[domain.split('_')[0]](img_reg) # [BxS, C, ...]
        return lv2feat_reg

    @cache_res_by_domain('lv2name2param_feat_reg')
    def get_lv2name2param_feat_reg(self, domain): # lv -> name -> [BxS, C/2, ...]
        lv2feat = self.get_lv2feat_reg(domain) # [BxS, C, ...]
        lv2name2param_qzxv = {}
        for lv, feat in lv2feat.items():
            mean, logits = feat.chunk(2, dim=1) # [BxS, C/2, ...]
            if self.training:
                var = self.poe.logits2var(logits) # [BxS, C/2, ...]
                std = torch.sqrt(var) # [BxS, C/2, ...]
                lv2name2param_qzxv[lv] = {'mean': mean, 'std': std}
            else:
                lv2name2param_qzxv[lv] = {'mean': mean}
        return lv2name2param_qzxv

    @cache_res_by_domain('lv2name2param_qzxv')
    def get_lv2name2param_qzxv(self, domain): # lv -> name -> [BxS, C/2, ...]
        if self.cfg.model.qzxv.based_on == 'feat_reg':
            return self.get_lv2name2param_feat_reg(domain)
        elif self.cfg.model.qzxv.based_on == 'atlas':
            return self.get_lv2name2param_atlas(domain)
        else:
            raise ValueError(f'Unsupported value for cfg.model.qzxv.based_on: {self.cfg.model.qzxv.based_on}')

    @cache_res_by_domain('lv2z') # lv -> [BxS, C/2, ...]
    def get_lv2z(self, domain):
        lv2name2param_qzxv = self.get_lv2name2param_qzxv(domain) # [BxS, C/2, ...]
        lv2z = {}
        for lv, name2param_qzxv in lv2name2param_qzxv.items():
            mean = name2param_qzxv['mean'] # [BxS, C/2, ...]
            if self.training:
                if 'std' not in name2param_qzxv:
                    lv2z[lv] = mean
                else:
                    std = name2param_qzxv['std'] # [BxS, C/2, ...]
                    lv2z[lv] = torch.randn_like(mean) * std + mean # [BxS, C/2, ...]
            else:
                lv2z[lv] = mean # [BxS, C/2, ...]

        return lv2z

    @cache_res_by_domain('disp_inv_final')
    def get_disp_inv_final(self, domain):
        _, lv_ori_resize2vel, _ = self.get_disps_levelwise(domain) # [BxS, D, ...]
        disps_inv = self.get_disps_affine_inv(domain).get(f'{self.cfg.net.reg.levels[0]}', [])
        for l in reversed(self.cfg.net.reg.levels):
            vel = lv_ori_resize2vel[f'{l}_0'] # [BxS, D, ...]
            disps_inv.append(self.lv2vel2disp['0'](-vel)) # [BxS, D, ...]
        disp_inv_final = self.lv2stn['0'].compose_flows(disps_inv) # [BxS, D, ...]
        return disp_inv_final

    @cache_res_by_domain('imgs_rec_reg')
    def get_imgs_rec_reg(self, domain):
        lv2z_reg = self.get_lv2z(domain) # [BxS, C/2, ...]

        detach = self.cfg.net.decoder_rec.detach_decoder_input
        if (domain == 'source' and detach[0]) or (domain == 'target' and detach[1]):
            lv2z_reg = {k: v.detach() for k, v in lv2z_reg.items()}

        if self.cfg.net.style_code.enable:
            style_code = self.get_style_code(domain) # [B, C]
        else:
            style_code = None

        imgs_rec_reg = self.domain2decoder_rec[domain.split('_')[0]](lv2z_reg, style_code) # [BxS, 1, ...]
        return imgs_rec_reg

    @cache_res_by_domain('imgs_rec_ori')
    def get_imgs_rec_ori(self, domain):
        imgs_rec_reg = self.get_imgs_rec_reg(domain) # [BxS, 1, ...]
        disp_inv_final = self.get_disp_inv_final(domain) # [BxS, D, ...]
        imgs_rec = self.lv2stn['0'](imgs_rec_reg, disp_inv_final) # [BxS, 1, ...]
        return imgs_rec

    @cache_res_by_domain('logits_seg_reg')
    def get_logits_seg_reg(self, domain):
        lv2z_reg = self.get_lv2z(domain) # [BxS, C/2, ...]
        logits_seg_reg = self.decoder_seg(lv2z_reg) # [BxS, K, ...]
        return logits_seg_reg

    @cache_res('logits_seg_atlas')
    def get_logits_seg_atlas(self):
        assert not self.training
        lv2z_atlas = {}
        for l in reversed(range(self.cfg.net.n_levels)):
            atlas = getattr(self, f'atlas_lv{l}') # [A, C, ...]
            mean, _ = atlas.chunk(2, dim=1) # [A, C/2, ...]
            lv2z_atlas[f'{l}'] = mean
        logits_seg_atlas = self.decoder_seg(lv2z_atlas) # [A, K, ...]
        return logits_seg_atlas

    @cache_res('imgs_rec_atlas')
    def get_imgs_rec_atlas(self):
        assert not self.training
        lv2z_atlas = {}
        for l in reversed(range(self.cfg.net.n_levels)):
            atlas = getattr(self, f'atlas_lv{l}') # [A, C, ...]
            mean, _ = atlas.chunk(2, dim=1) # [A, C/2, ...]
            lv2z_atlas[f'{l}'] = mean
        imgs_rec_atlas_source = self.domain2decoder_rec['source'](lv2z_atlas) # [A, 1, ...]
        imgs_rec_atlas_target = self.domain2decoder_rec['target'](lv2z_atlas) # [A, 1, ...]
        return (imgs_rec_atlas_source, imgs_rec_atlas_target)

    @cache_res_by_domain('logits_seg_ori')
    def get_logits_seg(self, domain):
        logits_seg_reg = self.get_logits_seg_reg(domain) # [BxS, K, ...]
        disp_inv_final = self.get_disp_inv_final(domain) # [BxS, D, ...]
        logits_seg = self.lv2stn['0'](logits_seg_reg, disp_inv_final) # [BxS, K, ...]
        return logits_seg

    @staticmethod
    def resize_vec(x, scale):
        if scale == 1:
            return x
        # x: [B, D, ...]
        dim = len(x.shape) - 2
        if dim == 2:
            interp_mode = 'bilinear'
        elif dim == 3:
            interp_mode = 'trilinear'
        else:
            raise ValueError

        if scale < 1:
            x = F.interpolate(x, scale_factor=scale, align_corners=True, mode=interp_mode)
            x = x * scale
        elif scale > 1:
            x = x * scale
            x = F.interpolate(x, scale_factor=scale, align_corners=True, mode=interp_mode)
        return x

    def forward(self, data):
        if self.training:
            if max(
                    self.cfg.model.ws_loss.same_qc_rand_bias_field,
                    self.cfg.model.ws_loss.same_disp_rand_bias_field,
            ) > 0.:
                cfg_this = self.cfg.dataset.aug.rand_bias_field
                assert cfg_this.enable
                for domain in ['source', 'target']:
                    img = data[f'img_{domain}'] # [B, 1, ...]
                    idxs = np.random.choice(img.shape[0], max(int(img.shape[0] * cfg_this.prob), 1), replace=False)
                    img_aug = mt.RandBiasField(coeff_range=cfg_this.coeff_range, prob=1.,
                                               dtype=None)(img[idxs]) # [B, 1, ...]
                    img_aug = torch.Tensor(img_aug)
                    data[f'img_{domain}_aug'] = img_aug
                    data[f'idxs_img_aug_{domain}'] = idxs

        if isinstance(data, dict):
            out = {}
            for domain in ['source', 'target']:
                if f'img_{domain}' in data:
                    logits = self.get_logits_seg(domain) # [BxS, K, ...]
                    out[f'logits_{domain}'] = logits
            return out
        else:
            assert isinstance(data, torch.Tensor)
            assert not self.training
            self.cfg.var.obj_model.clear_nested_dict_or_list(self.cfg.var.obj_model.recorder)
            if f'img_source' in self.data:
                assert 'img_target' not in self.data
                self.data['img_source'] = data
                logits = self.get_logits_seg('source')
                return logits
            elif f'img_target' in self.data:
                assert 'img_source' not in self.data
                self.data['img_target'] = data
                logits = self.get_logits_seg('target')
                return logits
            else:
                raise None
