import torch
import torch.nn as nn
from timm.models.layers import DropPath
from models_mamba import create_block, RMSNorm, rms_norm_fn, StrideEmbed
from timm.models.layers import trunc_normal_, lecun_normal_
import math
from functools import partial

class TorMamba(nn.Module):
    def __init__(self, byte_length=1600, stride_size=4, in_chans=1,
                 embed_dim=192, depth=4, 
                 decoder_embed_dim=128, decoder_depth=2,
                 num_classes=1000,
                 norm_pix_loss=False,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 bimamba_type="none",
                 is_pretrain=False,
                 alpha=1.0, beta=1.0,
                 device=None, dtype=None,
                 **kwargs):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.is_pretrain = is_pretrain
        self.stride_size = stride_size
        self.norm_pix_loss = norm_pix_loss
        self.alpha = alpha
        self.beta = beta

        self.patch_embed = StrideEmbed(byte_length, stride_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            create_block(embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=True,
                         residual_in_fp32=True, fused_add_norm=True,
                         layer_idx=i, if_bimamba=False, bimamba_type=bimamba_type,
                         drop_path=inter_dpr[i], if_devide_out=True)
            for i in range(depth)])
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)

        if is_pretrain:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
            decoder_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
            decoder_inter_dpr = [0.0] + decoder_dpr
            self.decoder_blocks = nn.ModuleList([
                create_block(decoder_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=True,
                             residual_in_fp32=True, fused_add_norm=True,
                             layer_idx=i, if_bimamba=False, bimamba_type=bimamba_type,
                             drop_path=decoder_inter_dpr[i], if_devide_out=True)
                for i in range(decoder_depth)])
            self.decoder_norm_f = RMSNorm(decoder_embed_dim, eps=1e-5)
            self.decoder_pred = nn.Linear(decoder_embed_dim, stride_size * in_chans, bias=True)
        else:
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.patch_embed.apply(self._init_weights)
        if not is_pretrain:
            self.head.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if is_pretrain:
            trunc_normal_(self.decoder_pos_embed, std=.02)
            trunc_normal_(self.mask_token, std=.02)
        self.apply(partial(self._init_weights, n_layer=depth))

    def _init_weights(self, m, n_layer=4):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
            if getattr(m, "weight", None) is not None:
                nn.init.constant_(m.weight, 1.0)
    def forward_triplet(self, anchor, positive, negative, mask_ratio=0.9):
        """
        Wrapper for triplet training, matches engine's call.
        """
        return self.forward(imgs_anchor=anchor, imgs_pos=positive, imgs_neg=negative, mask_ratio=mask_ratio, mode="mae+triplet")
    def forward_encoder(self, x, mask_ratio=0.9, if_mask=True):
        B, C, H, W = x.shape
        x = self.patch_embed(x.reshape(B, C, -1)) + self.pos_embed[:, :-1, :]
        if if_mask:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = ids_restore = None
        cls_tokens = self.cls_token + self.pos_embed[:, -1:, :]
        x = torch.cat((x, cls_tokens.expand(B, -1, -1)), dim=1)
        x = self.pos_drop(x)

        residual = None
        for blk in self.blocks:
            x, residual = blk(x, residual)
        x = rms_norm_fn(self.drop_path(x), self.norm_f.weight, self.norm_f.bias,
                        eps=self.norm_f.eps, residual=residual, prenorm=False, residual_in_fp32=True)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, :-1, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x_, x[:, -1:, :]], dim=1)
        x = x + self.decoder_pos_embed
        residual = None
        for blk in self.decoder_blocks:
            x, residual = blk(x, residual)
        x = rms_norm_fn(self.drop_path(x), self.decoder_norm_f.weight, self.decoder_norm_f.bias,
                        eps=self.decoder_norm_f.eps, residual=residual, prenorm=False, residual_in_fp32=True)
        x = self.decoder_pred(x)
        return x[:, :-1, :]

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        x: [B, N, C], sequence
        """
        B, N, C = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_rec_loss(self, imgs, pred, mask):
        target = self.stride_patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    def stride_patchify(self, imgs):
        B, C, H, W = imgs.shape
        return imgs.reshape(B, H * W // self.stride_size, self.stride_size)

    def forward(self, imgs=None, mask_ratio=0.9, mode="mae",
                imgs_anchor=None, imgs_pos=None, imgs_neg=None):
        if not self.is_pretrain:
            x, _, _ = self.forward_encoder(imgs, mask_ratio=mask_ratio, if_mask=False)
            return self.head(x[:, -1, :])

        if mode == "mae":
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio=mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_rec_loss(imgs, pred, mask)
            return loss, pred, mask

        elif mode == "mae+triplet":
            # MAE
            latent, mask, ids_restore = self.forward_encoder(imgs_anchor, mask_ratio=mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)
            mae_loss = self.forward_rec_loss(imgs_anchor, pred, mask)
            # Triplet
            feat_a, _, _ = self.forward_encoder(imgs_anchor, mask_ratio=0.0, if_mask=False)
            feat_p, _, _ = self.forward_encoder(imgs_pos, mask_ratio=0.0, if_mask=False)
            feat_n, _, _ = self.forward_encoder(imgs_neg, mask_ratio=0.0, if_mask=False)

            return mae_loss, feat_a[:, -1, :], feat_p[:, -1, :], feat_n[:, -1, :]


def tor_mamba_pretrain(**kwargs):
    return TorMamba(is_pretrain=True, stride_size=4, embed_dim=256, depth=4,
                    decoder_embed_dim=128, decoder_depth=2, **kwargs)

def tor_mamba_classifier(**kwargs):
    return TorMamba(is_pretrain=False, stride_size=4, embed_dim=256, depth=4, **kwargs)
