import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False



class ViTBackbone(nn.Module):

    def __init__(self, img_size=(256,192), pretrained=True,
                 model_name="vit_base_patch16_224", vitpose_ckpt="", vitpose_expert=3):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError(
                "timm not installed. `pip install timm` for the ViT backbone, "
                "or use backbone='simple'."
            )

        use_imnet = pretrained and not vitpose_ckpt
        self.vit = timm.create_model(
            model_name, pretrained=use_imnet, num_classes=0,
            img_size=img_size, dynamic_img_size=True,
        )
        if vitpose_ckpt:
            import os
            if os.path.exists(vitpose_ckpt):
                load_vitpose_backbone(self.vit, vitpose_ckpt, expert_idx=vitpose_expert)
            else:
                print(f"[ViTPose] WARNING ckpt not found: {vitpose_ckpt} "
                      f"-> using {'ImageNet' if use_imnet else 'random'} init")
        self.embed_dim = self.vit.embed_dim
        self.patch = 16
        H, W = img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size)
        self.gh, self.gw = H // self.patch, W // self.patch   # 16 x 12

    def forward(self, x):
        B = x.shape[0]
        feats = self.vit.forward_features(x)         # [B, 1+HpWp, C] (cls + patches)
        if feats.dim() == 3:
            n_patch = self.gh * self.gw
            feats = feats[:, -n_patch:, :]           # [B, HpWp, C]
            feats = feats.transpose(1, 2).reshape(B, self.embed_dim, self.gh, self.gw)
        return feats                                  # [B, C, gh, gw]


class SimpleConvBackbone(nn.Module):
    def __init__(self, img_size=(256,192)):
        super().__init__()
        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.net = nn.Sequential(
            blk(3, 64, 2), blk(64, 128, 2), blk(128, 256, 2), blk(256, 512, 2),
        )  # /16
        self.embed_dim = 512

    def forward(self, x):
        return self.net(x)


def _strip_state_dict(ck):
    sd = ck
    for k in ("state_dict", "model", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    return sd


# ViTPose++ dataset order 
VITPOSEPP_EXPERTS = {0: "AIC", 1: "MPII", 2: "COCO", 3: "AP10K",
                     4: "APT36K", 5: "WholeBody"}
AP10K_EXPERT = 3


def _remap_vitpose_to_timm(sd, expert_idx=AP10K_EXPERT):

    out = {}
    expert_w = {} 
    expert_b = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue
        nk = k[len("backbone."):]
        m = re.search(r"blocks\.(\d+)\.mlp\.experts\.(\d+)\.(weight|bias)", nk)
        if m:
            blk, idx, wb = int(m.group(1)), int(m.group(2)), m.group(3)
            (expert_w if wb == "weight" else expert_b).setdefault(blk, {})[idx] = v
            continue
        if nk.startswith("last_norm."):
            nk = "norm." + nk[len("last_norm."):]
        out[nk] = v

    is_moe = len(expert_w) > 0
    if is_moe:
        for blk, experts in expert_w.items():
            if expert_idx not in experts:
                continue
            shared_w = out.get(f"blocks.{blk}.mlp.fc2.weight")
            shared_b = out.get(f"blocks.{blk}.mlp.fc2.bias")
            if shared_w is None:
                continue
            out[f"blocks.{blk}.mlp.fc2.weight"] = torch.cat(
                [shared_w, experts[expert_idx]], dim=0)          # (576+192, 3072)
            out[f"blocks.{blk}.mlp.fc2.bias"] = torch.cat(
                [shared_b, expert_b[blk][expert_idx]], dim=0)    # (768,)
    return out, is_moe


def load_vitpose_backbone(vit_module, ckpt_path, expert_idx=AP10K_EXPERT, verbose=True):

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd, is_moe = _remap_vitpose_to_timm(_strip_state_dict(ck), expert_idx=expert_idx)
    if verbose and is_moe:
        print(f"[ViTPose++] MoE checkpoint: merging expert {expert_idx} "
              f"({VITPOSEPP_EXPERTS.get(expert_idx, '?')}) into standard MLP")

    model_sd = vit_module.state_dict()

    if "pos_embed" in sd and "pos_embed" in model_sd:
        if sd["pos_embed"].shape != model_sd["pos_embed"].shape:
            sd["pos_embed"] = _interp_pos_embed(sd["pos_embed"], model_sd["pos_embed"])

    keep, skip = {}, []
    for k, v in sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            keep[k] = v
        else:
            skip.append(k)

    missing = [k for k in model_sd if k not in keep]
    res = vit_module.load_state_dict(keep, strict=False)
    if verbose:
        print(f"[ViTPose] loaded {len(keep)}/{len(model_sd)} backbone tensors")
        if skip:
            print(f"[ViTPose] shape-mismatch/unused from ckpt: {len(skip)} "
                  f"(e.g. {skip[:3]})")
        if missing:
            print(f"[ViTPose] left at init (not in ckpt): {len(missing)} "
                  f"(e.g. {missing[:3]})")
    return len(keep), len(missing), len(skip)


def _interp_pos_embed(src, dst):
    import math
    def split_cls(p):
        n = p.shape[1]
        return (p[:, :1], p[:, 1:]) if _has_cls(n) else (None, p)
    def _has_cls(n):
        import math as _m
        return not float(_m.isqrt(n)) ** 2 == n and float(_m.isqrt(n - 1)) ** 2 == (n - 1)
    src_cls, src_grid = split_cls(src)
    dst_cls, dst_grid = split_cls(dst)
    C = src_grid.shape[-1]
    s_n, d_n = src_grid.shape[1], dst_grid.shape[1]
    sh, sw = _grid_hw(s_n); dh, dw = _grid_hw(d_n)
    g = src_grid.reshape(1, sh, sw, C).permute(0, 3, 1, 2)
    g = torch.nn.functional.interpolate(g, size=(dh, dw), mode="bicubic", align_corners=False)
    g = g.permute(0, 2, 3, 1).reshape(1, dh * dw, C)
    if dst_cls is not None and src_cls is not None:
        g = torch.cat([src_cls, g], dim=1)
    return g


def _grid_hw(n):
    import math
    if n == 192:   # ViTPose-B 256x192
        return 16, 12
    r = int(math.isqrt(n))
    while r > 1 and n % r:
        r -= 1
    return r, n // r



class DeconvHead(nn.Module):
    def __init__(self, in_ch, num_joints, hidden=256, n_deconv=2):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(n_deconv):
            layers += [
                nn.ConvTranspose2d(c, hidden, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden), nn.ReLU(True),
            ]
            c = hidden
        self.deconv = nn.Sequential(*layers)
        self.final = nn.Sequential(
    nn.Conv2d(hidden, num_joints, 1),
    # NO ReLU — let model output unbounded values
)
    def forward(self, x):
        return self.final(self.deconv(x))            # [B, N, 64, 64]


class DogPoseNet(nn.Module):
    def __init__(self, cfg=None, num_joints=None, img_size=None,
                 pretrained=None, backbone=None):
        super().__init__()
        if cfg is not None:
            num_joints = num_joints or cfg.model.num_joints
            img_size = img_size or (cfg.geom.input_h, cfg.geom.input_w)
            pretrained = cfg.model.pretrained if pretrained is None else pretrained
            backbone = backbone or cfg.model.backbone
            hidden = cfg.model.deconv_hidden
            n_deconv = cfg.model.n_deconv
            vit_name = cfg.model.vit_name
        else:
            num_joints = num_joints or 49
            img_size = img_size or (256, 192)
            pretrained = True if pretrained is None else pretrained
            backbone = backbone or "vit"
            hidden, n_deconv, vit_name = 256, 2, "vit_base_patch16_224"
        vitpose_ckpt = cfg.model.vitpose_ckpt if cfg is not None else ""
        vitpose_expert = cfg.model.vitpose_expert if cfg is not None else 3
        if backbone == "vit":
            self.backbone = ViTBackbone(img_size, pretrained, vit_name, vitpose_ckpt, vitpose_expert)
        else:
            self.backbone = SimpleConvBackbone(img_size)
        self.head = DeconvHead(self.backbone.embed_dim, num_joints, hidden, n_deconv)
        self.num_joints = num_joints

    def forward(self, x):
        return self.head(self.backbone(x))           # [B, N, hm_h, hm_w]



def soft_argmax(heatmaps):
    B, N, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, N, -1)
    prob = F.softmax(flat, dim=-1).reshape(B, N, H, W)
    xs = torch.linspace(0, 1, W, device=heatmaps.device).view(1, 1, 1, W)
    ys = torch.linspace(0, 1, H, device=heatmaps.device).view(1, 1, H, 1)
    x = (prob * xs).sum(dim=(2, 3))
    y = (prob * ys).sum(dim=(2, 3))
    return torch.stack([x, y], dim=-1)               # [B,N,2]


class MaskedHeatmapLoss(nn.Module):
    def __init__(self, invisible_weight=0.1):
        super().__init__()
        self.invisible_weight = invisible_weight  

    def forward(self, pred_hm, target_hm, target_weight):
        B, N = target_weight.shape
        per_joint = F.mse_loss(pred_hm, target_hm, reduction="none").mean(dim=(2, 3))  # [B,N]
        
        weight = target_weight * 1.0 + (1 - target_weight) * self.invisible_weight
        per_joint = per_joint * weight
        
        denom = weight.sum().clamp(min=1.0)
        return per_joint.sum() / denom


@torch.no_grad()
def pck(pred_coords, gt_coords, visibility, thr=0.1):
    d = torch.norm(pred_coords - gt_coords, dim=-1)          
    vis = visibility > 0
    correct, total = 0.0, 0.0
    for b in range(pred_coords.shape[0]):
        m = vis[b]
        if m.sum() < 2:
            continue
        pts = gt_coords[b][m]
        wh = pts.max(0).values - pts.min(0).values
        diag = torch.norm(wh).clamp(min=1e-6)
        correct += ((d[b][m] / diag) < thr).float().sum().item()
        total += m.sum().item()
    return correct / max(total, 1.0)


if __name__ == "__main__":

    from config import CFG
    print(f"timm available: {_HAS_TIMM} | backbone={CFG.model.backbone} | "
          f"ckpt={CFG.model.vitpose_ckpt or '(none)'} | expert={CFG.model.vitpose_expert}")
    net = DogPoseNet(cfg=CFG)
    x = torch.randn(2, 3, CFG.geom.input_h, CFG.geom.input_w)
    hm = net(x)
    print("heatmaps:", tuple(hm.shape))
    coords = soft_argmax(hm)
    print("coords:", tuple(coords.shape))
    tw = torch.ones(2, CFG.model.num_joints)
    tgt = torch.rand(2, CFG.model.num_joints, CFG.geom.hm_h, CFG.geom.hm_w)
    with torch.no_grad():
        print("loss:", round(MaskedHeatmapLoss()(hm, tgt, tw).item(), 4))
        print("pck:", round(pck(coords, torch.rand(2, CFG.model.num_joints, 2), tw), 4))
    print(f"params: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")
