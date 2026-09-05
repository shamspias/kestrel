import torch, warnings; warnings.filterwarnings("ignore")
from torch.utils.flop_counter import FlopCounterMode
from kestrel import KESTREL, KestrelConfig, count_params
from dataclasses import replace
torch.manual_seed(0)
configs = {
 "N": dict(stem_ch=16, conv_dims=(32,64), conv_depths=(1,2), attn_dims=(128,192), attn_depths=(3,2), neck_dim=128, d_model=128, embed_dim=128, dec_layers=3, dec_heads=4, num_queries=100),
 "S": dict(stem_ch=24, conv_dims=(48,96), conv_depths=(2,3), attn_dims=(192,288), attn_depths=(4,3), neck_dim=192, d_model=192, embed_dim=192, dec_layers=4, dec_heads=6, num_queries=300),
 "M": dict(),
 "L": dict(stem_ch=48, conv_dims=(96,192), conv_depths=(3,6), attn_dims=(384,512), attn_depths=(8,6), neck_dim=320, d_model=320, embed_dim=320, dec_layers=6, dec_heads=10, num_queries=300),
}
def gfl(m, x, **kw):
    with torch.no_grad(), FlopCounterMode(display=False) as fc: m(x, **kw)
    return fc
print(f"{'size':<5}{'params(M)':>10}{'GFLOPs@640':>12}{'GFLOPs@640 +masks':>19}{'GFLOPs@1280':>13}")
for name, kw in configs.items():
    cfg = replace(KestrelConfig(), **kw)
    m = KESTREL(cfg).eval().reparameterize()
    x = torch.randn(1,3,640,640)
    f = gfl(m, x, return_masks=False).get_total_flops()/1e9
    fs = gfl(m, x, return_masks=True).get_total_flops()/1e9
    f2 = gfl(m, torch.randn(1,3,1280,1280), return_masks=False).get_total_flops()/1e9
    print(f"{name:<5}{count_params(m)/1e6:>10.2f}{f:>12.1f}{fs:>19.1f}{f2:>13.1f}")
    if name == "M":
        fc = gfl(m, x, return_masks=False)
        counts = fc.get_flop_counts()
        tot = fc.get_total_flops()
        print("  M breakdown (detect):")
        for comp in ["stem","s4","s8","s16","s32","neck","dense","select","decoder","presence"]:
            key = f"KESTREL.{comp}"
            c = sum(v for k,v in counts.items() if k==key or k.startswith(key+".")) if False else counts.get(key, {})
            c = sum(c.values()) if isinstance(c, dict) else 0
            print(f"    {comp:<9} {c/1e9:6.1f} GFLOPs ({100*c/tot:4.1f}%)")
