# -*- coding: utf-8 -*-
"""
One-to-one Top-1 pairing between two folders (no threshold, no geometric verification; pairs purely by highest similarity)
- wm: folder of watermarked images
- clean: folder of clean (non-watermarked) images
- Writes directly into a single output directory: img00001_wm.jpg / img00001.jpg (numbering continues automatically)
"""

import os, re, argparse, shutil
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import open_clip
from scipy.optimize import linear_sum_assignment

def list_images(root):
    exts={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
    return [os.path.join(root,f) for f in os.listdir(root)
            if os.path.splitext(f.lower())[1] in exts and os.path.isfile(os.path.join(root,f))]

def next_index(out_dir, prefix="img"):
    if not os.path.isdir(out_dir): return 1
    pat=re.compile(rf"^{re.escape(prefix)}(\d{{5}})(?:_wm)?\.[A-Za-z0-9]+$")
    mx=0
    for f in os.listdir(out_dir):
        m=pat.match(f)
        if m: mx=max(mx,int(m.group(1)))
    return mx+1 if mx>0 else 1

def build_clip(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval().to(device)
    tfm = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    return model, tfm, device

@torch.no_grad()
def embed_paths(paths, model, tfm, device):
    feats=[]
    for p in paths:
        im = Image.open(p).convert("RGB")
        x = tfm(im).unsqueeze(0).to(device)
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
        feats.append(v.squeeze(0).cpu().numpy().astype("float32"))
    return np.stack(feats, axis=0)  # [N,D]

def main(args):
    wm_paths   = list_images(args.wm)
    clean_paths= list_images(args.clean)
    if not wm_paths or not clean_paths:
        raise SystemExit("No images found in the wm/clean directories")
    n = min(len(wm_paths), len(clean_paths))
    print(f"[INFO] wm:{len(wm_paths)} clean:{len(clean_paths)} -> target {n} pairs (no threshold)")

    model, tfm, device = build_clip()
    VW = embed_paths(wm_paths,   model, tfm, device)  # [Nw,D]
    VC = embed_paths(clean_paths,model, tfm, device)  # [Nc,D]

    # Cosine similarity matrix (Nw x Nc)
    sim = VW @ VC.T
    # We want to maximize total similarity; linear_sum_assignment minimizes -> use cost=1-sim
    N = min(sim.shape[0], sim.shape[1])
    cost = 1.0 - sim[:N, :N]
    rind, cind = linear_sum_assignment(cost)  # globally optimal one-to-one assignment

    os.makedirs(args.out, exist_ok=True)
    cur = next_index(args.out) if args.start is None else args.start
    saved = 0
    for iw, ic in zip(rind, cind):
        pw, pc = wm_paths[iw], clean_paths[ic]
        base = f"img{cur:05d}"
        we, ce = os.path.splitext(pw)[1].lower(), os.path.splitext(pc)[1].lower()
        shutil.copy2(pw, os.path.join(args.out, base+"_wm"+we))
        shutil.copy2(pc, os.path.join(args.out, base+ce))
        cur += 1; saved += 1

    print(f"[OK] Wrote {saved} pairs -> {args.out}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm",    required=True, help="Folder of watermarked images")
    ap.add_argument("--clean", required=True, help="Folder of clean (non-watermarked) images")
    ap.add_argument("--out",   default="./rename_output", help="Output directory (a single folder)")
    ap.add_argument("--start", type=int, default=None, help="Force the starting index (overrides automatic numbering)")
    args = ap.parse_args()
    main(args)