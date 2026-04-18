#!/usr/bin/env python3
"""
UNA Checkpoint Evaluator

Evaluates any checkpoint on eval images with full metrics + PNG output.
Auto-detects model type (fp32/fp16/bf16) from checkpoint.

Usage:
    python eval_checkpoint.py ck_bf16-53/UNA-16x-bf16-best.pth
    python eval_checkpoint.py ck_bf16-53/UNA-16x-bf16-best.pth --name bf16-53
    python eval_checkpoint.py UNA-16x-sft-v4.pth --images eval_amalfi.jpg eval_portrait_640.jpg
    python eval_checkpoint.py ck_fp32-01/UNA-16x-sft-best.pth --no-images
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ssim(x, y, win=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = win // 2
    g1 = x.mean(dim=1, keepdim=True)
    g2 = y.mean(dim=1, keepdim=True)
    mu1 = F.avg_pool2d(g1, win, stride=1, padding=pad)
    mu2 = F.avg_pool2d(g2, win, stride=1, padding=pad)
    s1sq = F.avg_pool2d(g1 ** 2, win, stride=1, padding=pad) - mu1 ** 2
    s2sq = F.avg_pool2d(g2 ** 2, win, stride=1, padding=pad) - mu2 ** 2
    s12 = F.avg_pool2d(g1 * g2, win, stride=1, padding=pad) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1sq + s2sq + C2)
    return (num / den).mean().item()


# ---------------------------------------------------------------------------
# Model loading — auto-detect variant
# ---------------------------------------------------------------------------

def load_any_model(path, device):
    """Load checkpoint, auto-detect fp32/fp16/bf16/v2/v2_ema variant."""
    loaded = torch.load(path, map_location='cpu', weights_only=False)

    # Detect v2 architecture (pickled AutoencoderV2WithLoss)
    if type(loaded).__name__ == 'AutoencoderV2WithLoss':
        from modeling_una_v2 import encode as _enc_v2, decode as _dec_v2
        loaded.to(device)
        loaded.eval()
        dtype = next(loaded.parameters()).dtype
        # Wrap encode/decode to match v1 signature for eval
        def encode_v2(model, x):
            bn, depth = _enc_v2(model, x)
            return bn, depth
        def decode_v2(model, bn_and_depth):
            bn, depth = bn_and_depth
            return _dec_v2(model, bn, depth)
        return loaded, dtype, lambda m, x: encode_v2(m, x), lambda m, b: decode_v2(m, b)

    # Detect v2_ema split architecture (AutoencoderWithLoss with split compress/decompress)
    if (type(loaded).__name__ == 'AutoencoderWithLoss'
            and hasattr(loaded, 'autoencoder')
            and hasattr(loaded.autoencoder, 'compressing_convs')
            and len(loaded.autoencoder.compressing_convs) > 0
            and hasattr(loaded.autoencoder.compressing_convs[0], 'expand')):
        from modeling_v2_ema import encode, decode
        loaded.to(device)
        loaded.eval()
        dtype = next(loaded.parameters()).dtype
        return loaded, dtype, encode, decode

    # Detect dtype from weights
    if isinstance(loaded, dict):
        sample = next(iter(loaded.values()))
    else:
        sample = next(loaded.parameters())
    src_dtype = sample.dtype

    # Pick the right modeling module
    if src_dtype == torch.bfloat16:
        from modeling_una_bf16 import load_model, encode, decode
        model = load_model(path, device)
    elif src_dtype == torch.float16:
        from modeling_una_fp16 import load_model, encode, decode
        model = load_model(path, device)
    else:
        # fp32 — try bf16 loader with no_bf16=True first (handles new format),
        # fall back to original modeling_una
        try:
            from modeling_una_bf16 import load_model, encode, decode
            model = load_model(path, device, no_bf16=True)
        except Exception:
            from modeling_una import load_model, encode, decode
            model = load_model(path, device)

    model.eval()
    dtype = next(model.parameters()).dtype
    return model, dtype, encode, decode


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path, dtype, device):
    """Load image, crop to multiple of 16, convert to tensor."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    nw = w // 16 * 16
    nh = h // 16 * 16
    if nw < 16 or nh < 16:
        return None, None
    if nw != w or nh != h:
        img = img.crop((0, 0, nw, nh))
    arr = np.array(img)
    if dtype == torch.float32:
        t = torch.from_numpy(arr).float().div_(255.0)
    else:
        t = torch.from_numpy(arr).to(dtype).div_(255.0)
    return img, t.permute(2, 0, 1).unsqueeze(0).to(device)


def save_image(tensor, path):
    """Save tensor to PNG with proper uint8 conversion."""
    arr = (tensor[0].cpu().float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='UNA Checkpoint Evaluator')
    parser.add_argument('checkpoint', help='Path to .pth checkpoint')
    parser.add_argument('--name', default=None,
                        help='Output folder name (default: derived from checkpoint path)')
    parser.add_argument('--eval-dir', default=None,
                        help='Eval images directory (default: assets/eval/)')
    parser.add_argument('--images', nargs='+', default=None,
                        help='Specific image filenames to evaluate')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', default='.tmp', help='Parent output directory')
    parser.add_argument('--no-images', action='store_true',
                        help='Skip saving PNG images, only print metrics')
    parser.add_argument('--max-mp', type=float, default=20.0,
                        help='Skip images larger than this (megapixels)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Derive output name
    if args.name is None:
        ck_dir = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
        ck_file = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.name = ck_dir if ck_dir not in ('.', '', 'shared', 'una_experiment') else ck_file

    # Eval directory
    if args.eval_dir is None:
        args.eval_dir = os.path.join(script_dir, 'assets', 'eval')

    # Output directory
    out_dir = os.path.join(script_dir, args.out, args.name)
    if not args.no_images:
        os.makedirs(out_dir, exist_ok=True)

    # Load model
    model, dtype, encode, decode = load_any_model(args.checkpoint, args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Model:      {total_params:,} params, dtype={dtype}')
    print()

    # Collect eval images
    if args.images:
        files = args.images
    else:
        files = sorted([f for f in os.listdir(args.eval_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # Header
    print(f'{"Image":<28} {"Size":>12} {"Fidelity":>9} {"PSNR":>8} {"SSIM":>7} '
          f'{"MSE":>10} {"CosSim":>8} {"MaxPxErr":>8}')
    print('-' * 95)

    all_fid, all_psnr, all_ssim, all_mse, all_cos = [], [], [], [], []

    for fname in files:
        fpath = os.path.join(args.eval_dir, fname)
        if not os.path.exists(fpath):
            print(f'{fname:<28} NOT FOUND')
            continue

        # Check size
        img_check = Image.open(fpath)
        w, h = img_check.size
        if w * h > args.max_mp * 1_000_000:
            print(f'{fname:<28} {w}x{h:>5}  SKIPPED (>{args.max_mp:.0f}MP)')
            continue

        img, t = load_image(fpath, dtype, args.device)
        if img is None:
            print(f'{fname:<28} FAILED to load')
            continue
        nw, nh = img.size

        with torch.no_grad():
            bn = encode(model, t)
            recon = decode(model, bn)

        # Metrics (always in float for accuracy)
        rf, tf = recon.float(), t.float()
        fid = (1 - F.l1_loss(rf, tf).item()) * 100
        mse = F.mse_loss(rf, tf).item()
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        ss = ssim(rf, tf)
        cos = F.cosine_similarity(rf.flatten().unsqueeze(0), tf.flatten().unsqueeze(0)).item()
        max_px = int(((rf - tf).abs() * 255).max().item())

        all_fid.append(fid)
        all_psnr.append(psnr)
        all_ssim.append(ss)
        all_mse.append(mse)
        all_cos.append(cos)

        print(f'{fname:<28} {nw}x{nh:>4} {fid:>8.2f}% {psnr:>7.2f} {ss:>7.4f} '
              f'{mse:>10.6f} {cos:>7.4f} {max_px:>8}')

        # Save images
        if not args.no_images:
            stem = os.path.splitext(fname)[0]
            img.save(os.path.join(out_dir, f'{stem}_original.png'))
            save_image(recon, os.path.join(out_dir, f'{stem}_{args.name}.png'))

        del t, recon
        torch.cuda.empty_cache()

    # Summary
    if all_fid:
        print('-' * 95)
        print(f'{"AVERAGE":<28} {"":>12} {np.mean(all_fid):>8.2f}% {np.mean(all_psnr):>7.2f} '
              f'{np.mean(all_ssim):>7.4f} {np.mean(all_mse):>10.6f} {np.mean(all_cos):>7.4f}')

    if not args.no_images and all_fid:
        print(f'\nOutputs saved to {out_dir}')


if __name__ == '__main__':
    main()
