#!/usr/bin/env python3
import os
import math
import torch
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from models.super_generator import FlowGuidedVFI

torch.set_grad_enabled(False)


# ============================================================
# Load Generator
# ============================================================

def load_generator(model_path, device):
    G = FlowGuidedVFI().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = checkpoint.get(
        'G_state_dict',
        checkpoint.get('G', checkpoint.get('state_dict', checkpoint))
    )
    G.load_state_dict(state, strict=True)
    G.eval()
    return G


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True,
                        help='Path to vimeo_triplet root (contains sequences/ and tri_testlist.txt)')
    parser.add_argument('--model',     required=True,
                        help='Path to checkpoint (.pth)')
    parser.add_argument('--split',     default='test',
                        choices=['test', 'train'],
                        help='Which split to evaluate (default: test)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Data   : {args.data_root}")

    G = load_generator(args.model, device)

    transform = transforms.ToTensor()

    list_file = os.path.join(
        args.data_root,
        'tri_testlist.txt' if args.split == 'test' else 'tri_trainlist.txt'
    )
    with open(list_file) as f:
        triplets = [line.strip() for line in f if line.strip()]

    print(f"Split  : {args.split}")
    print(f"Samples: {len(triplets)}")
    print("Starting evaluation...\n")

    psnr_list = []

    with torch.no_grad():
        for path in tqdm(triplets):
            full = os.path.join(args.data_root, 'sequences', path)

            im1 = transform(Image.open(os.path.join(full, 'im1.png')).convert('RGB'))
            im2 = transform(Image.open(os.path.join(full, 'im2.png')).convert('RGB'))
            im3 = transform(Image.open(os.path.join(full, 'im3.png')).convert('RGB'))

            I0 = (im1.unsqueeze(0).to(device) * 2) - 1   # [0,1] → [-1,1]
            I2 = (im3.unsqueeze(0).to(device) * 2) - 1

            # FlowGuidedVFI.interpolate() returns only the output frame
            pred = G.interpolate(I0, I2)

            pred    = torch.clamp((pred + 1) / 2, 0, 1)
            pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            gt_np   = im2.numpy().transpose(1, 2, 0)

            mse  = ((gt_np - pred_np) ** 2).mean()
            psnr = -10 * math.log10(mse + 1e-8)
            psnr_list.append(psnr)

    avg   = np.mean(psnr_list)
    std   = np.std(psnr_list)
    best  = np.max(psnr_list)
    worst = np.min(psnr_list)

    print("\n==============================")
    print(f"Samples evaluated : {len(psnr_list)}")
    print(f"Average PSNR      : {avg:.4f} dB")
    print(f"Std Dev           : {std:.4f} dB")
    print(f"Best              : {best:.4f} dB")
    print(f"Worst             : {worst:.4f} dB")
    print("==============================")


if __name__ == "__main__":
    main()