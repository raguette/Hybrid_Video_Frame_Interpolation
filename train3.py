

#!/usr/bin/env python3
"""
VFI Training — FlowGuidedVFI (IFNet + SuperGenerator)
======================================================

Key changes from train3.py (SuperGenerator-only):
  • G is now FlowGuidedVFI — forward returns (out, flow, wf1, wf3, flow_list)
  • Multi-scale auxiliary warp loss trains IFNet at all 3 pyramid levels
  • CombinedLoss now receives wf1, wf3, flow for direct IFNet supervision
  • GAN training is unchanged; discriminator still sees final output vs GT

Resuming from an old SuperGenerator checkpoint
----------------------------------------------
The SuperGenerator weights inside FlowGuidedVFI are compatible with your
old best_G.pth.  To fine-tune rather than restart from scratch:

    ckpt = torch.load("checkpoints/<run>/best_G.pth", map_location=device)
    G.generator.load_state_dict(ckpt, strict=True)
    # IFNet starts from random init — it will learn quickly via warp loss

"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
import csv
import math

from models import FlowGuidedVFI, PatchDiscriminator
from models.losses3 import CombinedLoss
from models.ifnet import warp            # for multi-scale auxiliary loss
from utils.dataset import VFIDataset

torch.set_float32_matmul_precision('high')


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VFI Training — FlowGuidedVFI")

    parser.add_argument('--name',        type=str, required=True)
    parser.add_argument('--data_root',   type=str, required=True)

    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--epochs',      type=int,   default=300)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int,   default=4)

    parser.add_argument('--save_dir',    type=str, default='checkpoints')
    parser.add_argument('--log_dir',     type=str, default='logs')

    # Optional: path to old SuperGenerator weights for fine-tuning
    parser.add_argument('--pretrain_generator', type=str, default=None,
                        help='Path to a best_G.pth from the old SuperGenerator-only run.')

    # Resume from a full checkpoint saved during this training run
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint_epoch_N.pth to resume training from.')

    return parser.parse_args()


# ============================================================
# Setup
# ============================================================

def setup_environment(args):
    args.save_dir = os.path.join(args.save_dir, args.name)
    args.log_dir  = os.path.join(args.log_dir,  args.name)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    return device


# ============================================================
# Dataset
# ============================================================

def create_dataloaders(args):
    train_list = os.path.join(args.data_root, 'tri_trainlist.txt')
    test_list  = os.path.join(args.data_root, 'tri_testlist.txt')

    train_set = VFIDataset(args.data_root, train_list, is_train=True)
    val_set   = VFIDataset(args.data_root, test_list,  is_train=False)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    return train_loader, val_loader


# ============================================================
# LR Schedule (Warmup + Cosine)
# ============================================================

def get_lr(iteration, total_iters, base_lr, warmup_iters=2000):
    if iteration < warmup_iters:
        return base_lr * iteration / warmup_iters
    progress    = (iteration - warmup_iters) / max(total_iters - warmup_iters, 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * cosine_decay


# ============================================================
# GAN Phase Schedule
# ============================================================

def get_phase_config(epoch):
    if   epoch < 100: return False, 0.0
    elif epoch < 250: return True,  0.005
    else:             return True,  0.002


# ============================================================
# Multi-Scale Auxiliary Warp Loss
# ============================================================

def multiscale_warp_loss(f1, f3, gt, flow_list):
    """
    Supervise IFNet at every pyramid level by comparing warped frames to GT.

    flow_list[0] — coarsest (scale 4)
    flow_list[1] — mid     (scale 2)
    flow_list[2] — finest  (scale 1, same flow as final output)

    The finest scale is already covered by CombinedLoss via wf1/wf3 args,
    so here we only supervise the two coarser intermediate levels.
    This prevents double-counting the scale-1 loss.

    Weights decay with scale: coarser levels get lower weight since their
    flows are naturally noisier / lower resolution.
    """
    aux_loss = torch.tensor(0.0, device=f1.device)
    weights  = [0.5, 1.0]           # scale-4, scale-2  (scale-1 in main loss)

    for fl, w in zip(flow_list[:-1], weights):
        aux_wf1 = warp(f1, fl[:, :2])
        aux_wf3 = warp(f3, fl[:, 2:])
        aux_loss = aux_loss + w * (F.l1_loss(aux_wf1, gt) + F.l1_loss(aux_wf3, gt))

    return aux_loss


# ============================================================
# Train One Epoch
# ============================================================

def train_epoch(
    G, D, loader, device,
    opt_G, opt_D, loss_fn,
    epoch, base_lr, iteration, total_iters,
    csv_writer, log_file,
):
    G.train()
    D.train()

    use_gan, gan_weight = get_phase_config(epoch)

    for i, (f1, f3, gt) in enumerate(loader):
        f1, f3, gt = f1.to(device), f3.to(device), gt.to(device)

        # ── Update learning rate ────────────────────────────────────
        lr = get_lr(iteration, total_iters, base_lr)
        for g in opt_G.param_groups: g['lr'] = lr
        for g in opt_D.param_groups: g['lr'] = lr * 2

        # ── Train Discriminator ─────────────────────────────────────
        if use_gan:
            opt_D.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    fake, *_ = G(f1, f3)

                real_pred = D(gt)
                fake_pred = D(fake.detach())
                loss_D = (
                    F.relu(1.0 - real_pred).mean() +
                    F.relu(1.0 + fake_pred).mean()
                ) / 2

            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_D.step()

        else:
            loss_D = torch.tensor(0.0)

        # ── Train Generator ─────────────────────────────────────────
        opt_G.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            fake, flow, wf1, wf3, flow_list = G(f1, f3)

            # Main reconstruction + IFNet supervision at finest scale
            loss_recon = loss_fn(fake, gt, wf1=wf1, wf3=wf3, flow=flow)

            # Multi-scale auxiliary warp loss for coarser IFNet levels
            # (λ=5 because each level contributes 2 L1 terms summed, ~10 effective)
            loss_aux = 5.0 * multiscale_warp_loss(f1, f3, gt, flow_list)

            loss_G = loss_recon + loss_aux

            if use_gan:
                loss_G = loss_G + gan_weight * (-D(fake).mean())

        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt_G.step()

        iteration += 1

        # ── Logging ─────────────────────────────────────────────────
        if i % 50 == 0:
            msg = (
                f"Epoch {epoch+1} | Batch {i} | "
                f"G: {loss_G.item():.4f} | "
                f"D: {loss_D.item():.4f} | "
                f"GAN: {use_gan} | "
                f"LR: {lr:.6f}"
            )
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
            csv_writer.writerow([epoch+1, i, loss_G.item(), loss_D.item()])

    return iteration


# ============================================================
# Validation
# ============================================================

def validate(G, loader, device):
    G.eval()
    psnrs = []

    with torch.no_grad():
        for f1, f3, gt in loader:
            f1, f3, gt = f1.to(device), f3.to(device), gt.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                fake, *_ = G(f1, f3)

            mse  = F.mse_loss(fake.float(), gt.float())
            psnr = 10 * torch.log10(4.0 / (mse + 1e-8))
            psnrs.append(psnr.item())

    return float(np.mean(psnrs))


# ============================================================
# Main
# ============================================================

def main():
    args   = parse_args()
    device = setup_environment(args)

    train_loader, val_loader = create_dataloaders(args)

    # ── Model init ──────────────────────────────────────────────────
    G = FlowGuidedVFI().to(device)
    D = PatchDiscriminator().to(device)

    optimizer_G = optim.AdamW(
        G.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4
    )
    optimizer_D = optim.AdamW(
        D.parameters(), lr=args.lr * 2, betas=(0.9, 0.99), weight_decay=1e-4
    )

    loss_fn = CombinedLoss().to(device)

    start_epoch = 0
    best_psnr   = 0.0
    total_iters = args.epochs * len(train_loader)
    iteration   = 0

    # ── Resume from checkpoint (takes priority over pretrain_generator) ──
    if args.resume is not None:
        print(f"[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        optimizer_G.load_state_dict(ckpt['opt_G'])
        optimizer_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt['epoch']          # epoch is saved as epoch+1, so this is correct
        best_psnr   = ckpt.get('psnr', 0.0)
        iteration   = start_epoch * len(train_loader)
        print(f"[Resume] Resumed from epoch {start_epoch} | Best PSNR so far: {best_psnr:.2f} dB")

    # ── Optional: warm-start SuperGenerator only (ignored if --resume given) ──
    elif args.pretrain_generator is not None:
        ckpt  = torch.load(args.pretrain_generator, map_location=device)
        state = ckpt.get('G', ckpt)
        missing, unexpected = G.generator.load_state_dict(state, strict=False)
        print(f"[Pretrain] Loaded SuperGenerator weights.")
        if missing:
            print(f"  Missing keys  : {missing}")
        if unexpected:
            print(f"  Unexpected    : {unexpected}")
        print("  IFNet starts from random initialisation.")

    # ── Logging setup ───────────────────────────────────────────────
    log_file = os.path.join(args.log_dir, "train.log")
    csv_path = os.path.join(args.log_dir, "metrics.csv")

    # Append to existing CSV if resuming, otherwise start fresh
    csv_mode = 'a' if args.resume is not None else 'w'
    csv_file = open(csv_path, csv_mode, newline='')
    csv_writer = csv.writer(csv_file)
    if csv_mode == 'w':
        csv_writer.writerow(["epoch", "batch", "loss_G", "loss_D"])

    # ── Training loop ───────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        iteration = train_epoch(
            G, D, train_loader, device,
            optimizer_G, optimizer_D, loss_fn,
            epoch, args.lr, iteration, total_iters,
            csv_writer, log_file,
        )

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            psnr = validate(G, val_loader, device)
            print(f"\nValidation PSNR (Epoch {epoch+1}): {psnr:.2f} dB")

            torch.save({
                'epoch': epoch + 1,
                'G':     G.state_dict(),
                'D':     D.state_dict(),
                'opt_G': optimizer_G.state_dict(),
                'opt_D': optimizer_D.state_dict(),
                'psnr':  psnr,
            }, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth"))

            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(
                    G.state_dict(),
                    os.path.join(args.save_dir, "best_G.pth")
                )
                print(f"  ↳ New best PSNR: {best_psnr:.2f} dB — saved best_G.pth")

        print(f"Epoch {epoch+1} done in {(time.time()-t0)/60:.2f} min\n")

    csv_file.close()


if __name__ == "__main__":
    main()
