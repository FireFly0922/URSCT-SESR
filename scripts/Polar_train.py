from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pytorch_msssim import MS_SSIM, ssim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from dataset.polar_data_loader import PolarizationDataset, split_scene_ids
from loss.Charbonnier_Loss import L1_Charbonnier_loss as CharbonnierLoss
from loss.Gradient_Loss import Gradient_Loss
from model.URSCT_model import URSCT


def parse_args():
    parser = argparse.ArgumentParser(description="Train URSCT on polarization triplets")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "Polar_Enh_opt.yaml",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device, normally 'cuda'; use 'cpu' only for debugging",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from a checkpoint path, or from model_latest.pth when no path is given",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one train batch and one validation batch, then exit",
    )
    return parser.parse_args()


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = torch.mean((prediction - target) ** 2)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def make_scheduler(optimizer, total_epochs: int, warmup_epochs: int, min_lr: float):
    base_lr = optimizer.param_groups[0]["lr"]
    min_ratio = min_lr / base_lr

    def lr_factor(epoch_index: int) -> float:
        if epoch_index < warmup_epochs:
            return float(epoch_index + 1) / max(1, warmup_epochs)
        denominator = max(1, total_epochs - warmup_epochs - 1)
        progress = min(1.0, (epoch_index - warmup_epochs) / denominator)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, best_psnr, best_ssim):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "PSNR": best_psnr,
            "SSIM": best_ssim,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("PSNR", 0.0)),
        float(checkpoint.get("SSIM", 0.0)),
    )


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as config_file:
        opt = yaml.safe_load(config_file)

    gpu_ids = [str(index) for index in opt.get("GPU", [0])]
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(gpu_ids))

    requested_device = args.device.lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Check the NVIDIA driver and install a "
            "CUDA-enabled PyTorch wheel before training."
        )
    device = torch.device(requested_device)

    model_opt = opt["MODEL_DETAIL"]
    train_opt = opt["TRAINING"]
    optim_opt = opt["OPTIM"]
    seed = int(train_opt["GLOBAL_SEED"])
    set_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_root = resolve_repo_path(train_opt["DATA_ROOT"])
    save_root = resolve_repo_path(train_opt["SAVE_DIR"]) / train_opt["MODEL_NAME"]
    model_dir = save_root / "models"
    result_dir = save_root / "results"
    log_dir = save_root / "log"
    for directory in (model_dir, result_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = split_scene_ids(
        data_root,
        num_scenes=int(train_opt["NUM_SCENES"]),
        val_scenes=int(train_opt["VAL_SCENES"]),
        seed=int(train_opt["SPLIT_SEED"]),
    )
    manifest = {
        "data_root": str(data_root),
        "input_channels": ["0_degree_grayscale", "45_degree_grayscale", "90_degree_grayscale"],
        "train_scene_ids": train_ids,
        "val_scene_ids": val_ids,
    }
    (save_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_dataset = PolarizationDataset(
        data_root, train_ids, train_opt["TRAIN_PS"], training=True
    )
    val_dataset = PolarizationDataset(
        data_root, val_ids, train_opt["VAL_PS"], training=False
    )
    num_workers = int(train_opt["NUM_WORKERS"])
    loader_generator = torch.Generator().manual_seed(seed)
    common_loader_args = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": loader_generator,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(optim_opt["BATCH"]),
        shuffle=True,
        **common_loader_args,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **common_loader_args)

    model = URSCT(model_opt).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(optim_opt["LR_INITIAL"]),
        betas=(float(optim_opt["BETA1"]), 0.999),
        eps=1e-8,
    )
    total_epochs = int(optim_opt["EPOCHS"])
    scheduler = make_scheduler(
        optimizer,
        total_epochs=total_epochs,
        warmup_epochs=int(optim_opt["WARMUP_EPOCHS"]),
        min_lr=float(optim_opt["LR_MIN"]),
    )
    amp_enabled = bool(train_opt["AMP"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    charbonnier_loss = CharbonnierLoss().to(device)
    gradient_loss = Gradient_Loss().to(device)
    ms_ssim_loss = MS_SSIM(
        win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3
    ).to(device)

    start_epoch, best_psnr, best_ssim = 1, 0.0, 0.0
    if args.resume:
        resume_path = (
            model_dir / "model_latest.pth"
            if args.resume == "auto"
            else resolve_repo_path(args.resume)
        )
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {resume_path}")
        start_epoch, best_psnr, best_ssim = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("=" * 72, flush=True)
    print(f"device={device}; visible_gpus={torch.cuda.device_count()}", flush=True)
    print(f"input_channels=[0_gray, 45_gray, 90_gray]; target=RGB", flush=True)
    print(f"train_scenes={len(train_ids)} {train_ids}", flush=True)
    print(f"val_scenes={len(val_ids)} {val_ids}", flush=True)
    print(f"patch=256x256; batch={optim_opt['BATCH']}; epochs={start_epoch}..{total_epochs}", flush=True)
    print(f"parameters={parameter_count:,}; AMP={amp_enabled}", flush=True)
    print("=" * 72, flush=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    training_start = time.time()
    try:
        for epoch in range(start_epoch, total_epochs + 1):
            epoch_start = time.time()
            model.train()
            sums = {"total": 0.0, "charbonnier": 0.0, "ms_ssim": 0.0, "gradient": 0.0}
            train_progress = tqdm(train_loader, desc=f"epoch {epoch:03d} train", dynamic_ncols=True)

            for batch_index, (inputs, targets) in enumerate(train_progress):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    restored = model(inputs)
                    loss_charbonnier = charbonnier_loss(restored, targets)
                    loss_ms_ssim = 1.0 - ms_ssim_loss(restored, targets)
                    loss_gradient = gradient_loss(restored, targets)
                    loss_total = loss_charbonnier + loss_ms_ssim + 2.0 * loss_gradient

                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()

                sums["total"] += loss_total.item()
                sums["charbonnier"] += loss_charbonnier.item()
                sums["ms_ssim"] += loss_ms_ssim.item()
                sums["gradient"] += loss_gradient.item()
                train_progress.set_postfix(loss=f"{loss_total.item():.4f}")

                if args.smoke_test:
                    break

            train_batches = batch_index + 1
            for name, value in sums.items():
                writer.add_scalar(f"train/{name}", value / train_batches, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

            should_validate = args.smoke_test or epoch % int(train_opt["VAL_INTERVAL"]) == 0
            if should_validate:
                model.eval()
                psnr_values = []
                ssim_values = []
                with torch.no_grad():
                    for val_index, (inputs, targets) in enumerate(
                        tqdm(val_loader, desc=f"epoch {epoch:03d} val", dynamic_ncols=True)
                    ):
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=amp_enabled):
                            restored = model(inputs)
                        psnr_values.append(psnr(restored, targets).float())
                        ssim_values.append(
                            ssim(restored.clamp(0, 1).float(), targets.float(), data_range=1.0)
                        )
                        if val_index == 0:
                            save_image(
                                torch.cat((inputs[0], restored[0].clamp(0, 1), targets[0]), dim=2),
                                result_dir / f"epoch_{epoch:03d}.png",
                            )
                        if args.smoke_test:
                            break

                current_psnr = torch.stack(psnr_values).mean().item()
                current_ssim = torch.stack(ssim_values).mean().item()
                writer.add_scalar("val/PSNR", current_psnr, epoch)
                writer.add_scalar("val/SSIM", current_ssim, epoch)
                print(
                    f"validation epoch={epoch} PSNR={current_psnr:.4f} SSIM={current_ssim:.4f}",
                    flush=True,
                )

                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    save_checkpoint(
                        model_dir / "model_bestPSNR.pth",
                        model, optimizer, scheduler, scaler, epoch, best_psnr, best_ssim,
                    )
                if current_ssim > best_ssim:
                    best_ssim = current_ssim
                    save_checkpoint(
                        model_dir / "model_bestSSIM.pth",
                        model, optimizer, scheduler, scaler, epoch, best_psnr, best_ssim,
                    )

            scheduler.step()
            if args.smoke_test:
                print("Smoke test passed: train forward/backward and validation forward completed.")
                return

            save_checkpoint(
                model_dir / "model_latest.pth",
                model, optimizer, scheduler, scaler, epoch, best_psnr, best_ssim,
            )
            print(
                f"epoch={epoch:03d}/{total_epochs} "
                f"loss={sums['total'] / train_batches:.5f} "
                f"lr={optimizer.param_groups[0]['lr']:.7f} "
                f"time={time.time() - epoch_start:.1f}s",
                flush=True,
            )
    finally:
        writer.close()

    print(f"Training finished in {(time.time() - training_start) / 3600:.2f} hours.")


if __name__ == "__main__":
    main()
