import os
import sys
import time
import json
import logging
import wandb
import math
from pathlib import Path
from typing import Iterable
import numpy as np
from tqdm import tqdm
from termcolor import cprint
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussianwm.encoder.models_ae import create_autoencoder
from gaussianwm.processor.cached_dataset import build_cached_dataset
import util.distributed_utils as distributed_utils
import util.lr_utils as lr_utils
from util.distributed_utils import NativeScalerWithGradNormCount as NativeScaler


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm=0, log_writer=None, cfg=None):
    model.train()
    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', distributed_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    accum_iter = cfg.train.accum_iter
    kl_weight = 1e-3

    optimizer.zero_grad()

    if log_writer is not None:
        cprint(f'log_dir: {log_writer.log_dir}', 'green')

    for data_iter_step, points in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        points = points.to(device, non_blocking=True)
        labels = points.clone()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            outputs = model(points, points)

            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            outputs = outputs['logits']
            loss_vol = criterion(outputs, labels)

            if loss_kl is not None:
                loss = loss_vol + kl_weight * loss_kl
            else:
                loss = loss_vol

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_vol=loss_vol.item())

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = distributed_utils.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, cfg):
    model.eval()
    criterion = torch.nn.MSELoss()

    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for points in metric_logger.log_every(data_loader, 50, header):
        points = points.to(device, non_blocking=True)
        labels = points.clone()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            outputs = model(points, points)

            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            outputs = outputs['logits']
            loss_vol = criterion(outputs, labels)

            if loss_kl is not None:
                loss = loss_vol + 1e-3 * loss_kl
            else:
                loss = loss_vol

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_vol=loss_vol.item())

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

    metric_logger.synchronize_between_processes()
    print(f"* Eval loss: {metric_logger.loss.global_avg:.6f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@hydra.main(version_base=None, config_path="../configs", config_name="train_vae")
def main(cfg: DictConfig):
    cfg.distributed.distributed = cfg.distributed.world_size > 1

    if cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    distributed_utils.init_distributed_mode(cfg.distributed)

    logger = logging.getLogger(__name__)
    logger.info(f'Job dir: {os.path.dirname(os.path.realpath(__file__))}')
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)

    if cfg.use_wandb and distributed_utils.is_main_process():
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name or cfg.model.name,
            sync_tensorboard=True
        )

    if cfg.seed is not None:
        seed = cfg.seed + distributed_utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

    cudnn.benchmark = True

    # Load cached point cloud datasets
    dataset_train = build_cached_dataset('train', cfg)
    dataset_val = build_cached_dataset('val', cfg)

    logger.info(f'Train dataset size: {len(dataset_train)}')
    logger.info(f'Val dataset size: {len(dataset_val)}')

    is_main_process = distributed_utils.is_main_process()

    if is_main_process and cfg.log_dir is not None and not cfg.eval_only:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=cfg.log_dir)
        cprint(f"Log directory: {cfg.log_dir}", "green")
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )

    model = create_autoencoder(
        depth=cfg.vae.vae_depth,
        dim=cfg.vae.latent_dim,
        M=cfg.vae.num_latents,
        latent_dim=cfg.vae.latent_dim,
        output_dim=cfg.vae.output_dim,
        N=cfg.vae.point_cloud_size,
        deterministic=not cfg.vae.use_kl
    ).to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model: {model_without_ddp}")
    logger.info(f'Number of params (M): {n_parameters / 1e6:.2f}')

    eff_batch_size = cfg.train.batch_size * cfg.train.accum_iter * distributed_utils.get_world_size()

    if cfg.optimizer.lr is None:
        cfg.optimizer.lr = cfg.optimizer.blr * eff_batch_size / 256

    logger.info(f'Base lr: {cfg.optimizer.lr * 256 / eff_batch_size:.2e}')
    logger.info(f'Actual lr: {cfg.optimizer.lr:.2e}')
    logger.info(f'Accumulate grad iterations: {cfg.train.accum_iter}')
    logger.info(f'Effective batch size: {eff_batch_size}')

    if cfg.distributed.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=cfg.optimizer.lr)
    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()

    logger.info(f"Criterion: {criterion}")

    distributed_utils.load_model(args=cfg, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if cfg.eval_only:
        test_stats = evaluate(model, data_loader_val, device, cfg)
        logger.info(f"Eval loss on {len(dataset_val)} test samples: {test_stats['loss']:.6f}")
        return

    logger.info(f"Starting training for {cfg.train.epochs} epochs")
    start_time = time.time()

    for epoch in range(cfg.start_epoch, cfg.train.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            cfg.optimizer.clip_grad,
            log_writer=log_writer,
            cfg=cfg
        )

        if cfg.output_dir and (epoch % cfg.train.eval_every == 0 or epoch + 1 == cfg.train.epochs):
            eval_stats = evaluate(model, data_loader_val, device, cfg)

        if cfg.output_dir and (epoch % cfg.train.save_every == 0 or epoch + 1 == cfg.train.epochs):
            distributed_utils.save_model(
                args=cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        if cfg.output_dir and is_main_process:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if cfg.use_wandb and is_main_process:
            wandb.log(log_stats)

    total_time = time.time() - start_time
    logger.info(f'Training time: {total_time / 3600:.2f} hours')

if __name__ == '__main__':
    main()
