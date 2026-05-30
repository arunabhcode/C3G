from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from torch import nn

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.step_tracker import StepTracker
from .decoder.decoder import Decoder, DepthRenderingMode
from .debug_visualizer import log_debug_visualizations, log_decoder_debug
from .encoder.encoder_vggt import EncoderVGGT
from .types import Gaussians


def compute_feature_losses(rendered, target, eps: float = 1e-8):
    """Cosine similarity loss between rendered and target features."""
    rendered_norm = rendered.norm(dim=2, keepdim=True)
    target_norm = target.norm(dim=2, keepdim=True)

    cosine_sim = (
        (rendered / rendered_norm.clamp(min=eps))
        * (target / target_norm.clamp(min=eps))
    ).sum(dim=2)
    cosine_loss = (1.0 - cosine_sim).mean()

    return cosine_loss


@dataclass
class DistillTrainCfg:
    feature_mse_loss_weight: float = 1.0
    feature_cosine_loss_weight: float = 1.0
    depth_mode: DepthRenderingMode | None = None
    context_view_loss: bool = True
    random_select_context_view: bool = False


@dataclass
class DebugDecoderCfg:
    enabled: bool = False
    sam_checkpoint: str = "./pretrained_weights/sam_vit_h.pth"
    sam_model_variant: str = "sam_vit_h"


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    weight_decay: float = 0.05
    feature_head_weight_decay: float = 0.01


class DistillationModelWrapper(LightningModule):
    """Training loop for MSE feature distillation using pre-computed SAM features."""

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        train_cfg: DistillTrainCfg,
        encoder: EncoderVGGT,
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        debug_decoder_cfg: DebugDecoderCfg | None = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.encoder = encoder
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self.sam_debug_decoder = None
        if debug_decoder_cfg and debug_decoder_cfg.enabled:
            from .sam_decoder import SAMMaskDecoderWrapper

            self.sam_debug_decoder = SAMMaskDecoderWrapper(
                debug_decoder_cfg.sam_checkpoint,
                model_variant=debug_decoder_cfg.sam_model_variant,
            )
            self.sam_debug_decoder.eval()
            for p in self.sam_debug_decoder.parameters():
                p.requires_grad = False

    @rank_zero_only
    def on_train_start(self) -> None:
        accum = self.trainer.accumulate_grad_batches or 1
        print(
            "Distillation training started: "
            f"max_steps={self.trainer.max_steps}, "
            f"accumulate_grad_batches={accum}, "
            f"log_every_n_steps={self.trainer.log_every_n_steps}",
            flush=True,
        )

    def _downsample_for_encoder(self, sam_features, h, w):
        """Downsample SAM features from 64x64 to patch resolution for the encoder.

        The InstillTransformer requires the context_feature sequence length to
        match the backbone patch-token sequence length.  The backbone produces
        (H/patch_size)*(W/patch_size) tokens per view, while raw SAM features
        have 64*64 = 4096 tokens per view.  We bilinearly interpolate to bridge
        this gap (same as the live-SAM pipeline does in
        ModelWrapper.forward_foundation_model with ``interpolate=True``).
        """
        b, v, c, fh, fw = sam_features.shape
        patch_size = self.encoder.patch_size
        patch_h = h // patch_size
        patch_w = w // patch_size
        if fh == patch_h and fw == patch_w:
            return sam_features
        flat = rearrange(sam_features, "b v c h w -> (b v) c h w")
        flat = F.interpolate(
            flat, size=(patch_h, patch_w), mode="bilinear", align_corners=False
        )
        return rearrange(flat, "(b v) c h w -> b v c h w", b=b, v=v)

    def _get_valid_region(self, sam_features):
        """Return valid (non-padding) region size in the 64x64 SAM embedding space."""
        return sam_features.shape[-2], sam_features.shape[-1]

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        context_sam = batch["context"]["sam_features"]
        target_sam = batch["target"]["sam_features"]

        # Downsample context features to patch resolution for the encoder.
        # Loss targets remain at full 64x64 SAM resolution.
        context_sam_enc = self._downsample_for_encoder(context_sam, h, w)

        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            context_feature=context_sam_enc,
        )

        gaussians_detached = Gaussians(
            means=gaussians.means.detach(),
            covariances=gaussians.covariances.detach(),
            harmonics=gaussians.harmonics.detach(),
            opacities=gaussians.opacities.detach(),
            feature=gaussians.feature,
        )

        if self.train_cfg.context_view_loss:
            extrinsics = torch.cat(
                [batch["target"]["extrinsics"], batch["context"]["extrinsics"]], dim=1
            )
            intrinsics = torch.cat(
                [batch["target"]["intrinsics"], batch["context"]["intrinsics"]], dim=1
            )
            near = torch.cat([batch["target"]["near"], batch["context"]["near"]], dim=1)
            far = torch.cat([batch["target"]["far"], batch["context"]["far"]], dim=1)
        else:
            extrinsics = batch["target"]["extrinsics"]
            intrinsics = batch["target"]["intrinsics"]
            near = batch["target"]["near"]
            far = batch["target"]["far"]

        output = self.decoder.forward(
            gaussians_detached,
            extrinsics,
            intrinsics,
            near,
            far,
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        if self.train_cfg.context_view_loss:
            all_sam = torch.cat([target_sam, context_sam], dim=1)
        else:
            all_sam = target_sam

        b, v_total, c_feat, fh, fw = all_sam.shape
        rendered_features = output.feature
        rendered_interp = F.interpolate(
            rearrange(rendered_features, "b v c h w -> (b v) c h w"),
            size=(fh, fw),
            mode="bilinear",
            align_corners=False,
        )
        rendered_interp = rearrange(
            rendered_interp, "(b v) c h w -> b v c h w", b=b, v=v_total
        )

        valid_h, valid_w = self._get_valid_region(all_sam)
        rendered_crop = rendered_interp[:, :, :, :valid_h, :valid_w]
        target_crop = all_sam[:, :, :, :valid_h, :valid_w]

        rendered_normed = F.normalize(rendered_crop, p=2, dim=2)
        target_normed = F.normalize(target_crop, p=2, dim=2)
        feature_mse_loss = F.mse_loss(rendered_normed, target_normed)
        feature_cosine_loss = compute_feature_losses(rendered_crop, target_crop)

        total_loss = (
            self.train_cfg.feature_mse_loss_weight * feature_mse_loss
            + self.train_cfg.feature_cosine_loss_weight * feature_cosine_loss
        )

        for loss_name, loss_value in (
            ("loss/feature_mse", feature_mse_loss),
            ("loss/feature_cosine", feature_cosine_loss),
        ):
            self.log(
                loss_name,
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        if self.train_cfg.context_view_loss:
            target_gt = torch.cat(
                [batch["target"]["image"], ((batch["context"]["image"] + 1) / 2)], dim=1
            )
        else:
            target_gt = batch["target"]["image"]

        for loss_fn in self.losses:
            loss = loss_fn.forward(
                output,
                batch,
                gaussians_detached,
                self.global_step,
                target_image=target_gt,
            )
            self.log(
                f"loss/{loss_fn.name}",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            total_loss = total_loss + loss

        self.log(
            "loss/total",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        accum = self.trainer.accumulate_grad_batches or 1
        if self.global_rank == 0 and (batch_idx + 1) % accum == 0:
            log_debug_visualizations(
                self.logger,
                self.global_step,
                get_cfg()["checkpointing"]["every_n_train_steps"],
                batch["target"]["image"][0, 0].detach().float(),
                target_sam[0, 0].detach().float(),
                rendered_interp[0, 0].detach().float(),
                (h, w),
                prefix="train",
                valid_region=(valid_h, valid_w),
            )
            log_decoder_debug(
                self.logger,
                self.global_step,
                get_cfg()["checkpointing"]["every_n_train_steps"],
                self.sam_debug_decoder,
                target_sam[0, 0].detach().float(),
                rendered_interp[0, 0].detach().float(),
                batch["target"]["image"][0, 0].detach().float(),
                (h, w),
                prefix="train",
                valid_region=(valid_h, valid_w),
            )
            print(
                f"train step {self.global_step} finished; "
                f"loss = {total_loss.detach().item():.6f}; "
                f"feature_mse = {feature_mse_loss.detach().item():.6f}; "
                f"feature_cosine = {feature_cosine_loss.detach().item():.6f}",
                flush=True,
            )

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        context_sam = batch["context"]["sam_features"]
        target_sam = batch["target"]["sam_features"]

        context_sam_enc = self._downsample_for_encoder(context_sam, h, w)

        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            context_feature=context_sam_enc,
        )

        gaussians_detached = Gaussians(
            means=gaussians.means.detach(),
            covariances=gaussians.covariances.detach(),
            harmonics=gaussians.harmonics.detach(),
            opacities=gaussians.opacities.detach(),
            feature=gaussians.feature,
        )

        output = self.decoder.forward(
            gaussians_detached,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

        b, v, c, fh, fw = target_sam.shape
        rendered_interp = F.interpolate(
            rearrange(output.feature, "b v c h w -> (b v) c h w"),
            size=(fh, fw),
            mode="bilinear",
            align_corners=False,
        )
        rendered_interp = rearrange(
            rendered_interp, "(b v) c h w -> b v c h w", b=b, v=v
        )

        valid_h, valid_w = self._get_valid_region(target_sam)
        rendered_crop = rendered_interp[:, :, :, :valid_h, :valid_w]
        target_crop = target_sam[:, :, :, :valid_h, :valid_w]

        rendered_normed = F.normalize(rendered_crop, p=2, dim=2)
        target_normed = F.normalize(target_crop, p=2, dim=2)
        val_mse = F.mse_loss(rendered_normed, target_normed)
        val_cosine = compute_feature_losses(rendered_crop, target_crop)
        for loss_name, loss_value in (
            ("val/feature_mse", val_mse),
            ("val/feature_cosine", val_cosine),
        ):
            self.log(
                loss_name,
                loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        if self.global_rank == 0:
            log_debug_visualizations(
                self.logger,
                self.global_step,
                get_cfg()["checkpointing"]["every_n_train_steps"],
                batch["target"]["image"][0, 0],
                target_sam[0, 0],
                rendered_interp[0, 0],
                (h, w),
                valid_region=(valid_h, valid_w),
            )
            log_decoder_debug(
                self.logger,
                self.global_step,
                get_cfg()["checkpointing"]["every_n_train_steps"],
                self.sam_debug_decoder,
                target_sam[0, 0].detach().float(),
                rendered_interp[0, 0].detach().float(),
                batch["target"]["image"][0, 0].detach().float(),
                (h, w),
                prefix="val",
                valid_region=(valid_h, valid_w),
            )

    def configure_optimizers(self):
        no_decay_keywords = ("bias", "LayerNorm", "layernorm", "layer_norm", "ln")
        feature_head_keyword = "feature_gmae_to_gaussians"

        decay_params = []
        no_decay_params = []
        feature_head_decay_params = []
        feature_head_no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_feature_head = feature_head_keyword in name
            is_no_decay = any(kw in name for kw in no_decay_keywords)

            if is_feature_head:
                if is_no_decay:
                    feature_head_no_decay_params.append(param)
                else:
                    feature_head_decay_params.append(param)
            else:
                if is_no_decay:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.optimizer_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
            {
                "params": feature_head_decay_params,
                "weight_decay": self.optimizer_cfg.feature_head_weight_decay,
            },
            {"params": feature_head_no_decay_params, "weight_decay": 0.0},
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = torch.optim.AdamW(
            param_groups, lr=self.optimizer_cfg.lr, betas=(0.9, 0.95)
        )

        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer, 1 / warm_up_steps, 1, total_iters=warm_up_steps
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=get_cfg()["trainer"]["max_steps"],
            eta_min=self.optimizer_cfg.lr * 0.1,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
