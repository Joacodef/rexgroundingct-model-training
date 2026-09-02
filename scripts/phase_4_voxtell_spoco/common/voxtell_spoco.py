"""
===============================================================================
MODULE:         VoxTell-SPOCO Model Architecture & Factory
LOCATION:       scripts/phase_4_voxtell_spoco/common/voxtell_spoco.py
OBJECTIVE:      Define VoxTellSpocoModel and VoxTellSpocoDecoder adapting pre-trained
                VoxTell foundation backbone with a metric embedding projection head
                producing dense continuous unit-hypersphere voxel vectors for SPOCO.
===============================================================================
"""

import json
import logging
import pydoc
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from voxtell.model.voxtell_model import VoxTellModel, VoxTellDecoder

logger = logging.getLogger("voxtell_spoco")


class VoxTellSpocoDecoder(VoxTellDecoder):
    """
    Signature:
        VoxTellSpocoDecoder(encoder, num_classes, n_conv_per_stage, deep_supervision, num_maskformer_stages=5, embedding_dim=16, ...)

    Objective:
        Subclass of VoxTellDecoder replacing the scalar einsum dot-product output
        with a 3D metric embedding head producing L2-normalized voxel vectors.

    Inputs:
        encoder: Backbone encoder (ResidualEncoder).
        num_classes (int): Number of segmentation classes (default 1).
        n_conv_per_stage: Number of convolution blocks per decoder stage.
        deep_supervision (bool): Whether to output multi-scale predictions.
        num_maskformer_stages (int): Number of stages to fuse mask embeddings (default 5).
        embedding_dim (int): Dimensionality of metric embedding space D (default 16).

    Outputs:
        torch.Tensor: Normalized 3D metric embeddings of shape (B, D, H, W, D).
    """

    def __init__(
        self,
        encoder: Any,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        num_maskformer_stages: int = 5,
        embedding_dim: int = 16,
        **kwargs: Any,
    ) -> None:
        """Initialize VoxTellSpocoDecoder with metric embedding projection head."""
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage,
            deep_supervision=deep_supervision,
            num_maskformer_stages=num_maskformer_stages,
            **kwargs,
        )
        self.embedding_dim = embedding_dim

    def forward(
        self,
        skips: List[torch.Tensor],
        mask_embeddings: List[torch.Tensor],
        return_embeddings: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Signature:
            forward(skips: list[torch.Tensor], mask_embeddings: list[torch.Tensor], return_embeddings: bool = True) -> torch.Tensor | list[torch.Tensor]

        Objective:
            Forward pass through decoder upsampling feature skips and directly
            normalizing the native 32D full-resolution feature volume to metric embedding space.

        Inputs:
            skips (list[torch.Tensor]): Encoder skip connections (bottleneck last).
            mask_embeddings (list[torch.Tensor]): Per-stage text query embeddings.
            return_embeddings (bool): Whether to return continuous metric embeddings (default True).

        Outputs:
            torch.Tensor: Continuous normalized 32D embeddings of shape (B, 32, H, W, D) or standard logits.
        """
        lres_input = skips[-1]
        seg_outputs = []
        mask_embeddings_rev = mask_embeddings[::-1]

        for stage_idx in range(len(self.stages)):
            x = self.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = self.stages[stage_idx](x)

            if stage_idx == (len(self.stages) - 1):
                # Final full-resolution stage (shape: B, 32, H, W, D)
                if return_embeddings:
                    # Direct L2 normalization of native 32D pretrained feature volume onto unit hypersphere S^31
                    embed = F.normalize(x, p=2, dim=1)
                    seg_outputs.append(embed)
                else:
                    seg_pred = torch.einsum("b c h w d, b n c -> b n h w d", x, mask_embeddings_rev[-1])
                    seg_outputs.append(seg_pred)
            elif stage_idx >= len(self.stages) - len(mask_embeddings_rev):
                mask_emb = mask_embeddings_rev.pop(0)
                batch_size, _, channels = mask_emb.shape
                mask_emb_reshaped = mask_emb.view(batch_size, self.num_heads, -1)
                fusion_features = torch.einsum(
                    "b c h w d, b n c -> b n h w d",
                    x,
                    mask_emb_reshaped,
                )
                x = torch.cat((x, fusion_features), dim=1)
                if not return_embeddings:
                    seg_outputs.append(self.seg_layers[stage_idx](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision or return_embeddings:
            return seg_outputs[0]
        return seg_outputs


class VoxTellSpocoModel(VoxTellModel):
    """
    Signature:
        VoxTellSpocoModel(input_channels=1, embedding_dim=16, ...)

    Objective:
        VoxTell foundation model adapted for SPOCO metric learning.
        Conditioned on 2560-dim Qwen text embeddings, the model maps each 3D voxel
        (z, y, x) into a continuous D-dimensional unit hypersphere vector.

    Inputs:
        input_channels (int): Input image channels (default 1 for CT).
        embedding_dim (int): Dimensionality of metric embedding space (default 16).
        text_embedding_dim (int): Dimension of input text query embeddings (default 2560).
        deep_supervision (bool): Whether deep supervision is enabled (default False).

    Outputs:
        torch.Tensor: Dense voxel embeddings of shape (B, N_prompts, D, Z, Y, X).
    """

    def __init__(
        self,
        input_channels: int = 1,
        embedding_dim: int = 16,
        deep_supervision: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize VoxTellSpocoModel replacing standard decoder with VoxTellSpocoDecoder."""
        super().__init__(
            input_channels=input_channels,
            deep_supervision=deep_supervision,
            **kwargs,
        )
        self.embedding_dim = embedding_dim

        # Replace standard VoxTell decoder with VoxTellSpocoDecoder
        n_stages = len(self.encoder.output_channels)
        n_conv_per_stage_decoder = [kwargs.get("n_conv_per_stage_decoder", 2)] * (n_stages - 1) if isinstance(kwargs.get("n_conv_per_stage_decoder", 2), int) else kwargs.get("n_conv_per_stage_decoder")
        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [2] * (n_stages - 1)

        self.decoder = VoxTellSpocoDecoder(
            encoder=self.encoder,
            num_classes=1,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            num_maskformer_stages=kwargs.get("num_maskformer_stages", 5),
            embedding_dim=embedding_dim,
            num_heads=self.num_heads,
        )

    def forward(
        self,
        img: torch.Tensor,
        text_embedding: torch.Tensor,
        return_embeddings: bool = True,
    ) -> torch.Tensor:
        """
        Signature:
            forward(img: torch.Tensor, text_embedding: torch.Tensor, return_embeddings: bool = True) -> torch.Tensor

        Objective:
            Extract 3D features, fuse text query embeddings via transformer decoder,
            and decode into full-resolution dense metric embeddings per prompt.

        Inputs:
            img (torch.Tensor): 3D CT volume tensor of shape (B, 1, Z, Y, X).
            text_embedding (torch.Tensor): Text query embeddings of shape (B, N, D_text).
            return_embeddings (bool): Whether to return metric embeddings (default True).

        Outputs:
            torch.Tensor: Dense metric embeddings of shape (B, N, D, Z, Y, X) or logits (B, N, Z, Y, X).
        """
        # 1. Multi-scale feature extraction via pre-trained ResidualEncoder
        skips = self.encoder(img)
        selected_feature = skips[self.selected_decoder_layer]

        # 2. Reshape and project bottleneck features
        bottleneck_embed = rearrange(selected_feature, "b c d h w -> b h w d c")
        bottleneck_embed = self.project_bottleneck_embed(bottleneck_embed)
        bottleneck_embed = rearrange(bottleneck_embed, "b h w d c -> (h w d) b c")

        # 3. Project text embeddings
        if text_embedding.dim() == 4:
            text_embedding = text_embedding.squeeze(2)
        text_embed = repeat(text_embedding, "b n dim -> n b dim")
        text_embed = self.project_text_embed(text_embed)

        # 4. Transformer cross-attention fusion between image and text features
        mask_embedding, _ = self.transformer_decoder(
            tgt=text_embed,
            memory=bottleneck_embed,
            pos=self.pos_embed,
            memory_key_padding_mask=None,
        )
        mask_embedding = repeat(mask_embedding, "n b dim -> b n dim")

        # 5. Project mask embeddings across resolution stages
        mask_embeddings = [
            projection(mask_embedding)
            for projection in self.project_to_decoder_channels
        ]

        # 6. Decode per text prompt query
        outs = []
        num_prompts = text_embedding.shape[1]
        for prompt_idx in range(num_prompts):
            prompt_embeds = [m[:, prompt_idx : prompt_idx + 1] for m in mask_embeddings]
            out = self.decoder(skips, prompt_embeds, return_embeddings=return_embeddings)
            outs.append(out)

        # Stack across prompt dimension: (B, N, D, Z, Y, X) or (B, N, Z, Y, X)
        return torch.stack(outs, dim=1)


def load_voxtell_spoco_model(
    model_dir: str,
    device: str,
    embedding_dim: int = 32,
    deep_supervision: bool = False,
) -> VoxTellSpocoModel:
    """
    Signature:
        load_voxtell_spoco_model(model_dir: str, device: str, embedding_dim: int = 32, deep_supervision: bool = False) -> VoxTellSpocoModel

    Objective:
        Load plans.json configuration, instantiate VoxTellSpocoModel, and load
        pre-trained foundation checkpoint weights for native 32D SPOCO metric learning.

    Inputs:
        model_dir (str): Directory containing plans.json and checkpoint_final.pth.
        device (str): Computation device string (e.g. 'cuda:1').
        embedding_dim (int): Metric embedding dimension D (default 32).
        deep_supervision (bool): Whether to enable deep supervision (default False).

    Outputs:
        VoxTellSpocoModel: Initialized VoxTellSpocoModel loaded with pre-trained weights.
    """
    model_dir_path = Path(model_dir)
    plans_file = model_dir_path / "plans.json"

    if not plans_file.exists():
        raise FileNotFoundError(f"Missing plans.json at {plans_file}")

    with open(plans_file, "r") as f:
        plans = json.load(f)

    arch_kwargs = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
    arch_kwargs = dict(**arch_kwargs)
    for required_import_key in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
        if arch_kwargs[required_import_key] is not None:
            arch_kwargs[required_import_key] = pydoc.locate(arch_kwargs[required_import_key])

    model = VoxTellSpocoModel(
        input_channels=1,
        embedding_dim=embedding_dim,
        deep_supervision=deep_supervision,
        **arch_kwargs,
        decoder_layer=4,
        text_embedding_dim=2560,
        num_maskformer_stages=5,
        num_heads=32,
        query_dim=2048,
        project_to_decoder_hidden_dim=2048,
    )

    ckpt_path = model_dir_path / "fold_0" / "checkpoint_final.pth"
    if not ckpt_path.exists():
        ckpt_path = model_dir_path / "checkpoint_final.pth"

    if ckpt_path.exists():
        logger.info(f"Loading pre-trained VoxTell backbone weights from {ckpt_path}")
        checkpoint_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint_data.get("network_weights", checkpoint_data.get("model", checkpoint_data))

        # 100% full weight loading (zero uninitialized projection layers)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Successfully loaded 100% pre-trained weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
    else:
        logger.warning(f"Pre-trained checkpoint not found at {ckpt_path}. Initializing with random weights.")

    return model.to(device)

