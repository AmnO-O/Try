#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mask-targeted classifier on top of any AutoModelForMaskedLM.

Architecture:

    encoder  ->  [MASK] position  ->  pretrained prediction head  ->  Linear(hidden, 2)

Works across architectures with different internals (ModernBERT keeps the
transformer under `.model`; RoBERTa/XLMRoBERTa under `.roberta` and pack the
vocab decoder inside `.lm_head`; BERT uses `.bert` / `.cls.predictions`).
_extract_parts() normalises all of them to encoder + decoder-free head + decoder.

The shadow MLM decoder (vocab x hidden, 197-256M params) is never needed for
classification, so only the label-word rows seed the readout before it is freed.

Two readout initialisations are supported:

  'verbalizer' : the Linear weights start as the pretrained decoder rows for the
                 label words (' yes' / ' no'). No random init at all.
  'random'     : standard Linear init. Language-neutral, which matters because
                 mmBERT is multilingual and a single shared readout has to serve
                 EN, IT and NL at once. An English verbalizer is a poor fit for
                 Dutch comments.

Both produce the same readout interface; only the initialisation differs, so the
flag costs nothing and the choice is settled by validation macro-F1.
"""

import os
import re
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from config import LABEL_CLASSES

READOUT_INITS = ('random', 'verbalizer')


def layer_index_in_name(name: str) -> Optional[int]:
    """Block index parsed from a parameter name, without relying on module paths.

    mmBERT / ModernBERT -> 'model.layers.21.attn...' -> 21
    BERT-style          -> 'encoder.layer.9.attention...' -> 9
    embeddings          -> None
    """
    match = re.search(r'(?:^|\.)(?:layers|layer|blocks)\.(\d+)\b', name)
    return int(match.group(1)) if match else None


def unfreeze_last_n(model: nn.Module, n_layers: int) -> int:
    """Enables gradients for the last n encoder blocks. 0 freezes the backbone.

    Block indices come from parameter names rather than from module paths, so the
    function keeps working across encoder layouts.
    """
    for param in model.parameters():
        param.requires_grad = False

    if n_layers <= 0:
        return 0

    groups = {}
    for name, param in model.named_parameters():
        index = layer_index_in_name(name)
        if index is not None:
            groups.setdefault(index, []).append(param)

    for index in sorted(groups)[-n_layers:]:
        for param in groups[index]:
            param.requires_grad = True
    return len(groups)


def label_token_ids(tokenizer, words: Sequence[str] = LABEL_CLASSES) -> List[int]:
    """Vocabulary id of each label word, as a single leading-space token.

    SentencePiece vocabularies store '_yes' rather than 'yes', so the word is
    encoded with a leading space and without special tokens to get the exact id
    whose embedding the MLM decoder was trained on.
    """
    ids = []
    for word in words:
        encoded = tokenizer.encode(f' {word}', add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Label word {word!r} is not a single token (got {encoded}). "
                "Choose label words that exist as one piece in this vocabulary, "
                "or use readout_init='random'."
            )
        ids.append(encoded[0])
    return ids


class MaskTargetedClassifier(nn.Module):
    """mmBERT encoder + pretrained MLM head + one linear readout at [MASK]."""

    def __init__(
        self,
        encoder: nn.Module,
        mlm_head: nn.Module,
        hidden_dim: int,
        num_classes: int = 2,
        use_mlm_head: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.mlm_head = mlm_head
        self.use_mlm_head = use_mlm_head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Gather the hidden state sitting under [MASK] for each row.
        batch_index = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[batch_index, mask_positions]

        if self.use_mlm_head:
            pooled = self.mlm_head(pooled)
        return self.classifier(self.dropout(pooled))


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


_ACTIVATIONS = {'gelu': F.gelu, 'gelu_new': F.gelu}


class _ProjectionHead(nn.Module):
    """The pretrained prediction head with its vocab decoder removed.

    ModernBERT ships a decoder-free head already (mlm.head). BERT/RoBERTa/
    XLMRoBERTa keep the decoder *inside* lm_head / cls.predictions, so here we
    rebuild just dense -> act -> layer_norm (RoBERTa also adds a residual) and
    discard the linear-to-vocab projection.
    """

    def __init__(self, lm_head: nn.Module, hidden_act: str = 'gelu', residual: bool = False):
        super().__init__()
        self.dense = lm_head.dense
        self.act = _ACTIVATIONS.get(hidden_act, F.gelu)
        self.layer_norm = lm_head.layer_norm
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.dense(x))
        if self.residual:
            h = x + h
        return self.layer_norm(h)


def _extract_parts(mlm, model_name: str):
    """(encoder, head, decoder) from any AutoModelForMaskedLM, agnostically.

    Different Classes hold the transformer under different attributes
    (ModernBERT: 'model'; RoBERTa/XLMRoBERTa: 'roberta'; BERT: 'bert'), and the
    pretrained prediction head may or may not embed its own linear-to-vocab
    decoder. Both differences are normalised here so load_model only ever deals
    with: an encoder without the head, a size-preserving head, and a (hidden,
    vocab) decoder for the readout init.
    """
    encoder = (getattr(mlm, 'model', None)
               or getattr(mlm, 'roberta', None)
               or getattr(mlm, 'bert', None))
    if encoder is None:
        raise ValueError(f'{model_name}: no encoder submodule found '
                         "(expected 'model', 'roberta' or 'bert').")

    decoder = mlm.get_output_embeddings()
    if decoder is None:
        raise ValueError(f'{model_name}: get_output_embeddings() returned None.')

    hidden_act = getattr(mlm.config, 'hidden_act', 'gelu')
    if hasattr(mlm, 'head'):
        head = mlm.head                      # ModernBERT: already decoder-free
    elif hasattr(mlm, 'lm_head'):
        head = _ProjectionHead(mlm.lm_head, hidden_act, residual=True)   # RoBERTa / XLMRoBERTa
    elif hasattr(mlm, 'cls') and hasattr(mlm.cls, 'predictions'):
        head = _ProjectionHead(mlm.cls.predictions, hidden_act, residual=False)  # BERT
    else:
        raise ValueError(f'{model_name}: no recognisable prediction head.')
    return encoder, head, decoder


class MLPReadout(nn.Module):
    """Two-layer readout: Linear(hidden) -> GELU -> Dropout -> Linear(2).

    The single Linear is a logistic on top of the pretrained MLM head, which
    already supplied one nonlinearity. An extra hidden layer can capture more
    language-specific decision boundaries, but on a ~7k-row corpus the risk is
    overfitting the head region, so the hidden width is kept small (384) and
    dropout goes between the layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_readout(
    decoder: nn.Linear,
    readout_init: str,
    num_classes: int,
    label_ids: Optional[Sequence[int]] = None,
    readout_layers: int = 1,
    hidden_dim: int = 768,
    dropout: float = 0.1,
) -> nn.Module:
    """Readout head: a single Linear, or a small MLP.

    With 'verbalizer' the single Linear starts as an exact copy of the vocab
    readout restricted to the label rows. That init is only defined for the
    one-layer case: an MLP's first layer is hidden-width wide, so the two
    decoder rows cannot be mapped onto it.
    """
    if readout_init not in READOUT_INITS:
        raise ValueError(f'readout_init must be one of {READOUT_INITS}, got {readout_init!r}')

    if readout_layers == 1:
        classifier = nn.Linear(decoder.in_features, num_classes)
        if readout_init == 'random':
            return classifier
    elif readout_init == 'verbalizer':
        raise ValueError(
            "readout_init='verbalizer' is only defined for a single Linear readout "
            "(readout_layers=1). Use readout_init='random' with a deeper readout."
        )
    else:
        return MLPReadout(decoder.in_features, hidden_dim, num_classes, dropout)

    if label_ids is None:
        raise ValueError("readout_init='verbalizer' needs label token ids")
    if len(label_ids) != num_classes:
        raise ValueError(f'Got {len(label_ids)} label ids for {num_classes} classes; they must align.')
    if decoder.out_features < max(label_ids) + 1:
        raise ValueError('Label id exceeds the decoder vocabulary size.')

    with torch.no_grad():
        classifier.weight.copy_(decoder.weight[list(label_ids)].detach())
        if classifier.bias is not None and decoder.bias is not None:
            classifier.bias.copy_(decoder.bias[list(label_ids)].detach())
    return classifier


def load_model(
    model_name: str,
    unfreeze_layers: int = 2,
    readout_init: str = 'random',
    readout_layers: int = 1,
    readout_hidden: int = 384,
    use_mlm_head: bool = True,
    dropout: float = 0.1,
    tokenizer=None,
    device: Optional[torch.device] = None,
) -> MaskTargetedClassifier:
    """Builds the classifier from the MLM checkpoint, discarding the big decoder."""
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)

    mlm = AutoModelForMaskedLM.from_pretrained(model_name)
    encoder, pretrained_head, decoder = _extract_parts(mlm, model_name)
    hidden_dim = decoder.in_features

    label_ids = label_token_ids(tokenizer) if readout_init == 'verbalizer' else None
    classifier = build_readout(
        decoder,
        readout_init,
        len(LABEL_CLASSES),
        label_ids,
        readout_layers=readout_layers,
        hidden_dim=readout_hidden,
        dropout=dropout,
    )

    # Tiny and either pretrained or random, so it always trains.
    for param in classifier.parameters():
        param.requires_grad = True

    mlm_head = pretrained_head if use_mlm_head else nn.Identity()
    if use_mlm_head:
        for param in mlm_head.parameters():
            param.requires_grad = True

    n_blocks = unfreeze_last_n(encoder, unfreeze_layers)

    model = MaskTargetedClassifier(
        encoder=encoder,
        mlm_head=mlm_head,
        hidden_dim=hidden_dim,
        num_classes=len(LABEL_CLASSES),
        use_mlm_head=use_mlm_head,
        dropout=dropout,
    )
    model.classifier = classifier

    if device is not None:
        model.to(device)

    # Release the 197M-parameter decoder now that the readout has been built.
    del mlm

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if unfreeze_layers <= 0:
        state = 'frozen'
    else:
        indices = sorted({
            layer_index_in_name(name)
            for name, param in model.named_parameters()
            if param.requires_grad and layer_index_in_name(name) is not None
        })
        state = f'blocks {indices} trainable (of {n_blocks} found)'
    print(f'  backbone: {state} | readout: {readout_init}/{"L" + str(readout_layers)} | '
          f'trainable {trainable / 1e6:.2f}M / {total / 1e6:.2f}M params')
    return model


def save_model(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_weights(model: nn.Module, path: str, device: Optional[torch.device] = None) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device or 'cpu'))
    return model


def trainable_parameter_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Splits trainable params into (encoder, head) so each gets its own LR."""
    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (encoder_params if name.startswith('encoder.') else head_params).append(param)
    return encoder_params, head_params