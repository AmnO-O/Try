#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mmBERT as a mask-targeted classifier for Task A (stereotype detection).

Architecture:

    encoder (ModernBertModel)  ->  [MASK] position  ->  pretrained mlm_head  ->  Linear(768, 2)

The full `ModernBertForMaskedLM` ships a `decoder` of shape (vocab=256000, 768),
about 197M parameters that classification never needs, so only the rows used by
the readout are kept.

Two readout initialisations are supported:

  'verbalizer' : the Linear weights start as the pretrained decoder rows for the
                 label words (' yes' / ' no'). No random init at all.
  'random'     : standard Linear init. Language-neutral, which matters because
                 mmBERT is multilingual and a single shared readout has to serve
                 EN, IT and NL at once. An English verbalizer is a poor fit for
                 Dutch comments.

Both are a single Linear layer; only its initialisation differs, so the flag
costs nothing and the choice is settled by validation macro-F1.
"""

import os
import re
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
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


def build_readout(
    decoder: nn.Linear,
    readout_init: str,
    num_classes: int,
    label_ids: Optional[Sequence[int]],
) -> nn.Linear:
    """Linear readout, optionally initialised from the MLM decoder's label rows.

    With 'verbalizer' the layer starts as an exact copy of the vocab readout
    restricted to the label rows, so on the first forward pass it produces the
    same numbers a full MLM would at those two positions.

    Both variants keep a bias, so a checkpoint saved under one initialisation
    loads cleanly under the other; only the starting point differs.
    """
    if readout_init not in READOUT_INITS:
        raise ValueError(f'readout_init must be one of {READOUT_INITS}, got {readout_init!r}')

    classifier = nn.Linear(decoder.in_features, num_classes)

    if readout_init == 'random':
        return classifier

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
    use_mlm_head: bool = True,
    dropout: float = 0.1,
    tokenizer=None,
    device: Optional[torch.device] = None,
) -> MaskTargetedClassifier:
    """Builds the classifier from the MLM checkpoint, discarding the big decoder."""
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)

    mlm = AutoModelForMaskedLM.from_pretrained(model_name)
    encoder = mlm.model
    hidden_dim = encoder.config.hidden_size

    label_ids = label_token_ids(tokenizer) if readout_init == 'verbalizer' else None
    classifier = build_readout(mlm.get_output_embeddings(), readout_init, len(LABEL_CLASSES), label_ids)

    # Tiny and either pretrained or random, so it always trains.
    for param in classifier.parameters():
        param.requires_grad = True

    mlm_head = mlm.head if use_mlm_head else nn.Identity()
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
    print(f'  backbone: {state} | readout: {readout_init} | '
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