#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seq2Seq cross-attention classifier for Task A (stereotype yes/no).

The key insight is that a pretrained encoder-decoder model (mBART-50, NLLB-200)
already has fully-trained cross-attention in every decoder layer.  Feeding the
video context through the *encoder* and the YouTube comment through the *decoder*
means the pretrained cross-attention query-key-value projections do the heavy
lifting of aligning the comment against the context — no scratch-initialised
DecoderBlock layers needed.

Architecture:

    video (title + description)  →  Encoder   →  H_v  (memory)
    comment                      →  Decoder   →  H_c  (cross-attends to H_v)

    H_c[<eos> position]  →  Classification head  →  2-way logit

EOS token choice
----------------
The decoder in BART / mBART / NLLB is causally-masked: token i can only attend
to tokens 0 … i.  Only the *final* token has therefore seen the entire comment,
so we extract the hidden state at the last ``</s>`` (EOS) position.

Mean-pooling mode (``--pooling mean``) extracts the average over all
non-padding decoder positions instead.

Internal structure of the Seq2Seq models
-----------------------------------------
HuggingFace wraps its seq2seq models in a two-level hierarchy:

    BartForConditionalGeneration          (self.model in Seq2SeqCrossClassifier)
    └── model : BartModel                 (self.model.model)
        ├── encoder : BartEncoder         (self.model.model.encoder)
        └── decoder : BartDecoder         (self.model.model.decoder)

mBART and NLLB follow exactly the same layout (MBartForConditionalGeneration,
NllbForConditionalGeneration).  All internal code uses ``self.model.model``
rather than ``self.model`` when it needs to reach the inner BartModel / MBartModel.

``named_parameters()`` on the outer ``Seq2SeqCrossClassifier`` therefore yields
names prefixed with ``model.model.encoder.`` and ``model.model.decoder.``, not
``model.encoder.`` or ``model.decoder.``.

Why not ``MBartForSequenceClassification`` directly?
------------------------------------------------------
Hugging Face's built-in class locates the EOS token with:

    eos_mask = input_ids.eq(self.config.eos_token_id)
    sentence_representation = hidden_states[eos_mask, :]

where ``input_ids`` is the *encoder* input.  When ``input_ids`` (context, up to
256 tokens) and ``decoder_input_ids`` (comment, up to 128 tokens) differ in
length, the mask shape mismatches the decoder ``hidden_states`` and the call
crashes.  Our wrapper indexes on ``decoder_input_ids`` instead.

Supported backbones
-------------------
Any ``AutoModelForSeq2SeqLM`` checkpoint that uses BART / mBART / M2M-100 /
NLLB conventions:

  ``facebook/mbart-large-50``            (mBART-50,  ≈610 M params)
  ``facebook/nllb-200-distilled-600M``   (NLLB-200,  ≈620 M params)
  ``facebook/nllb-200-distilled-1.3B``   (NLLB-200,  ≈1.3 B params)
  ``facebook/bart-base``                 (English only, ≈139 M, fast dev)
"""

import re
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

# Default backbone: mBART-50 covers EN, IT, NL with named language-ID tokens.
DEFAULT_BACKBONE = 'facebook/mbart-large-50'

# Language codes for the tokenizers used by the supported backbones.
# mBART-50 uses IETF-style tags; NLLB-200 uses BCP-47 + script.
MBART_LANG_CODES = {'EN': 'en_XX', 'IT': 'it_IT', 'NL': 'nl_XX', 'FA': 'fa_IR'}
NLLB_LANG_CODES  = {'EN': 'eng_Latn', 'IT': 'ita_Latn', 'NL': 'nld_Latn', 'FA': 'pes_Arab'}

_NLLB_HUBS = frozenset({
    'facebook/nllb-200-distilled-600m',
    'facebook/nllb-200-distilled-1.3b',
    'facebook/nllb-200-1.3b',
    'facebook/nllb-200-3.3b',
})


# ---------------------------------------------------------------------------
# Tokenizer family helpers
# ---------------------------------------------------------------------------

def is_nllb_backbone(backbone: str) -> bool:
    """True when the backbone uses NLLB-200 / M2M-100 lang-tag conventions."""
    return backbone.lower() in _NLLB_HUBS or 'nllb' in backbone.lower()


def lang_code_for(backbone: str, lang: str) -> str:
    """The tokenizer's language-identification token for a dataset language.

    ``lang`` is one of the two-letter codes used in the corpus filenames
    (``'EN'``, ``'IT'``, ``'NL'``, ``'FA'``).  Returns the backbone-specific
    string; raises ``KeyError`` with a clear message for unknown languages.
    """
    table = NLLB_LANG_CODES if is_nllb_backbone(backbone) else MBART_LANG_CODES
    if lang not in table:
        available = ', '.join(sorted(table))
        raise KeyError(
            f"Language {lang!r} not in the lang-code table for backbone "
            f"{backbone!r}.  Available: {available}."
        )
    return table[lang]


def backbone_requires_lang_tag(backbone: str) -> bool:
    """Whether the backbone's tokenizer needs explicit src/tgt lang codes set.

    BART (English only) uses a plain ``<s>`` BOS and does not have per-language
    tokens; mBART-50 and NLLB-200 require ``src_lang`` / ``tgt_lang``.
    """
    name = backbone.lower()
    return any(k in name for k in ('mbart', 'nllb', 'm2m'))


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def unfreeze_last_n_seq2seq(full_model: nn.Module, n_enc: int, n_dec: int) -> None:
    """Freeze the whole backbone then re-open the top N encoder/decoder layers.

    Works on any ``AutoModelForSeq2SeqLM`` instance (BART, mBART, NLLB) by
    matching parameter names with regex rather than accessing ``.encoder`` /
    ``.decoder`` attributes directly.  Both BART and mBART store the actual
    encoder/decoder under a two-level hierarchy:

        BartForConditionalGeneration.model.encoder.layers.N.*
        BartForConditionalGeneration.model.decoder.layers.N.*

    The regex ``encoder\\.layers\\.(\\d+)\\.`` and its decoder counterpart
    match this regardless of the outer wrapper class name.

    ``n_enc`` and ``n_dec`` each specify how many of the *last* encoder /
    decoder transformer blocks to re-open.  0 means keep the stack fully frozen.
    The embeddings, lm_head, and model-level layer norms are never explicitly
    unfrozen here; only the matched block parameters are opened.
    """
    # 1. Freeze everything.
    for p in full_model.parameters():
        p.requires_grad = False

    # 2. Collect per-block parameter lists by matching names.
    enc_groups: dict = {}
    dec_groups: dict = {}
    for name, param in full_model.named_parameters():
        m_enc = re.search(r'encoder\.layers\.(\d+)\.', name)
        if m_enc:
            enc_groups.setdefault(int(m_enc.group(1)), []).append(param)
            continue
        m_dec = re.search(r'decoder\.layers\.(\d+)\.', name)
        if m_dec:
            dec_groups.setdefault(int(m_dec.group(1)), []).append(param)

    # 3. Unfreeze the top N enc / dec blocks.
    for idx in (sorted(enc_groups)[-n_enc:] if n_enc > 0 else []):
        for p in enc_groups[idx]:
            p.requires_grad = True

    for idx in (sorted(dec_groups)[-n_dec:] if n_dec > 0 else []):
        for p in dec_groups[idx]:
            p.requires_grad = True


def eval_frozen_modules(module: nn.Module) -> None:
    """Switch every fully-frozen subtree to eval() so Dropout is deterministic.

    Mirrors ``model.eval_frozen_modules`` from the existing codebase.  Called
    after ``net.train()`` so frozen encoder / decoder blocks do not inject
    random dropout noise into their outputs.
    """
    frozen = {
        name: sub
        for name, sub in module.named_modules()
        if (params := list(sub.parameters(recurse=True)))
        and not any(p.requires_grad for p in params)
    }
    for name, sub in frozen.items():
        parent = name.rsplit('.', 1)[0] if '.' in name else None
        if parent is not None and parent in frozen:
            continue  # ancestor already covers this sub-tree
        sub.eval()


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class SeqClassificationHead(nn.Module):
    """Dense → Tanh → Dropout → Linear head; mirrors the Hugging Face convention.

    Applied to the scalar representation extracted from the decoder output
    (either the EOS hidden state or the mean-pooled non-padding positions).

    Parameters
    ----------
    input_dim:
        Hidden size of the encoder-decoder model (``config.d_model``).
    num_classes:
        Number of output classes (2 for stereotype yes/no).
    dropout:
        Dropout probability before and after the dense projection.
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dense    = nn.Linear(input_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout2(x)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Seq2SeqCrossClassifier(nn.Module):
    """Pretrained encoder-decoder cross-attention classifier.

    The video context (title + description) is the encoder input; the YouTube
    comment is the decoder input.  The pretrained cross-attention inside every
    decoder block aligns the comment against the context.  The hidden state at
    the comment's ``</s>`` (EOS) token (or its mean-pooled alternative) is fed
    into ``SeqClassificationHead`` to produce the 2-way stereotype logit.

    Parameters
    ----------
    backbone:
        HuggingFace model-hub ID, e.g. ``'facebook/mbart-large-50'``.
    num_classes:
        2 for yes / no stereotype classification.
    pooling:
        ``'eos'``: EOS token representation (default, recommended for causal
        decoder).  ``'mean'``: average over all non-padding decoder positions.
    dropout:
        Classification head dropout.
    unfreeze_enc:
        Number of top encoder transformer blocks to fine-tune (0 = freeze all).
    unfreeze_dec:
        Number of top decoder transformer blocks to fine-tune (default 2).
    """

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        num_classes: int = 2,
        pooling: str = 'eos',
        dropout: float = 0.1,
        unfreeze_enc: int = 0,
        unfreeze_dec: int = 2,
    ):
        super().__init__()
        if pooling not in ('eos', 'mean'):
            raise ValueError(f"pooling must be 'eos' or 'mean', got {pooling!r}")

        self.pooling       = pooling
        self.backbone_name = backbone

        # ----------------------------------------------------------------
        # Load the full encoder-decoder backbone.
        # self.model  = BartForConditionalGeneration  (or mBART / NLLB equiv)
        # self.model.model = BartModel / MBartModel   (the inner seq2seq trunk)
        # self.model.model.encoder  = BartEncoder
        # self.model.model.decoder  = BartDecoder
        # ----------------------------------------------------------------
        self.model = AutoModelForSeq2SeqLM.from_pretrained(backbone)
        self.config = self.model.config

        # d_model is the hidden size used by the encoder and decoder.
        self.d_model = self.config.d_model

        # EOS token id — used when pooling == 'eos'.
        eos_id = getattr(self.config, 'eos_token_id', None)
        if pooling == 'eos' and eos_id is None:
            raise ValueError(
                f"backbone {backbone!r} has no eos_token_id in its config.  "
                "Use --pooling mean or choose a backbone with a defined EOS."
            )
        self.eos_token_id = eos_id

        # Freeze the backbone, then selectively re-open the top N layers.
        # We pass self.model (BartForConditionalGeneration) and the helper
        # uses named_parameters() + regex — no direct .encoder / .decoder
        # attribute access that would fail on the outer wrapper class.
        unfreeze_last_n_seq2seq(self.model, unfreeze_enc, unfreeze_dec)

        self.head = SeqClassificationHead(self.d_model, num_classes, dropout)

    # ------------------------------------------------------------------
    # train() override — keeps frozen sub-trees deterministic
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """``train(True)`` re-applies the frozen-module invariant.

        After ``super().train(mode)`` walks all children and re-enables their
        Dropout, we switch every fully-frozen subtree back to eval() so the
        frozen encoder / decoder blocks are deterministic.  Mirrors the same
        pattern in ``XAttnClassifier.train``.
        """
        super().train(mode)
        if mode:
            eval_frozen_modules(self.model)
        return self

    # ------------------------------------------------------------------
    # Representation extraction
    # ------------------------------------------------------------------

    def _extract_repr(
        self,
        decoder_hidden: torch.Tensor,
        comment_ids: torch.Tensor,
        comment_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Extract a single (B, d_model) vector from the decoder output.

        ``decoder_hidden`` has shape (B, L_dec, d_model).

        EOS mode
        --------
        Finds the position of the *last* EOS token in each row of
        ``comment_ids`` and gathers the corresponding hidden state.  Using the
        *last* EOS is safe because mBART / NLLB tokenizers always place ``</s>``
        at the very end of a tokenized sequence.

        If a row has no EOS token (e.g., a comment truncated so heavily that
        the tokenizer dropped the trailing ``</s>``), we fall back to the last
        non-padding position.

        Mean-pool mode
        --------------
        Averages hidden states at all non-padding positions.
        """
        B, L, D = decoder_hidden.shape

        if self.pooling == 'mean':
            if comment_mask is not None:
                mask  = comment_mask.to(decoder_hidden.dtype)               # (B, L)
                denom = mask.sum(1, keepdim=True).clamp(min=1.0)
                return (decoder_hidden * mask.unsqueeze(-1)).sum(1) / denom  # (B, D)
            return decoder_hidden.mean(1)

        # ---------- EOS mode ----------
        eos_mask = comment_ids.eq(self.eos_token_id)                         # (B, L)

        # Fallback for rows with no EOS: take the last non-padding position.
        no_eos_rows = ~eos_mask.any(dim=1)                                   # (B,) bool
        if bool(no_eos_rows.any()):
            if comment_mask is not None:
                last_real = comment_mask.sum(1) - 1                          # (B,)
            else:
                last_real = torch.full((B,), L - 1, dtype=torch.long,
                                       device=comment_ids.device)
            last_real = last_real.clamp(min=0)
            for b in no_eos_rows.nonzero(as_tuple=False).flatten():
                eos_mask[b, int(last_real[b])] = True

        # Among all EOS positions in a row take the *last* one.
        eos_positions = (
            eos_mask.long() * torch.arange(L, device=eos_mask.device).unsqueeze(0)
        ).max(dim=1).values                                                  # (B,)

        batch_idx = torch.arange(B, device=decoder_hidden.device)
        return decoder_hidden[batch_idx, eos_positions]                      # (B, D)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        context_ids:  torch.LongTensor,
        context_mask: torch.LongTensor,
        comment_ids:  torch.LongTensor,
        comment_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """Run the encoder-decoder and return 2-way logits.

        We call the *inner* seq2seq model (``self.model.model``, i.e. BartModel /
        MBartModel) rather than the outer ``BartForConditionalGeneration`` wrapper.
        The wrapper would apply the large lm_head projection (vocab × d_model)
        which we do not need and which wastes compute and memory.

        The inner model's ``forward()`` returns ``Seq2SeqModelOutput`` whose
        ``last_hidden_state`` is the *decoder* hidden state (B, L_dec, d_model),
        matching the HuggingFace API for BartModel, MBartModel, and their kin.

        Note on gradients: even when the encoder is fully frozen, we do NOT wrap
        it in ``torch.no_grad()``.  The decoder's cross-attention reads the
        encoder's output as keys/values; if those tensors were detached, the
        classification loss gradient could not flow back through the top unfrozen
        decoder layers' cross-attention.  Parameters with ``requires_grad=False``
        simply do not accumulate gradients, but the activation tensors must remain
        in the computation graph.

        Parameters
        ----------
        context_ids, context_mask:
            Encoder input — video title + description, shape (B, L_enc).
        comment_ids, comment_mask:
            Decoder input — YouTube comment, shape (B, L_dec).

        Returns
        -------
        logits : (B, num_classes)
        """
        outputs = self.model.model(
            input_ids=context_ids,
            attention_mask=context_mask,
            decoder_input_ids=comment_ids,
            decoder_attention_mask=comment_mask,
        )
        # outputs.last_hidden_state  →  decoder hidden states (B, L_dec, D)
        repr_vec = self._extract_repr(
            outputs.last_hidden_state,
            comment_ids,
            comment_mask,
        )
        return self.head(repr_vec)


# ---------------------------------------------------------------------------
# Parameter group utilities
# ---------------------------------------------------------------------------

# Parameter prefixes as seen from the outer Seq2SeqCrossClassifier.
# Because self.model is BartForConditionalGeneration and self.model.model is
# BartModel, the full chain from the classifier root is 'model.model.encoder.'
# and 'model.model.decoder.'.  Anything else (lm_head, shared embeddings, and
# the classification head) that is trainable falls into the head group.
_ENC_PREFIX  = 'model.model.encoder.'
_DEC_PREFIX  = 'model.model.decoder.'
_HEAD_PREFIX = 'head.'


def trainable_groups(
    net: Seq2SeqCrossClassifier,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """(encoder_params, decoder_params, head_params) — trainable only.

    Three groups because they want very different step sizes:

    - **encoder** — a pretrained stack that should shift slowly (1e-5 … 2e-5).
    - **decoder** — also pretrained; the top ``--unfreeze-dec`` layers carry the
      cross-attention work; still needs a small LR (1e-5 … 2e-5).
    - **head** — ``SeqClassificationHead``, freshly initialised; fine at 1e-4.

    Frozen parameters are excluded; an empty group can be skipped when building
    the optimizer param groups.
    """
    enc_params: List[nn.Parameter] = []
    dec_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []

    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(_ENC_PREFIX):
            enc_params.append(param)
        elif name.startswith(_DEC_PREFIX):
            dec_params.append(param)
        else:
            # Covers head.*, model.lm_head.*, model.model.shared.* if unfrozen.
            head_params.append(param)

    return enc_params, dec_params, head_params


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    backbone:     str   = DEFAULT_BACKBONE,
    pooling:      str   = 'eos',
    dropout:      float = 0.1,
    unfreeze_enc: int   = 0,
    unfreeze_dec: int   = 2,
    num_classes:  int   = 2,
) -> Seq2SeqCrossClassifier:
    """Build and return the classifier in a consistent initial state.

    The returned model has:
    - Frozen encoder / decoder blocks beyond the top ``unfreeze_*`` layers.
    - The frozen subtrees already in eval() (deterministic Dropout).
    - The classification head in train() mode.

    Call ``net.eval()`` explicitly before inference.
    """
    net = Seq2SeqCrossClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pooling=pooling,
        dropout=dropout,
        unfreeze_enc=unfreeze_enc,
        unfreeze_dec=unfreeze_dec,
    )
    # net.train() walks all children (enabling Dropout everywhere), then the
    # overridden train() calls eval_frozen_modules to restore frozen subtrees
    # back to eval().
    net.train()
    return net
