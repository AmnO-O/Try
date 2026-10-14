#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does _extract_parts hand back the *pretrained* head, on every layout?

`load_model` is the prompt baseline: encoder -> pretrained MLM head -> Linear,
with the big vocab decoder freed. With `readout_init='verbalizer'` the Linear
starts as the pretrained decoder rows for the two label words, and that means
what it says only if the hidden state reaching it went through the encoder's
real pretrained transform - dense -> activation -> LayerNorm, decoder removed.

Two latent bugs were found here while measuring it (both fixed):

  * BERT: `mlm.cls.predictions` is the *container* (`BertLMPredictionHead`); the
    dense/LayerNorm pair lives one hop down in `.transform`. Passing the
    container to the head rebuild raised
    `AttributeError: 'BertLMPredictionHead' object has no attribute 'dense'`
    on every BERT-style checkpoint.
  * RoBERTa: `RobertaLMHead.forward` is dense -> gelu -> layer_norm. The old
    `_ProjectionHead(..., residual=True)` added `x` back, so the "pretrained"
    head was not the pretrained head.

Both were in paths the shipped runs do not exercise (the xattn model uses
`AutoModel` directly; the default `--model-name` is mmBERT), which is why they
survived: nothing on the happy path calls them.

Models are built from tiny configs, so this is offline and cheap.

Run:  python src/probe_extract_parts.py
Exit code 0 = every check holds.
"""

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

from transformers import (  # noqa: E402
    BertConfig, BertForMaskedLM,
    ModernBertConfig, ModernBertForMaskedLM,
    RobertaConfig, RobertaForMaskedLM,
)

from model import _extract_parts  # noqa: E402

HIDDEN = 32
FAILURES = []


def check(ok, label, extra=''):
    print(f'    [{"OK " if ok else "FAIL"}] {label}' + (f'  {extra}' if extra else ''))
    if not ok:
        FAILURES.append(label)
    return bool(ok)


def base_config():
    return dict(vocab_size=60, hidden_size=HIDDEN, num_hidden_layers=2,
                num_attention_heads=4, intermediate_size=64)


x = torch.randn(3, HIDDEN)

print('=== BERT (BertForMaskedLM) ===')
bert = BertForMaskedLM(BertConfig(**base_config())).eval()
encoder, head, decoder = _extract_parts(bert, 'tiny-bert')
with torch.no_grad():
    got, want = head(x), bert.cls.predictions.transform(x)
check(encoder is bert.bert, 'encoder is the .bert submodule')
check(got.shape == x.shape, 'head preserves the hidden size', str(tuple(got.shape)))
check(torch.allclose(got, want, atol=1e-6), 'head(x) == the pretrained transform',
      f'max |diff| {(got - want).abs().max().item():.2e}')
check(decoder is bert.get_output_embeddings(), 'the decoder is get_output_embeddings()')
check(not hasattr(bert.cls.predictions, 'dense'),
      'cls.predictions is only a container (the old code AttributeErrors here)')
check(max(p.shape[0] for _, p in head.named_parameters()) == HIDDEN,
      'no vocab-sized parameter in the head (decoder-free)')

print('\n=== RoBERTa (RobertaForMaskedLM) ===')
roberta = RobertaForMaskedLM(RobertaConfig(**base_config())).eval()
encoder, head, decoder = _extract_parts(roberta, 'tiny-roberta')
with torch.no_grad():
    got = head(x)
    want = roberta.lm_head.layer_norm(F.gelu(roberta.lm_head.dense(x)))
    residual = roberta.lm_head.layer_norm(x + F.gelu(roberta.lm_head.dense(x)))
check(encoder is roberta.roberta, 'encoder is the .roberta submodule')
check(torch.allclose(got, want, atol=1e-6), 'head(x) == dense -> gelu -> layer_norm',
      f'max |diff| {(got - want).abs().max().item():.2e}')
check(not torch.allclose(got, residual, atol=1e-3),
      'the old residual=True was a different function (this is the bug)',
      f'max |diff| {(got - residual).abs().max().item():.2e}')
check(decoder is roberta.get_output_embeddings(), 'the decoder is get_output_embeddings()')

print('\n=== ModernBERT / mmBERT (ModernBertForMaskedLM) ===')
modern = ModernBertForMaskedLM(
    ModernBertConfig(vocab_size=60, hidden_size=HIDDEN, num_hidden_layers=2,
                     num_attention_heads=4, intermediate_size=64,
                     max_position_embeddings=64,
                     pad_token_id=0, bos_token_id=1, eos_token_id=2,
                     cls_token_id=1, sep_token_id=2)).eval()
encoder, head, decoder = _extract_parts(modern, 'tiny-modernbert')
with torch.no_grad():
    got, want = head(x), modern.head(x)
check(encoder is modern.model, 'encoder is the .model submodule')
check(head is modern.head, 'the head is mlm.head itself (already decoder-free)')
check(torch.allclose(got, want, atol=1e-6), 'head(x) == mlm.head(x)',
      f'max |diff| {(got - want).abs().max().item():.2e}')
check(decoder is modern.get_output_embeddings(), 'the decoder is get_output_embeddings()')

print()
if FAILURES:
    print(f'FAILURES ({len(FAILURES)}):')
    for name in FAILURES:
        print('  -', name)
    sys.exit(1)
print('all checks passed')
