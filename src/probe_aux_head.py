#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit checks for the optional auxiliary hate_speech head (--hate-aux-weight).

No training, no downloads beyond the cached MiniLM backbone; random tensors plus
the real dataset wiring. Four properties:

1. `hate_aux=False` (the default) builds a model whose state dict has no
   `hate_head.*` keys: existing checkpoints and the recorded rebuild recipe are
   untouched.
2. With `hate_aux=True`, `forward(return_aux=True)` returns the main (B, 2) and
   auxiliary (B, 3) logits; the aux branch refuses the attention-dump
   combination and models built without the head.
3. One backward on `CE(main) + 0.5 * CE(aux)` reaches the pooled-feature path:
   gradients exist on both heads, the decoder blocks and the pooling query, all
   finite; the frozen encoder stays gradient-free at `unfreeze_layers=0`.
4. Dataset wiring: `PairedDataset(with_hate=True)` yields a 6-tuple with the
   mapped label, the default yields the historical 5-tuple, `collate_paired`
   stacks each shape, and an unknown `hate_speech` value raises at build time.

Run: python src/probe_aux_head.py
"""
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_xattn
from model import load_tokenizer
from train_xattn import HATE_MAP, PairedDataset, collate_paired


def tiny_df():
    return pd.DataFrame({
        'yt_comment': ['those people are all the same', 'i like this video', 'ok'],
        'yt_title': ['a title', 'another title', 'third title'],
        'yt_description': ['a description', 'another description', 'third description'],
        'st_y': [1, 0, 0],
        'hate_speech': ['yes_implicit', 'no', 'yes_explicit'],
    })


def check_dataset(tokenizer):
    ds5 = PairedDataset(tiny_df(), tokenizer, 32, 32)
    assert len(ds5[0]) == 5, 'default dataset must keep the historical 5-tuple'
    ds6 = PairedDataset(tiny_df(), tokenizer, 32, 32, with_hate=True)
    assert len(ds6[0]) == 6, 'with_hate=True must add the hate label'
    assert ds6[0][5] == HATE_MAP['yes_implicit']
    assert ds6[1][5] == HATE_MAP['no']
    b5 = collate_paired([ds5[0], ds5[1]], tokenizer.pad_token_id)
    b6 = collate_paired([ds6[0], ds6[1]], tokenizer.pad_token_id)
    assert len(b5) == 5 and len(b6) == 6
    assert b6[5].tolist() == [HATE_MAP['yes_implicit'], HATE_MAP['no']]
    bad = tiny_df()
    bad.loc[0, 'hate_speech'] = 'maybe'
    try:
        PairedDataset(bad, tokenizer, 32, 32, with_hate=True)
    except ValueError as exc:
        assert 'maybe' in str(exc)
    else:
        raise AssertionError('unknown hate_speech value must raise')
    print('  [1/3] dataset + collate wiring: OK (5-tuple / 6-tuple / error path)')


def check_model():
    torch.manual_seed(0)
    common = dict(unfreeze_layers=0, heads=4, layers=1, dropout=0.3, ffn_mult=4,
                  use_pooled_v=False, separate_encoders=False)
    net_off = model_xattn.build_model(model_xattn.DEFAULT_BACKBONE, **common)
    assert not any(k.startswith('hate_head.') for k in net_off.state_dict()), \
        'hate_aux=False must not add any parameter'
    net = model_xattn.build_model(model_xattn.DEFAULT_BACKBONE, hate_aux=True, **common)
    assert any(k.startswith('hate_head.') for k in net.state_dict()), \
        'hate_aux=True must add the auxiliary head'

    vocab = net.encoder.config.vocab_size
    b, lc, lv = 4, 16, 24
    ids_c = torch.randint(0, vocab, (b, lc)); mask_c = torch.ones_like(ids_c)
    ids_v = torch.randint(0, vocab, (b, lv)); mask_v = torch.ones_like(ids_v)
    y = torch.tensor([1, 0, 1, 0])
    h = torch.tensor([0, 1, 2, 1])

    net.eval()
    plain = net(ids_c, mask_c, ids_v, mask_v)
    assert plain.shape == (b, 2)
    logits, aux = net(ids_c, mask_c, ids_v, mask_v, return_aux=True)
    assert logits.shape == (b, 2) and aux.shape == (b, 3)
    assert torch.equal(plain, logits), 'return_aux must not change the main logits'
    try:
        net(ids_c, mask_c, ids_v, mask_v, return_aux=True, return_attn=True)
    except ValueError:
        pass
    else:
        raise AssertionError('return_aux + return_attn must raise')
    try:
        net_off(ids_c, mask_c, ids_v, mask_v, return_aux=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('return_aux on a model without the head must raise')
    print('  [2/3] forward contract: OK (shapes, parity, guarded combinations)')

    net.train()
    logits, aux = net(ids_c, mask_c, ids_v, mask_v, return_aux=True)
    loss = (torch.nn.functional.cross_entropy(logits, y)
            + 0.5 * torch.nn.functional.cross_entropy(aux, h))
    loss.backward()

    def grad_ok(mod):
        return any(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.parameters())

    assert grad_ok(net.head) and grad_ok(net.hate_head) and grad_ok(net.blocks)
    assert net.pool_query.grad is not None and torch.isfinite(net.pool_query.grad).all()
    enc_touched = any(p.grad is not None for enc in net.encoders() for p in enc.parameters())
    assert not enc_touched, 'frozen encoder must stay gradient-free at unfreeze_layers=0'
    print('  [3/3] backward: OK (both heads + blocks + pool query; frozen encoder untouched)')


def main():
    tokenizer = load_tokenizer(model_xattn.DEFAULT_BACKBONE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    print('probe_aux_head:')
    check_dataset(tokenizer)
    check_model()
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
