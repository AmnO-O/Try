#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence for the encoder-decoder fusion stack (comment = decoder input).

The model used to fuse with bare cross-attention: `XAttnBlock` only ever sent
the comment into the video (Q=comment, K=V=video) and never let the comment's own
tokens interact. The stack is now a small Transformer decoder:

    self-attn(comment, full)  ->  cross-attn(comment -> video)  ->  FFN

This probe pins down the properties that decision rests on, on the real
`DecoderBlock` class with a tiny `d_model` (no pretrained weights, no network):

  1. self-attention is *full*, not causal: changing a later comment token must
     move the output at an earlier one. A causal mask would leave it fixed, and
     for a classifier that reads the whole comment a causal mask is simply wrong.
  2. the same self-attention actually mixes tokens (position 0 moves when only
     the last position changes) - the thing XAttnBlock did not do.
  3. `key_padding_mask` is honoured on both attentions: junk in padded comment
     slots or padded video slots cannot reach a valid output.
  4. an all-padded comment row - which the data layer rejects, but which the
     model keeps a belt for - stays finite in forward *and* backward. Bare
     self-attention over an all -inf row is NaN in both, so this is the check
     that the block guard is load-bearing.
  5. the model actually ships this block (not the old one).

Run:  python src/probe_decoder.py
Exit code 0 = every check holds.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

import model_xattn  # noqa: E402
from model_xattn import DecoderBlock, XAttnBlock  # noqa: E402

D = 16
HEADS = 4
L = 6
LV = 5
FAILURES = []


def check(ok, label, extra=''):
    print(f'    [{"OK " if ok else "FAIL"}] {label}' + (f'  {extra}' if extra else ''))
    if not ok:
        FAILURES.append(label)
    return bool(ok)


def make_block(cls=DecoderBlock, seed=0):
    torch.manual_seed(seed)
    blk = cls(D, HEADS, dropout=0.0, ffn_mult=4)
    blk.eval()
    # Default nn.Linear init is tiny at d=16, which makes a genuine attention
    # effect indistinguishable from float32 noise. Scale the dense weights up so
    # "position 0 moved" is a real signal, and keep the module gradient-free.
    with torch.no_grad():
        for name, p in blk.named_parameters():
            if p.ndim >= 2:                       # attention & FFN projections
                p.normal_(0.0, 0.5)
            elif name.endswith('bias'):
                p.zero_()                         # leave LayerNorm weight at 1
    for p in blk.parameters():
        p.requires_grad_(False)
    return blk


def run(blk, h, H_v, mask_c, mask_v):
    with torch.no_grad():
        return blk(h, H_v, mask_c, mask_v)


def main():
    torch.manual_seed(1234)
    h = torch.randn(1, L, D)
    H_v = torch.randn(1, LV, D)
    no_pad_c = torch.zeros(1, L, dtype=torch.bool)     # True = ignore
    no_pad_v = torch.zeros(1, LV, dtype=torch.bool)

    blk = make_block()
    X = make_block(XAttnBlock)

    print('=== 1. self-attention is full, not causal ===')
    # (a) directly on the weights: an earlier query must put mass on later keys.
    # A causal mask leaves that block of the matrix exactly zero, so this both
    # states the property and would catch a causal regression.
    hs = blk.norm_self(h)
    with torch.no_grad():
        _, w = blk.self_attn(hs, hs, hs, need_weights=True, average_attn_weights=False)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        _, w_causal = blk.self_attn(hs, hs, hs, attn_mask=causal,
                                    need_weights=True, average_attn_weights=False)
    late_mass = w[0, :, 0, 1:].sum().item()            # heads, query 0 -> keys 1..L-1
    late_mass_causal = w_causal[0, :, 0, 1:].sum().item()
    check(late_mass > 0.0, 'a later key carries mass for an earlier query (full attention)',
          f'mass = {late_mass:.4f}')
    check(late_mass_causal == 0.0, 'the same readout is zero under a causal mask',
          f'mass = {late_mass_causal:.4f}')

    # (b) through the block: change only the LAST token and see an earlier output
    # move. Random self-attention keys collapse onto the self position (q=k), so
    # the softmax is too sparse to show this cleanly; pin the self sublayer to
    # uniform attention (q=k=0, v=identity) so the mixing path is isolated and the
    # effect is deterministic. Cross-attn and FFN are per-position, so any
    # dependence of position 0 on the last position can only come from self-attn.
    mean_blk = make_block()
    with torch.no_grad():
        eye = torch.eye(D)
        zero = torch.zeros(D, D)
        mean_blk.self_attn.in_proj_weight.copy_(torch.cat([zero, zero, eye], 0))
        mean_blk.self_attn.in_proj_bias.zero_()
        mean_blk.self_attn.out_proj.weight.copy_(eye)
        mean_blk.self_attn.out_proj.bias.zero_()
    hb = torch.randn(1, L, D)
    hb2 = hb.clone()
    # Not a constant shift: LayerNorm is invariant to adding the same value to
    # every feature of a position, so that would vanish at norm_self and the test
    # would be measuring nothing.
    hb2[0, -1] += torch.linspace(-3.0, 3.0, D)
    out = run(mean_blk, hb, H_v, no_pad_c, no_pad_v)
    out2 = run(mean_blk, hb2, H_v, no_pad_c, no_pad_v)
    d0 = (out2[0, 0] - out[0, 0]).abs().max().item()
    dlast = (out2[0, -1] - out[0, -1]).abs().max().item()
    check(d0 > 1e-3, 'a later token moves an earlier block output',
          f'|d pos0| = {d0:.3e}')
    check(dlast > 1e-3, 'the perturbed position itself moves', f'|d pos_last| = {dlast:.3e}')

    print()
    print('=== 2. the old cross-only block does NOT mix comment tokens ===')
    # XAttnBlock signature is (h_q, h_kv, kv_pad_mask): self-mixing is absent by
    # construction, so position 0 is unchanged when only a later token changes.
    with torch.no_grad():
        xo = X(hb, H_v, no_pad_v)
        xo2 = X(hb2, H_v, no_pad_v)
    xd0 = (xo2[0, 0] - xo[0, 0]).abs().max().item()
    check(xd0 < 1e-9, 'XAttnBlock leaves position 0 fixed (no self-attn)',
          f'|d pos0| = {xd0:.3e}')
    check(d0 > xd0 + 1e-3, 'DecoderBlock adds the intra-comment mixing that was missing',
          f'{xd0:.3e} -> {d0:.3e}')

    print()
    print('=== 3. key_padding_mask is honoured on both attentions ===')
    # (a) padded COMMENT slots cannot reach a valid output.
    mask_c = no_pad_c.clone()
    mask_c[0, 3] = True
    mask_c[0, 4] = True                                # positions 3,4 are padding
    clean = h.clone()
    junk = h.clone()
    junk[0, 3] = 99.0
    junk[0, 4] = -77.0                                 # garbage in the padded slots
    oc = run(blk, clean, H_v, mask_c, no_pad_v)
    oj = run(blk, junk, H_v, mask_c, no_pad_v)
    valid = [0, 1, 2, 5]
    cdiff = (oj[0, valid] - oc[0, valid]).abs().max().item()
    check(cdiff < 1e-6, 'junk in padded comment slots never reaches valid positions',
          f'max |d| = {cdiff:.3e}')

    # (b) padded VIDEO slots cannot reach the comment output.
    mask_v = no_pad_v.clone()
    mask_v[0, 4] = True
    Hv_clean = H_v.clone()
    Hv_junk = H_v.clone()
    Hv_junk[0, 4] = 123.0
    vc = run(blk, h, Hv_clean, no_pad_c, mask_v)
    vj = run(blk, h, Hv_junk, no_pad_c, mask_v)
    vdiff = (vj - vc).abs().max().item()
    check(vdiff < 1e-6, 'junk in padded video slots never reaches the comment',
          f'max |d| = {vdiff:.3e}')

    print()
    print('=== 4. an all-padded comment row stays finite, forward and backward ===')
    bad_c = torch.zeros(2, L, dtype=torch.bool)
    bad_c[0] = True                                    # row 0: no comment tokens
    hh = torch.randn(2, L, D, requires_grad=True)
    Hv2 = torch.randn(2, LV, D)
    out = blk(hh, Hv2, bad_c, torch.zeros(2, LV, dtype=torch.bool))
    check(bool(torch.isfinite(out[0]).all()), 'row with no comment tokens is finite in forward')
    # Backward must stay finite too: the NaN softmax in bare self-attention also
    # poisons the gradient, which is the failure that kills a training run.
    out[1].sum().backward()
    check(bool(torch.isfinite(hh.grad).all()), 'its gradient is finite (no NaN poisoning)',
          '' if hh.grad.isfinite().all() else 'NaN present')
    check(bool(torch.isfinite(out[1]).all()), 'the valid row is unaffected',
          f'|row1|max = {out[1].abs().max().item():.3e}')

    print()
    print('=== 5. the model ships DecoderBlock, not XAttnBlock ===')
    net = model_xattn.build_model(
        'sentence-transformers/paraphrase-MiniLM-L3-v2', unfreeze_layers=0,
        heads=4, layers=2, separate_encoders=False)
    net.eval()
    check(all(isinstance(b, DecoderBlock) for b in net.blocks),
          'every fusion block is a DecoderBlock', f'{len(net.blocks)} blocks')
    check(not any(isinstance(b, XAttnBlock) for b in net.blocks),
          'no bare cross-attention block remains in the stack')

    print()
    if FAILURES:
        print(f'FAILURES ({len(FAILURES)}):')
        for name in FAILURES:
            print('  -', name)
        sys.exit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()
