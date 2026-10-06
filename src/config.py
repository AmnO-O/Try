#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared configuration for Task A: stereotype detection (SemEval 2027).

Single binary label: stereotype = 'yes' / 'no' on a YouTube comment plus the
title and description of the video it replies to.

The classifier is MLM-prompt based: the input ends in a [MASK] slot, its hidden
state passes through the pretrained MLM head, and a single Linear(768, 2) turns
that into the two class logits. The readout is either initialised from the
pretrained MLM decoder rows for the label words (' yes' / ' no') or randomly
initialised; see READOUT_INIT.
"""

import os

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('LGBT_DATA_DIR', os.path.join(REPO_ROOT, 'LGBT'))
WEIGHTS_DIR = os.environ.get('LGBT_WEIGHTS_DIR', os.path.join(REPO_ROOT, 'weights'))

TRAIN_GLOB = 'StereoQueerEval_*_training.tsv'
LANGUAGES = ('EN', 'IT', 'NL')

STEREOTYPE_VALUES = ('yes', 'no')

REQUIRED_COLUMNS = [
    'StereoQueerEval_id',
    'yt_title',
    'yt_description',
    'yt_comment',
    'stereotype',
    'hate_speech',
    'target',
]

# -----------------------------------------------------------------------------
# PROMPT
# -----------------------------------------------------------------------------
# 'yes' is class 1 (positive stereotype), 'no' is class 0.
MASK_LABEL = '[MASK]'
LABEL_CLASSES = ('no', 'yes')
CLASS_TO_ID = {'no': 0, 'yes': 1}

# mmBERT is multilingual, so one prompt cannot be spelled the same way in every
# language. Each entry is (template, question). {mask} is where the MLM slot goes.
#
# Field order is video -> comment -> question -> mask. Putting the question
# immediately before the slot keeps the two adjacent in the sequence, which is
# the shape MLM pretraining saw most often; it also leaves the comment as the
# last content field before the question, so anything truncated from the middle
# is video boilerplate rather than the labelled text.
PROMPT_TEMPLATES = {
    'EN': (
        'video: {video}\ncomment: {context}\nquestion: does this comment convey a stereotype about queer people? answer: {mask}',
        'does this comment convey a stereotype about queer people?',
    ),
    'IT': (
        'video: {video}\ncommento: {context}\ndomanda: questo commento veicola uno stereotipo sulle persone queer? risposta: {mask}',
        'questo commento veicola uno stereotipo sulle persone queer?',
    ),
    'NL': (
        'video: {video}\nreactie: {context}\nvraag: geeft deze reactie een stereotype weer over queer mensen? antwoord: {mask}',
        'geeft deze reactie een stereotype weer over queer mensen?',
    ),
}
DEFAULT_PROMPT = PROMPT_TEMPLATES['EN']

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
MMBERT_MODEL_NAME = 'jhu-clsp/mmbert-base'
HIDDEN_DIM = 768
MAX_LEN = 320
SEP_TOKEN = '[SEP]'

# Whether to reuse the MLM head (dense + norm) that ships with mmBERT.
USE_MLM_HEAD = True

# Initialisation of the Linear(768, 2) readout:
#   'random'     - standard init. Language-neutral, which matters because one
#                  shared readout has to serve EN, IT and NL, and English label
#                  words are a poor verbalizer for Dutch or Italian comments.
#   'verbalizer' - weights copied from the pretrained MLM decoder rows for the
#                  label words ' no' and ' yes', so the layer starts as a real
#                  vocabulary readout instead of noise.
# Both are a plain Linear and both train; pick by validation macro-F1.
READOUT_INIT = 'random'
READOUT_INITS = ('random', 'verbalizer')

# Freeze the backbone except the last N encoder blocks; 0 freezes everything.
UNFREEZE_LAYERS = 2
BACKBONE_LR = 2e-5
HEAD_LR = 1e-4

# Readout head depth. 1 = the plain Linear(768, 2); 2 = Linear(768->384) ->
# GELU -> Dropout -> Linear(384, 2). A deeper head captures more expressiveness
# but risks overfitting on ~7k rows, so the hidden width stays small.
READOUT_LAYERS = 1
READOUT_HIDDEN = 384

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 5
VAL_FRACTION = 0.1
SPLIT_SEED = 42
WEIGHT_DECAY = 0.01

BEST_WEIGHTS = os.path.join(WEIGHTS_DIR, 'best_mmbert_stereotype_prompt.pt')


def best_weights_path(readout_init: str = READOUT_INIT) -> str:
    """Checkpoint path names the readout init, so 'random' and 'verbalizer' runs
    do not overwrite each other. Training and prediction both resolve through
    this, so passing the same --readout-init at both stages lines them up."""
    return os.path.join(WEIGHTS_DIR, f'best_mmbert_stereotype_prompt_{readout_init}.pt')


# -----------------------------------------------------------------------------
# PROMPT BUDGET
# -----------------------------------------------------------------------------

class PromptBudget:
    """Word budget for one rendered prompt.

    The [MASK] slot sits at the end of every prompt, so a prompt that overflows
    max_len gets its slot truncated away and the sample becomes unusable. The
    budget converts max_len tokens to words (this corpus runs about 1.35 tokens
    per word, diacritics included) with a small reserve for special tokens.
    """

    def __init__(self, max_len: int = MAX_LEN, tokens_per_word: float = 1.35, reserve: int = 24):
        self.total_words = max(16, int(max_len / tokens_per_word) - reserve)
        self.comment_words = max(8, int(self.total_words * 0.6))