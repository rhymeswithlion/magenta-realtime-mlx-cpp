#!/usr/bin/env python3
"""Tokenize text with SentencePiece (no TensorFlow import).

Invoked only under the **reference** Python (``magenta_rt`` + assets installed).
"""

import os
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sentencepiece as spm
from magenta_rt import asset

prompt = sys.argv[1]
out_dir = sys.argv[2]

sp = spm.SentencePieceProcessor()
sp.Load(asset.fetch("vocabularies/musiccoca_mv212f_vocab.model"))

labels = sp.EncodeAsIds(prompt.lower())
ids = [1, *labels[:127]]
num_tokens = len(ids)
ids = ids + [0] * (128 - len(ids))
ids_np = np.array(ids, dtype=np.int32).reshape(1, 128)
pads_np = np.ones((1, 128), dtype=np.float32)
pads_np[0, :num_tokens] = 0.0

os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "sp_ids.npy"), ids_np)
np.save(os.path.join(out_dir, "sp_pads.npy"), pads_np)
print(f"Tokenized: {num_tokens} tokens")
