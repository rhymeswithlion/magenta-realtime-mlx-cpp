#!/usr/bin/env python3

# Copyright 2026 Brian Cruz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
