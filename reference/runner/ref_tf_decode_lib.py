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

"""TensorFlow SpectroStream decoder + RVQ dequant (shared by reference scripts)."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

warnings.filterwarnings("ignore")


@dataclass
class SSV2DecoderBundle:
    decoder: tf.Module
    codebooks: np.ndarray
    rvq_depth: int
    codebook_size: int
    emb_dim: int


def load_ssv2_decoder_bundle() -> SSV2DecoderBundle:
    from magenta_rt import asset, utils

    decoder = utils.load_model_cached(
        "tf", asset.fetch("savedmodels/ssv2_48k_stereo/decoder", is_dir=True)
    )
    quantizer = utils.load_model_cached(
        "tf", asset.fetch("savedmodels/ssv2_48k_stereo/quantizer", is_dir=True)
    )
    rvq_depth, codebook_size, emb_dim = 64, 1024, 256
    codebooks = np.zeros((rvq_depth, codebook_size, emb_dim), dtype=np.float32)
    cb_idx = 0
    for v in quantizer.trainable_variables:
        if "embedding" in v.name and cb_idx < rvq_depth:
            codebooks[cb_idx] = v.numpy().T
            cb_idx += 1
    return SSV2DecoderBundle(
        decoder=decoder,
        codebooks=codebooks,
        rvq_depth=rvq_depth,
        codebook_size=codebook_size,
        emb_dim=emb_dim,
    )


def decode_tokens_to_waveform(bundle: SSV2DecoderBundle, tokens: np.ndarray) -> np.ndarray:
    """Decode RVQ token rows to stereo float32 waveform (matches ``ref_tf_ops`` decode)."""
    from magenta_rt import utils

    b, s, k = 1, tokens.shape[0], tokens.shape[1]
    embeddings = utils.rvq_dequantization(tokens.reshape(-1, k), bundle.codebooks)
    embeddings = embeddings.reshape(b, s, bundle.emb_dim)
    with tf.device("/cpu:0"):
        samples = bundle.decoder(tf.constant(embeddings)).numpy()
    return np.asarray(samples[0], dtype=np.float32)
