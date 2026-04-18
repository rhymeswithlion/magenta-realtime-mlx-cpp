#!/usr/bin/env python3
"""TensorFlow-only steps: embed text / RVQ tokenize, decode RVQ to audio.

Invoked only under the **reference** Python. No SentencePiece import here.
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from magenta_rt import asset, utils  # noqa: E402
from magenta_rt.musiccoca import MUSICCOCA_RVQ_VAR_ORDER  # noqa: E402

mode = sys.argv[1]
data_dir = sys.argv[2]

if mode == "embed":
    encoder_path = asset.fetch("savedmodels/musiccoca_mv212f_cpu_novocab", is_dir=True)
    encoder = tf.saved_model.load(str(encoder_path))

    ids = np.load(os.path.join(data_dir, "sp_ids.npy"))
    pads = np.load(os.path.join(data_dir, "sp_pads.npy"))

    with tf.device("/cpu:0"):
        result = encoder.signatures["embed_text"](
            inputs_0=tf.constant(ids), inputs_0_1=tf.constant(pads)
        )
    style = result["contrastive_txt_embed"].numpy()[0]
    np.save(os.path.join(data_dir, "style.npy"), style)
    print(f"Style embedding: {style.shape}, sum={style.sum():.4f}")

    quant_path = asset.fetch("savedmodels/musiccoca_mv212_quant", is_dir=True)
    var_path = f"{quant_path}/variables/variables"
    rvq_depth, codebook_size, emb_dim = 12, 1024, 768
    codebooks = np.zeros((rvq_depth, codebook_size, emb_dim), dtype=np.float32)
    for k, v_name in enumerate(MUSICCOCA_RVQ_VAR_ORDER[:rvq_depth]):
        var = tf.train.load_variable(var_path, f"variables/{v_name}/.ATTRIBUTES/VARIABLE_VALUE")
        codebooks[k] = var.T
    tokens, _ = utils.rvq_quantization(style.reshape(1, -1), codebooks)
    tokens = tokens[0]
    np.save(os.path.join(data_dir, "style_tokens.npy"), tokens)
    print(f"Style tokens: {tokens.shape}")

elif mode == "decode":
    import glob

    from ref_tf_decode_lib import decode_tokens_to_waveform, load_ssv2_decoder_bundle

    bundle = load_ssv2_decoder_bundle()
    token_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npy")))
    waveforms = []
    for f in token_files:
        tokens = np.load(f)
        wf = decode_tokens_to_waveform(bundle, tokens)
        waveforms.append(wf)
        print(f"  Decoded {f}: {wf.shape}")

    for i, wf in enumerate(waveforms):
        np.save(os.path.join(data_dir, f"waveform_{i:02d}.npy"), wf)
    print(f"Decoded {len(waveforms)} chunks")
