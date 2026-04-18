---
license: cc-by-4.0
pretty_name: Magenta RealTime — C++ MLX bundle
tags:
- music
- audio-generation
- magenta
- mlx
- apple-silicon
- realtime
language:
- en
---

# Magenta RealTime — C++ MLX runtime bundle

This dataset is a re-packaging of
[Google's Magenta RealTime weights](https://huggingface.co/google/magenta-realtime)
for the C++ MLX runtime in
[`rhymeswithlion/magenta-realtime-mlx-cpp`](https://github.com/rhymeswithlion/magenta-realtime-mlx-cpp).

It contains exactly what `mlx-stream` needs at startup; nothing more, nothing
less. The upstream `.pt` / `.npy` checkpoints are intentionally **not**
mirrored here — they're only useful for the (Python) re-export tooling on the
project's `main` distribution.

## Contents

```
.
├── spectrostream_encoder.safetensors      (35 MB)
├── spectrostream_decoder.safetensors     (136 MB)
├── spectrostream_codebooks.safetensors    (64 MB)
├── musiccoca_encoder.safetensors         (911 MB)
├── musiccoca_codebooks.safetensors        (36 MB)
├── musiccoca_vocab.model                 (1.5 MB)
├── depthformer/
│   └── depthformer_base.safetensors     (1.33 GB)
└── mlxfn/
    ├── encode_base_bf16.mlxfn               (~5 MB)
    ├── depth_step_base_bf16_cl01.mlxfn        ...
    │   ...
    ├── depth_step_base_bf16_cl15.mlxfn      (15 files, ~4 KB each)
    ├── temporal_step_base_bf16_cl01.mlxfn     ...
    │   ...
    ├── temporal_step_base_bf16_cl49.mlxfn   (49 files, ~200 KB each)
    └── manifest.json
```

Total: 73 files, ~2.7 GB.

Most `.safetensors` are direct conversions of the upstream `.pt` / `.npy`
artefacts (tensor keys, dtypes, and storage layout preserved verbatim). The
exception is `musiccoca_encoder.safetensors`, which has been lightly
fine-tuned to better match the original (pre-conversion) MusicCoCa XLA
SavedModel's outputs — see the provenance table at the bottom.

The `mlxfn/*.mlxfn` files are
[`mx::export_function`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.export_function.html)
serialisations of the Depthformer encode + per-cache-length decode steps.
They're an optional accelerator: the C++ binary still works without them, but
falls back to a ~50 % slower capturing-lambda compile path. Weights are passed
in as flowing function arguments, so each `.mlxfn` file is small (KB-scale,
not GB).

## How to use

The runtime expects this exact layout under `.weights-cache/`. The repo's
`Makefile` does the right thing:

```bash
git clone https://github.com/rhymeswithlion/magenta-realtime-mlx-cpp
cd magenta-realtime-mlx-cpp
make mlx-stream
```

This invokes `scripts/download_weights_from_hf.py` (which snapshots this
dataset into `.weights-cache/`), builds the C++ binary, and starts streaming
audio.

To download the bundle by itself:

```bash
huggingface-cli download \
  --repo-type dataset \
  rhymeswithlion/magenta-realtime-mlx-cpp \
  --local-dir ./weights-cache
```

## License

This bundle is licensed under
[Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/legalcode),
matching the upstream
[`google/magenta-realtime`](https://huggingface.co/google/magenta-realtime)
weights.

Copyright 2025 Google LLC.

`musiccoca_encoder.safetensors` has been lightly fine-tuned to better match
the outputs of the original (pre-conversion) MusicCoCa XLA SavedModel, and
the resulting fine-tune deltas are licensed under
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0),
Copyright 2026 Brian Cruz. It is shape-compatible with the upstream encoder
and is loaded by the runtime without code changes.

The C++ MLX runtime that consumes these weights
([`rhymeswithlion/magenta-realtime-mlx-cpp`](https://github.com/rhymeswithlion/magenta-realtime-mlx-cpp))
is also licensed under
[Apache License 2.0](https://github.com/rhymeswithlion/magenta-realtime-mlx-cpp/blob/HEAD/LICENSE),
Copyright 2026 Brian Cruz.

## Magenta RealTime Terms of Use

Reproduced from the upstream
[model card](https://huggingface.co/google/magenta-realtime):

> Magenta RealTime is offered under a combination of licenses: the codebase is
> licensed under
> [Apache 2.0](https://github.com/magenta/magenta-realtime/blob/main/LICENSE),
> and the model weights under
> [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).
> In addition, we specify the following usage terms:
>
> Copyright 2025 Google LLC
>
> Use these materials responsibly and do not generate content, including
> outputs, that infringe or violate the rights of others, including rights
> in copyrighted content.
>
> Google claims no rights in outputs you generate using Magenta RealTime.
> You and your users are solely responsible for outputs and their subsequent
> uses.
>
> Unless required by applicable law or agreed to in writing, all software and
> materials distributed here under the Apache 2.0 or CC-BY licenses are
> distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
> KIND, either express or implied. See the licenses for the specific
> language governing permissions and limitations under those licenses. You
> are solely responsible for determining the appropriateness of using,
> reproducing, modifying, performing, displaying or distributing the software
> and materials, and any outputs, and assume any and all risks associated
> with your use or distribution of any of the software and materials, and
> any outputs, and your exercise of rights and permissions under the
> licenses.

## Attribution

Please attribute Google's Magenta RealTime when using this bundle:

```
@article{gdmlyria2025live,
    title={Live Music Models},
    author={Caillon, Antoine and McWilliams, Brian and Tarakajian, Cassie
            and Simon, Ian and Manco, Ilaria and Engel, Jesse and Constant,
            Noah and Li, Pen and Denk, Timo I. and Lalama, Alberto and
            Agostinelli, Andrea and Huang, Anna and Manilow, Ethan and
            Brower, George and Erdogan, Hakan and Lei, Heidi and Rolnick,
            Itai and Grishchenko, Ivan and Orsini, Manu and Kastelic, Matej
            and Zuluaga, Mauricio and Verzetti, Mauro and Dooley, Michael
            and Skopek, Ondrej and Ferrer, Rafael and Borsos, Zal{\'a}n and
            van den Oord, {\"A}aron and Eck, Douglas and Collins, Eli and
            Baldridge, Jason and Hume, Tom and Donahue, Chris and Han,
            Kehang and Roberts, Adam},
    journal={arXiv:2508.04651},
    year={2025}
}
```

## Source bundle provenance

| Artefact                                  | Source                                                    |
| ----------------------------------------- | --------------------------------------------------------- |
| `spectrostream_*.safetensors`             | Converted from upstream `spectrostream_*.pt` / `.npy`     |
| `musiccoca_encoder.safetensors`           | Converted from upstream `musiccoca_encoder.pt`, then lightly fine-tuned by Brian Cruz to better match the original (pre-conversion) MusicCoCa XLA SavedModel; fine-tune deltas Apache-2.0 |
| `musiccoca_codebooks.safetensors`         | Converted from upstream `musiccoca_codebooks.npy`         |
| `musiccoca_vocab.model`                   | Verbatim copy of upstream SentencePiece vocab             |
| `depthformer/depthformer_base.safetensors`| Converted from upstream `depthformer_base.pt`             |
| `mlxfn/*.mlxfn`                           | Traced via `mx.export_function` from the converted weights |

Conversion + export tooling is published separately and is not part of this
runtime bundle.
