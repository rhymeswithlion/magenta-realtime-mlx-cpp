#pragma once

// SpectroStream codec: STFT pre-processor + neural encoder + RVQ
// dequantizer + neural decoder + iSTFT post-processor. Both directions
// (encode and decode) are MLX-native so the streaming binary never
// leaves Metal.
//
// All MLX activations use NHWC layout (B, H, W, C). Weights loaded
// from ``.safetensors`` are stored in the canonical conv layout
// ``(out, in, kH, kW)`` and transposed to MLX's ``(out, kH, kW, in)``
// once at construction.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "magenta_realtime_mlx/dtype.h"
#include "magenta_realtime_mlx/weights.h"
#include "mlx/mlx.h"

namespace magenta_realtime_mlx {

// ---------------------------------------------------------------------------
// Padding helpers (NHWC). These are pure functions and exposed for testing.
// ---------------------------------------------------------------------------

mlx::core::array causal_same_pad(const mlx::core::array& x,
                                 std::pair<int, int> kernel_size,
                                 std::pair<int, int> stride);

mlx::core::array same_pad(const mlx::core::array& x,
                          std::pair<int, int> kernel_size,
                          std::pair<int, int> stride);

// Fold ``W`` into ``C``: (B, H, W, C) -> (B, H, W/f, C*f).
mlx::core::array fold_width(const mlx::core::array& x, int w_factor);

// Inverse of ``fold_width``: (B, H, W, C*f) -> (B, H, W*f, C).
mlx::core::array unfold_width(const mlx::core::array& x, int w_factor);

// ---------------------------------------------------------------------------
// Conv wrappers
// ---------------------------------------------------------------------------

class WNConv2d {
 public:
  // ``prefix`` is the SpectroStream key prefix in the bundle, e.g.
  // ``"input_proj.conv1"``. We expect ``prefix + ".weight"`` and
  // optionally ``prefix + ".bias"``.
  WNConv2d(const WeightBundle& bundle, std::string_view prefix,
           std::pair<int, int> kernel_size, std::pair<int, int> stride,
           bool has_bias, mlx::core::Dtype dtype);

  std::pair<int, int> kernel_size() const noexcept { return kernel_size_; }
  std::pair<int, int> stride() const noexcept { return stride_; }

  mlx::core::array operator()(const mlx::core::array& x) const;

 private:
  std::pair<int, int> kernel_size_;
  std::pair<int, int> stride_;
  mlx::core::array weight_;                 // (O, kH, kW, I)
  std::optional<mlx::core::array> bias_;    // (O,)
};

class WNConvTranspose2d {
 public:
  WNConvTranspose2d(const WeightBundle& bundle, std::string_view prefix,
                    std::pair<int, int> kernel_size,
                    std::pair<int, int> stride, bool has_bias,
                    mlx::core::Dtype dtype);

  std::pair<int, int> kernel_size() const noexcept { return kernel_size_; }
  std::pair<int, int> stride() const noexcept { return stride_; }

  mlx::core::array operator()(const mlx::core::array& x) const;

 private:
  std::pair<int, int> kernel_size_;
  std::pair<int, int> stride_;
  mlx::core::array weight_;                 // (O, kH, kW, I)
  std::optional<mlx::core::array> bias_;    // (O,)
};

// ---------------------------------------------------------------------------
// Residual blocks used by the decoder
// ---------------------------------------------------------------------------

class Res1x1Block {
 public:
  Res1x1Block(const WeightBundle& bundle, std::string_view prefix,
              int in_ch, int mid_ch, int out_ch, bool pre_elu,
              mlx::core::Dtype dtype);
  mlx::core::array operator()(const mlx::core::array& x) const;

 private:
  bool pre_elu_;
  WNConv2d conv1_;
  WNConv2d conv2_;
  WNConv2d shortcut_;
};

class Res3x3Block {
 public:
  Res3x3Block(const WeightBundle& bundle, std::string_view prefix,
              int channels, bool causal, mlx::core::Dtype dtype);
  mlx::core::array operator()(const mlx::core::array& x) const;

 private:
  bool causal_;
  WNConv2d conv1_;
  WNConv2d conv2_;
};

class DecoderResBlock {
 public:
  DecoderResBlock(const WeightBundle& bundle, std::string_view prefix,
                  int in_ch, int out_ch, std::pair<int, int> stride,
                  std::pair<int, int> transpose_kernel,
                  std::pair<int, int> post_kernel, mlx::core::Dtype dtype);
  mlx::core::array operator()(const mlx::core::array& x) const;

 private:
  std::pair<int, int> stride_;
  int crop_h_;
  int crop_w_start_;
  int crop_w_end_;
  WNConvTranspose2d conv_transpose_;
  WNConv2d conv_;
  std::optional<WNConv2d> shortcut_;
};

class EncoderResBlock {
 public:
  EncoderResBlock(const WeightBundle& bundle, std::string_view prefix,
                  int in_ch, int out_ch, std::pair<int, int> stride,
                  std::pair<int, int> pre_kernel,
                  std::pair<int, int> strided_kernel, mlx::core::Dtype dtype);
  mlx::core::array operator()(const mlx::core::array& x) const;

 private:
  std::pair<int, int> stride_;
  std::pair<int, int> pre_kernel_;
  std::pair<int, int> strided_kernel_;
  WNConv2d conv1_;
  WNConv2d conv2_;
  std::optional<WNConv2d> shortcut_;
};

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

class SpectroStreamDecoder {
 public:
  // Built from a bundle representing ``spectrostream_decoder.safetensors``.
  // ``dtype`` controls the activation/weight precision (parsed from the
  // CLI ``--dtype`` flag).
  SpectroStreamDecoder(const WeightBundle& bundle,
                       mlx::core::Dtype dtype = mlx::core::float32);

  // Forward pass. ``embeddings`` shape: (B, S, 256). Returns NHWC
  // ``(B, H, 480, 4)`` where the last axis interleaves stereo channels with
  // real/imag components: ``[s0_real, s0_imag, s1_real, s1_imag]``.
  mlx::core::array operator()(const mlx::core::array& embeddings) const;

  static constexpr int kEmbeddingDim = 256;

 private:
  static constexpr int kTemporalPad = 1;
  static constexpr int kTemporalCrop = 4;

  mlx::core::Dtype dtype_;
  Res1x1Block input_proj_;
  Res3x3Block input_conv_;
  std::vector<DecoderResBlock> decoder_blocks_;
  WNConv2d output_conv_;
};

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

class SpectroStreamEncoder {
 public:
  SpectroStreamEncoder(const WeightBundle& bundle,
                       mlx::core::Dtype dtype = mlx::core::float32);

  // ``stft_features`` shape: (B*2, H, 480, 2) NHWC (stereo batched, real/imag
  // in the channel dim). Returns embeddings of shape (B, S, 256).
  mlx::core::array operator()(const mlx::core::array& stft_features) const;

  static constexpr int kEmbeddingDim = 256;

 private:
  mlx::core::Dtype dtype_;
  WNConv2d base_conv_;
  std::vector<EncoderResBlock> encoder_blocks_;
  EncoderResBlock encoder_block_6_;
  Res3x3Block bottleneck_conv_;
  Res1x1Block bottleneck_proj_;
};

// ---------------------------------------------------------------------------
// STFT pre-processing (matches the upstream ``_stft_preprocess`` bit-for-bit)
// ---------------------------------------------------------------------------

// Convert ``(B, T, 2)`` stereo audio samples to NHWC STFT features of shape
// ``(B*2, H, 480, 2)`` where the last axis interleaves real/imag. Runs on MLX.
mlx::core::array stft_preprocess(const mlx::core::array& samples);

// ---------------------------------------------------------------------------
// iSTFT post-processing (matches the upstream ``_istft_postprocess`` bit-for-bit)
// ---------------------------------------------------------------------------

// Convert the decoder's NHWC spectral output to (B, target_length, 2) audio.
// All operations run in MLX (Metal); the function returns an ``mx::array``
// without forcing an evaluation.
mlx::core::array istft_postprocess(const mlx::core::array& spec, int B,
                                   int target_length);

}  // namespace magenta_realtime_mlx
