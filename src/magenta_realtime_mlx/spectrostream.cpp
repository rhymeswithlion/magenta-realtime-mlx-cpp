// Copyright 2026 Brian Cruz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "magenta_realtime_mlx/spectrostream.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

#include "magenta_realtime_mlx/nn_ops.h"
#include "mlx/fft.h"

namespace magenta_realtime_mlx {

namespace mx = mlx::core;

namespace {
// Helper: build an mx::Shape from int components (Shape elements are int32_t).
template <class... Ts>
mx::Shape S(Ts... vs) {
  return mx::Shape{static_cast<int32_t>(vs)...};
}
}  // namespace

// ---------------------------------------------------------------------------
// Padding helpers (NHWC: B, H, W, C)
// ---------------------------------------------------------------------------

mx::array causal_same_pad(const mx::array& x,
                          std::pair<int, int> kernel_size,
                          std::pair<int, int> stride) {
  const int kh = kernel_size.first;
  const int kw = kernel_size.second;
  const int sh = stride.first;
  const int sw = stride.second;
  const int iw = static_cast<int>(x.shape(2));

  const int pad_h_top = kh - sh;
  const int ow = static_cast<int>(std::ceil(static_cast<double>(iw) / sw));
  const int pad_w_total = std::max((ow - 1) * sw + kw - iw, 0);
  const int pad_w_left = pad_w_total / 2;
  const int pad_w_right = pad_w_total - pad_w_left;

  if (pad_h_top > 0 || pad_w_left > 0 || pad_w_right > 0) {
    std::vector<std::pair<int, int>> pads = {
        {0, 0}, {pad_h_top, 0}, {pad_w_left, pad_w_right}, {0, 0}};
    return mx::pad(x, pads);
  }
  return x;
}

mx::array same_pad(const mx::array& x, std::pair<int, int> kernel_size,
                   std::pair<int, int> stride) {
  const int ih = static_cast<int>(x.shape(1));
  const int iw = static_cast<int>(x.shape(2));
  const int kh = kernel_size.first;
  const int kw = kernel_size.second;
  const int sh = stride.first;
  const int sw = stride.second;
  const int oh = static_cast<int>(std::ceil(static_cast<double>(ih) / sh));
  const int ow = static_cast<int>(std::ceil(static_cast<double>(iw) / sw));
  const int pad_h = std::max((oh - 1) * sh + kh - ih, 0);
  const int pad_w = std::max((ow - 1) * sw + kw - iw, 0);
  if (pad_h > 0 || pad_w > 0) {
    std::vector<std::pair<int, int>> pads = {
        {0, 0},
        {pad_h / 2, pad_h - pad_h / 2},
        {pad_w / 2, pad_w - pad_w / 2},
        {0, 0}};
    return mx::pad(x, pads);
  }
  return x;
}

mx::array fold_width(const mx::array& x, int w_factor) {
  const int B = static_cast<int>(x.shape(0));
  const int H = static_cast<int>(x.shape(1));
  const int W = static_cast<int>(x.shape(2));
  const int C = static_cast<int>(x.shape(3));
  return mx::reshape(x, S(B, H, W / w_factor, w_factor * C));
}

mx::array unfold_width(const mx::array& x, int w_factor) {
  const int B = static_cast<int>(x.shape(0));
  const int H = static_cast<int>(x.shape(1));
  const int W = static_cast<int>(x.shape(2));
  const int C = static_cast<int>(x.shape(3));
  const int C_new = C / w_factor;
  return mx::reshape(x, S(B, H, W * w_factor, C_new));
}

// ---------------------------------------------------------------------------
// Conv wrappers
// ---------------------------------------------------------------------------

namespace {

mx::array convert_conv2d_weight(const mx::array& w, mx::Dtype dtype) {
  return mx::astype(mx::transpose(w, {0, 2, 3, 1}), dtype);
}

mx::array convert_conv_transpose2d_weight(const mx::array& w, mx::Dtype dtype) {
  return mx::astype(mx::transpose(w, {1, 2, 3, 0}), dtype);
}

mx::array load_bias(const WeightBundle& b, std::string_view prefix,
                    mx::Dtype dtype) {
  return mx::astype(b.tensor(std::string(prefix) + ".bias"), dtype);
}

}  // namespace

WNConv2d::WNConv2d(const WeightBundle& bundle, std::string_view prefix,
                   std::pair<int, int> kernel_size,
                   std::pair<int, int> stride, bool has_bias, mx::Dtype dtype)
    : kernel_size_(kernel_size),
      stride_(stride),
      weight_(convert_conv2d_weight(
          bundle.tensor(std::string(prefix) + ".weight"), dtype)) {
  if (has_bias) bias_ = load_bias(bundle, prefix, dtype);
}

mx::array WNConv2d::operator()(const mx::array& x) const {
  mx::array y = mx::conv2d(x, weight_, stride_, /*padding=*/{0, 0});
  if (bias_.has_value()) y = mx::add(y, *bias_);
  return y;
}

WNConvTranspose2d::WNConvTranspose2d(const WeightBundle& bundle,
                                     std::string_view prefix,
                                     std::pair<int, int> kernel_size,
                                     std::pair<int, int> stride, bool has_bias,
                                     mx::Dtype dtype)
    : kernel_size_(kernel_size),
      stride_(stride),
      weight_(convert_conv_transpose2d_weight(
          bundle.tensor(std::string(prefix) + ".weight"), dtype)) {
  if (has_bias) bias_ = load_bias(bundle, prefix, dtype);
}

mx::array WNConvTranspose2d::operator()(const mx::array& x) const {
  mx::array y = mx::conv_transpose2d(x, weight_, stride_, /*padding=*/{0, 0});
  if (bias_.has_value()) y = mx::add(y, *bias_);
  return y;
}

// ---------------------------------------------------------------------------
// Residual blocks
// ---------------------------------------------------------------------------

Res1x1Block::Res1x1Block(const WeightBundle& bundle, std::string_view prefix,
                         int /*in_ch*/, int /*mid_ch*/, int /*out_ch*/,
                         bool pre_elu, mx::Dtype dtype)
    : pre_elu_(pre_elu),
      conv1_(bundle, std::string(prefix) + ".conv1", {1, 1}, {1, 1}, true,
             dtype),
      conv2_(bundle, std::string(prefix) + ".conv2", {1, 1}, {1, 1}, true,
             dtype),
      shortcut_(bundle, std::string(prefix) + ".shortcut", {1, 1}, {1, 1},
                true, dtype) {}

mx::array Res1x1Block::operator()(const mx::array& x) const {
  mx::array skip = shortcut_(x);
  mx::array main = pre_elu_ ? elu(x) : x;
  main = conv1_(main);
  main = elu(main);
  main = conv2_(main);
  return mx::add(main, skip);
}

Res3x3Block::Res3x3Block(const WeightBundle& bundle, std::string_view prefix,
                         int /*channels*/, bool causal, mx::Dtype dtype)
    : causal_(causal),
      conv1_(bundle, std::string(prefix) + ".conv1", {3, 3}, {1, 1}, true,
             dtype),
      conv2_(bundle, std::string(prefix) + ".conv2", {3, 3}, {1, 1}, true,
             dtype) {}

mx::array Res3x3Block::operator()(const mx::array& x) const {
  mx::array residual = x;
  mx::array y = elu(x);
  y = conv1_(causal_ ? causal_same_pad(y, {3, 3}, {1, 1})
                     : same_pad(y, {3, 3}, {1, 1}));
  y = elu(y);
  y = conv2_(causal_ ? causal_same_pad(y, {3, 3}, {1, 1})
                     : same_pad(y, {3, 3}, {1, 1}));
  return mx::add(y, residual);
}

DecoderResBlock::DecoderResBlock(const WeightBundle& bundle,
                                 std::string_view prefix, int in_ch, int out_ch,
                                 std::pair<int, int> stride,
                                 std::pair<int, int> transpose_kernel,
                                 std::pair<int, int> post_kernel,
                                 mx::Dtype dtype)
    : stride_(stride),
      crop_h_(transpose_kernel.first - stride.first),
      crop_w_start_((transpose_kernel.second - stride.second) / 2),
      crop_w_end_((transpose_kernel.second - stride.second) -
                  ((transpose_kernel.second - stride.second) / 2)),
      conv_transpose_(bundle, std::string(prefix) + ".conv_transpose",
                      transpose_kernel, stride, true, dtype),
      conv_(bundle, std::string(prefix) + ".conv", post_kernel, {1, 1}, true,
            dtype) {
  if (in_ch != out_ch) {
    shortcut_.emplace(bundle, std::string(prefix) + ".shortcut",
                      std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1},
                      true, dtype);
  }
}

mx::array DecoderResBlock::operator()(const mx::array& x) const {
  mx::array residual = x;
  mx::array y = elu(x);
  y = conv_transpose_(y);

  if (crop_h_ > 0) {
    const int H = static_cast<int>(y.shape(1));
    y = mx::slice(y, S(0, 0, 0, 0),
                  S(y.shape(0), H - crop_h_, y.shape(2), y.shape(3)));
  }
  if (crop_w_start_ > 0 || crop_w_end_ > 0) {
    const int W = static_cast<int>(y.shape(2));
    y = mx::slice(
        y, S(0, 0, crop_w_start_, 0),
        S(y.shape(0), y.shape(1), W - crop_w_end_, y.shape(3)));
  }

  y = elu(y);
  y = conv_(causal_same_pad(y, conv_.kernel_size(), {1, 1}));

  const int sh = stride_.first;
  const int sw = stride_.second;
  if (sh != 1 || sw != 1) {
    const int H = static_cast<int>(residual.shape(1));
    const int W = static_cast<int>(residual.shape(2));
    if (sh != 1) {
      residual = mx::repeat(residual, sh, /*axis=*/1);
      residual = mx::slice(
          residual, S(0, 0, 0, 0),
          S(residual.shape(0), H * sh, residual.shape(2), residual.shape(3)));
    }
    if (sw != 1) {
      residual = mx::repeat(residual, sw, /*axis=*/2);
      residual = mx::slice(
          residual, S(0, 0, 0, 0),
          S(residual.shape(0), residual.shape(1), W * sw, residual.shape(3)));
    }
  }
  if (shortcut_.has_value()) residual = (*shortcut_)(residual);
  return mx::add(y, residual);
}

// ---------------------------------------------------------------------------
// Encoder residual block
// ---------------------------------------------------------------------------

EncoderResBlock::EncoderResBlock(const WeightBundle& bundle,
                                 std::string_view prefix, int in_ch, int out_ch,
                                 std::pair<int, int> stride,
                                 std::pair<int, int> pre_kernel,
                                 std::pair<int, int> strided_kernel,
                                 mx::Dtype dtype)
    : stride_(stride),
      pre_kernel_(pre_kernel),
      strided_kernel_(strided_kernel),
      conv1_(bundle, std::string(prefix) + ".conv1", pre_kernel, {1, 1}, true,
             dtype),
      conv2_(bundle, std::string(prefix) + ".conv2", strided_kernel, stride,
             true, dtype) {
  if (in_ch != out_ch) {
    shortcut_.emplace(bundle, std::string(prefix) + ".shortcut",
                      std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1},
                      true, dtype);
  }
}

mx::array EncoderResBlock::operator()(const mx::array& x) const {
  mx::array residual = x;
  mx::array y = elu(x);
  y = conv1_(causal_same_pad(y, pre_kernel_, {1, 1}));
  y = elu(y);
  y = conv2_(causal_same_pad(y, strided_kernel_, stride_));

  const int sh = stride_.first;
  const int sw = stride_.second;
  if (sh != 1 || sw != 1) {
    // Average pool along (H, W) by (sh, sw): mirror numpy reshape+mean.
    const int B = static_cast<int>(residual.shape(0));
    const int H = static_cast<int>(residual.shape(1));
    const int W = static_cast<int>(residual.shape(2));
    const int C = static_cast<int>(residual.shape(3));
    residual = mx::reshape(residual, S(B, H / sh, sh, W / sw, sw, C));
    // Mean over axes 2 and 4.
    residual = mx::mean(residual, /*axis=*/2, /*keepdims=*/false);
    residual = mx::mean(residual, /*axis=*/3, /*keepdims=*/false);
  }
  if (shortcut_.has_value()) residual = (*shortcut_)(residual);
  return mx::add(y, residual);
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

namespace {

struct DecBlockSpec {
  int in_ch;
  int out_ch;
  std::pair<int, int> stride;
  std::pair<int, int> transpose_kernel;
  std::pair<int, int> post_kernel;
};

inline std::vector<DecBlockSpec> dec_block_specs() {
  return {
      {512, 1024, {2, 1}, {4, 3}, {3, 3}},
      {512, 256, {2, 2}, {4, 4}, {3, 3}},
      {256, 256, {1, 2}, {3, 4}, {3, 3}},
      {256, 256, {1, 2}, {3, 4}, {3, 3}},
      {256, 128, {1, 3}, {3, 6}, {3, 3}},
      {128, 128, {1, 2}, {3, 4}, {3, 3}},
      {128, 64, {1, 2}, {3, 4}, {3, 3}},
  };
}

std::vector<DecoderResBlock> build_decoder_blocks(const WeightBundle& bundle,
                                                  mx::Dtype dtype) {
  std::vector<DecoderResBlock> out;
  out.reserve(7);
  const auto specs = dec_block_specs();
  for (size_t i = 0; i < specs.size(); ++i) {
    const auto& s = specs[i];
    out.emplace_back(bundle, "decoder_blocks." + std::to_string(i), s.in_ch,
                     s.out_ch, s.stride, s.transpose_kernel, s.post_kernel,
                     dtype);
  }
  return out;
}

}  // namespace

SpectroStreamDecoder::SpectroStreamDecoder(const WeightBundle& bundle,
                                           mx::Dtype dtype)
    : dtype_(dtype),
      input_proj_(bundle, "input_proj", 256, 2560, 2560, /*pre_elu=*/false,
                  dtype),
      input_conv_(bundle, "input_conv", 512, /*causal=*/true, dtype),
      decoder_blocks_(build_decoder_blocks(bundle, dtype)),
      output_conv_(bundle, "output_conv", {7, 7}, {1, 1}, true, dtype) {}

mx::array SpectroStreamDecoder::operator()(const mx::array& embeddings) const {
  if (embeddings.ndim() != 3 || embeddings.shape(2) != kEmbeddingDim) {
    throw std::invalid_argument(
        "SpectroStreamDecoder: embeddings must be (B, S, 256)");
  }
  const int B = static_cast<int>(embeddings.shape(0));
  const int D = static_cast<int>(embeddings.shape(2));

  mx::array x_emb = mx::astype(embeddings, dtype_);
  mx::array zero_frame = mx::zeros(S(B, kTemporalPad, D), dtype_);
  mx::array x = mx::concatenate({x_emb, zero_frame}, /*axis=*/1);

  x = mx::expand_dims(x, /*axis=*/2);

  x = input_proj_(x);          // (B, S+1, 1, 2560)
  x = unfold_width(x, 5);      // (B, S+1, 5, 512)
  x = input_conv_(x);

  x = decoder_blocks_[0](x);

  // Stereo split: (B, H, W, 1024) -> (2B, H, W, 512).
  {
    const int H = static_cast<int>(x.shape(1));
    const int W = static_cast<int>(x.shape(2));
    const int C = static_cast<int>(x.shape(3));
    x = mx::reshape(x, S(B, H, W, 2, C / 2));
    x = mx::transpose(x, {3, 0, 1, 2, 4});
    x = mx::reshape(x, S(2 * B, H, W, C / 2));
  }

  for (size_t i = 1; i < decoder_blocks_.size(); ++i) {
    x = decoder_blocks_[i](x);
  }

  x = elu(x);
  x = output_conv_(causal_same_pad(x, {7, 7}, {1, 1}));

  // Temporal crop: drop the first kTemporalCrop frames along H.
  {
    const int H = static_cast<int>(x.shape(1));
    x = mx::slice(x, S(0, kTemporalCrop, 0, 0),
                  S(x.shape(0), H, x.shape(2), x.shape(3)));
  }

  // Stereo recombine: (2B, H, 480, 2) -> (B, H, 480, 4).
  {
    const int H_out = static_cast<int>(x.shape(1));
    const int W_out = static_cast<int>(x.shape(2));
    const int C_out = static_cast<int>(x.shape(3));
    x = mx::reshape(x, S(2, B, H_out, W_out, C_out));
    x = mx::transpose(x, {1, 2, 3, 0, 4});
    x = mx::reshape(x, S(B, H_out, W_out, 2 * C_out));
  }

  return x;
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

namespace {

struct EncBlockSpec {
  int in_ch;
  int out_ch;
  std::pair<int, int> stride;
  std::pair<int, int> pre_kernel;
  std::pair<int, int> strided_kernel;
};

inline std::vector<EncBlockSpec> enc_block_specs() {
  return {
      {32, 64, {1, 2}, {3, 3}, {3, 4}},
      {64, 64, {1, 2}, {3, 3}, {3, 4}},
      {64, 128, {1, 3}, {3, 3}, {3, 6}},
      {128, 128, {1, 2}, {3, 3}, {3, 4}},
      {128, 128, {1, 2}, {3, 3}, {3, 4}},
      {128, 256, {2, 2}, {3, 3}, {4, 4}},
  };
}

std::vector<EncoderResBlock> build_encoder_blocks(const WeightBundle& bundle,
                                                  mx::Dtype dtype) {
  std::vector<EncoderResBlock> out;
  out.reserve(6);
  const auto specs = enc_block_specs();
  for (size_t i = 0; i < specs.size(); ++i) {
    const auto& s = specs[i];
    out.emplace_back(bundle, "encoder_blocks." + std::to_string(i), s.in_ch,
                     s.out_ch, s.stride, s.pre_kernel, s.strided_kernel, dtype);
  }
  return out;
}

}  // namespace

SpectroStreamEncoder::SpectroStreamEncoder(const WeightBundle& bundle,
                                           mx::Dtype dtype)
    : dtype_(dtype),
      base_conv_(bundle, "base_conv", {7, 7}, {1, 1}, true, dtype),
      encoder_blocks_(build_encoder_blocks(bundle, dtype)),
      encoder_block_6_(bundle, "encoder_block_6", /*in_ch=*/512,
                       /*out_ch=*/256, /*stride=*/{2, 1},
                       /*pre_kernel=*/{3, 3}, /*strided_kernel=*/{4, 3}, dtype),
      bottleneck_conv_(bundle, "bottleneck_conv", 256, /*causal=*/true, dtype),
      bottleneck_proj_(bundle, "bottleneck_proj", 1280, 1280, 256,
                       /*pre_elu=*/true, dtype) {}

mx::array SpectroStreamEncoder::operator()(const mx::array& stft_features) const {
  const int B_stereo = static_cast<int>(stft_features.shape(0));
  const int B = B_stereo / 2;

  mx::array x = mx::astype(stft_features, dtype_);
  x = base_conv_(causal_same_pad(x, {7, 7}, {1, 1}));

  for (const auto& blk : encoder_blocks_) {
    x = blk(x);
  }

  // Stereo recombination: (B*2, H, W, 256) -> (B, H, W, 512).
  {
    const int H_enc = static_cast<int>(x.shape(1));
    const int W_enc = static_cast<int>(x.shape(2));
    const int C_enc = static_cast<int>(x.shape(3));
    x = mx::reshape(x, S(2, B, H_enc, W_enc, C_enc));
    x = mx::transpose(x, {1, 2, 3, 0, 4});
    x = mx::reshape(x, S(B, H_enc, W_enc, 2 * C_enc));
  }

  x = encoder_block_6_(x);
  x = bottleneck_conv_(x);

  // Fold W by 5: (B, H, 5, 256) -> (B, H, 1, 1280).
  x = fold_width(x, 5);
  x = bottleneck_proj_(x);

  // (B, S, 1, 256) -> (B, S, 256).
  x = mx::squeeze(x, /*axis=*/2);
  return x;
}

// ---------------------------------------------------------------------------
// STFT pre-processing
// ---------------------------------------------------------------------------

mx::array stft_preprocess(const mx::array& samples) {
  constexpr int N_FFT = 960;
  constexpr int HOP = 480;
  constexpr int FREQ_BINS = 480;

  if (samples.ndim() != 3) {
    throw std::invalid_argument("stft_preprocess: expected (B, T, 2) input");
  }
  const int B = static_cast<int>(samples.shape(0));
  int T = static_cast<int>(samples.shape(1));
  const int C_audio = static_cast<int>(samples.shape(2));
  if (C_audio != 2) {
    throw std::invalid_argument("stft_preprocess: expected stereo input (C=2)");
  }

  mx::array x = samples;
  const int raw_frames = (T - N_FFT) / HOP + 1;
  const int target_frames = ((raw_frames + 3) / 4) * 4;
  if (target_frames > raw_frames) {
    const int target_T = (target_frames - 1) * HOP + N_FFT;
    std::vector<std::pair<int, int>> pads = {
        {0, 0}, {0, target_T - T}, {0, 0}};
    x = mx::pad(x, pads);
    T = target_T;
  }

  // (B, T, 2) -> (B, 2, T) -> (B*2, T).
  x = mx::transpose(x, {0, 2, 1});
  x = mx::reshape(x, S(B * C_audio, T));
  x = mx::astype(x, mx::float32);

  const int num_frames = (T - N_FFT) / HOP + 1;

  // Periodic Hann window of length N_FFT.
  mx::array t_idx = mx::astype(mx::arange(N_FFT), mx::float32);
  const float two_pi_over_n =
      2.0f * static_cast<float>(M_PI) / static_cast<float>(N_FFT);
  mx::array window = mx::subtract(
      mx::array(0.5f),
      mx::multiply(mx::array(0.5f),
                   mx::cos(mx::multiply(t_idx, mx::array(two_pi_over_n)))));

  // Build frames via strided slicing. Gather (B*2, num_frames, N_FFT).
  // Use an arange + broadcast trick: indices[i, j] = j + i*HOP.
  mx::array frame_idx = mx::astype(mx::arange(num_frames), mx::int32);
  mx::array sample_idx = mx::astype(mx::arange(N_FFT), mx::int32);
  mx::array indices = mx::add(
      mx::multiply(mx::expand_dims(frame_idx, /*axis=*/1), mx::array(HOP)),
      mx::expand_dims(sample_idx, /*axis=*/0));  // (num_frames, N_FFT)

  // gather along axis=-1 of x: take_along_axis expects same rank as x.
  mx::array idx_flat = mx::reshape(indices, S(num_frames * N_FFT));
  mx::array idx_b =
      mx::broadcast_to(mx::reshape(idx_flat, S(1, num_frames * N_FFT)),
                       S(B * C_audio, num_frames * N_FFT));
  mx::array gathered = mx::take_along_axis(x, idx_b, /*axis=*/1);
  mx::array frames = mx::reshape(gathered, S(B * C_audio, num_frames, N_FFT));

  mx::array windowed =
      mx::multiply(frames, mx::reshape(window, S(1, 1, N_FFT)));

  // RFFT along last axis -> (B*2, num_frames, 481).
  mx::array spectrum = mx::fft::rfft(windowed, /*n=*/N_FFT, /*axis=*/-1);

  // Drop Nyquist bin (keep 0..FREQ_BINS-1).
  spectrum = mx::slice(spectrum, S(0, 0, 0),
                       S(spectrum.shape(0), num_frames, FREQ_BINS));

  // (B*2, H, 480) complex -> (B*2, H, 480, 2) real/imag.
  mx::array real_part =
      mx::expand_dims(mx::real(spectrum), /*axis=*/-1);
  mx::array imag_part =
      mx::expand_dims(mx::imag(spectrum), /*axis=*/-1);
  mx::array stacked =
      mx::concatenate({real_part, imag_part}, /*axis=*/-1);
  return mx::astype(stacked, mx::float32);
}

// ---------------------------------------------------------------------------
// iSTFT post-processing
// ---------------------------------------------------------------------------

mx::array istft_postprocess(const mx::array& spec, int B, int target_length) {
  constexpr int N_FFT = 960;
  constexpr int HOP = 480;

  const int frames = static_cast<int>(spec.shape(1));
  const int freq = static_cast<int>(spec.shape(2));  // 480

  // 1. Restore Nyquist bin via right-pad of size 1 along axis 2.
  std::vector<std::pair<int, int>> pads = {
      {0, 0}, {0, 0}, {0, 1}, {0, 0}};
  mx::array x = mx::pad(spec, pads);  // (B, H, 481, 4)

  // 2. (B, H, 481, 4) -> (B, H, 481, stereo=2, ri=2)
  x = mx::reshape(x, S(B, frames, freq + 1, 2, 2));

  // 3. Build complex64 from real/imag halves: shape (B, H, 481, 2)
  mx::array x_real = mx::squeeze(
      mx::slice(x, S(0, 0, 0, 0, 0), S(B, frames, freq + 1, 2, 1)), /*axis=*/-1);
  mx::array x_imag = mx::squeeze(
      mx::slice(x, S(0, 0, 0, 0, 1), S(B, frames, freq + 1, 2, 2)), /*axis=*/-1);
  mx::array imag_unit = mx::array(std::complex<float>(0.0f, 1.0f));
  mx::array x_complex = mx::add(
      mx::astype(x_real, mx::complex64),
      mx::multiply(mx::astype(x_imag, mx::complex64), imag_unit));

  // 4. (B, H, 481, 2) -> (B, 2, H, 481) -> (B*2, H, 481)
  x_complex = mx::transpose(x_complex, {0, 3, 1, 2});
  x_complex = mx::reshape(x_complex, S(B * 2, frames, freq + 1));

  // 5. iRFFT along last axis.
  mx::array time_frames =
      mx::fft::irfft(x_complex, /*n=*/N_FFT, /*axis=*/-1);
  time_frames = mx::astype(time_frames, mx::float32);

  // 6. Synthesis window: hann_window / hann_squared_periodic_sum.
  mx::array t = mx::astype(mx::arange(N_FFT), mx::float32);
  const float two_pi_over_n =
      2.0f * static_cast<float>(M_PI) / static_cast<float>(N_FFT);
  mx::array two_pi_t_over_n = mx::multiply(t, mx::array(two_pi_over_n));
  mx::array window = mx::subtract(
      mx::array(0.5f),
      mx::multiply(mx::array(0.5f), mx::cos(two_pi_t_over_n)));
  mx::array hann_sq = mx::multiply(window, window);
  mx::array periodic = mx::sum(mx::reshape(hann_sq, S(2, HOP)), /*axis=*/0);
  mx::array periodic_full = mx::concatenate({periodic, periodic}, /*axis=*/0);
  mx::array synth_window = mx::divide(window, periodic_full);

  // 7. Apply synthesis window: shape (B*2, H, N_FFT).
  mx::array windowed =
      mx::multiply(time_frames, mx::reshape(synth_window, S(1, 1, N_FFT)));

  // 8. Overlap-add (HOP = N_FFT / 2). For each hop ``i``:
  //     audio[i*HOP + j]    = windowed[i, j]                     for j in [0, HOP)
  //                         + windowed[i-1, j + HOP]             for i > 0
  //     audio[frames*HOP + j] = windowed[frames-1, j + HOP]      for j in [0, HOP)
  mx::array left =
      mx::slice(windowed, S(0, 0, 0), S(windowed.shape(0), frames, HOP));
  mx::array right = mx::slice(
      windowed, S(0, 0, HOP), S(windowed.shape(0), frames, N_FFT));
  mx::array right_shifted = mx::pad(
      right, std::vector<std::pair<int, int>>{{0, 0}, {1, 0}, {0, 0}});
  right_shifted = mx::slice(
      right_shifted, S(0, 0, 0), S(right_shifted.shape(0), frames, HOP));
  mx::array body = mx::add(left, right_shifted);
  mx::array body_flat =
      mx::reshape(body, S(static_cast<int>(windowed.shape(0)), frames * HOP));
  mx::array tail = mx::slice(
      right, S(0, frames - 1, 0), S(windowed.shape(0), frames, HOP));
  tail = mx::reshape(tail, S(static_cast<int>(windowed.shape(0)), HOP));
  mx::array audio = mx::concatenate({body_flat, tail}, /*axis=*/-1);

  // 9. Trim to target_length and reshape stereo: (B*2, T) -> (B, 2, T) -> (B, T, 2)
  audio = mx::slice(audio, S(0, 0), S(audio.shape(0), target_length));
  audio = mx::reshape(audio, S(B, 2, target_length));
  audio = mx::transpose(audio, {0, 2, 1});
  // ``transpose`` is a logical view: the underlying buffer is still in
  // ``(B, 2, T)`` order, but the array reports shape ``(B, T, 2)``. Callers
  // that read ``data<float>()`` linearly (host memcpy, WAV writer, PortAudio
  // queue) treat the buffer as interleaved L/R/L/R, but a non-contiguous
  // transpose hands them planar L...L R...R. That misinterpretation plays
  // the left channel at 2x speed for the first half of the chunk and the
  // right channel at 2x speed for the second half, with a discontinuity
  // blip mid-chunk (and another at every chunk boundary). Force a
  // contiguous copy so ``data<float>()`` is in interleaved memory order.
  audio = mx::contiguous(audio);
  return audio;
}

}  // namespace magenta_realtime_mlx
