#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

#include "magenta_realtime_mlx/dtype.h"

namespace mrt = magenta_realtime_mlx;
namespace mx = mlx::core;

TEST_CASE("parse_dtype accepts fp32 spellings", "[dtype]") {
  REQUIRE(mrt::parse_dtype("fp32") == mx::float32);
  REQUIRE(mrt::parse_dtype("float32") == mx::float32);
  REQUIRE(mrt::parse_dtype("F32") == mx::float32);
}

TEST_CASE("parse_dtype accepts fp16 spellings", "[dtype]") {
  REQUIRE(mrt::parse_dtype("fp16") == mx::float16);
  REQUIRE(mrt::parse_dtype("float16") == mx::float16);
  REQUIRE(mrt::parse_dtype("Half") == mx::float16);
}

TEST_CASE("parse_dtype accepts bf16 spellings", "[dtype]") {
  REQUIRE(mrt::parse_dtype("bf16") == mx::bfloat16);
  REQUIRE(mrt::parse_dtype("bfloat16") == mx::bfloat16);
}

TEST_CASE("parse_dtype rejects nonsense", "[dtype]") {
  REQUIRE_THROWS_AS(mrt::parse_dtype("int8"), std::invalid_argument);
  REQUIRE_THROWS_AS(mrt::parse_dtype(""), std::invalid_argument);
}

TEST_CASE("dtype_name round-trips", "[dtype]") {
  for (const auto* name : {"fp32", "fp16", "bf16"}) {
    REQUIRE(mrt::dtype_name(mrt::parse_dtype(name)) == name);
  }
}
