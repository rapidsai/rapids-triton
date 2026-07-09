/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <rapids_triton/exceptions.hpp>
#include <type_traits>

namespace triton {
namespace backend {
namespace rapids {

template <typename T>
T safe_multiply(T a, T b) {
  static_assert(std::is_unsigned_v<T> && std::is_integral_v<T>,
                "safe_multiply only defined for unsigned integers");
  T result = a * b;
  if (a != 0 && result / a != b) {
    throw TritonException(Error::Internal, "Overflow detected in multiply");
  }
  return result;
}

}  // namespace rapids
}  // namespace backend
}  // namespace triton
