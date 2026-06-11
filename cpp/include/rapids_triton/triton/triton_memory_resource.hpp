/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <triton/core/tritonbackend.h>
#include <triton/core/tritonserver.h>
#include <cstddef>
#include <cstdint>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/device.hpp>
#include <rmm/aligned.hpp>
#include <cuda/memory_resource>
#include <stdexcept>
#include <utility>

namespace triton {
namespace backend {
namespace rapids {
struct triton_memory_resource {
  triton_memory_resource(TRITONBACKEND_MemoryManager* manager,
                         device_id_t device_id,
                         cuda::mr::any_resource<cuda::mr::device_accessible> fallback)
    : manager_{manager}, device_id_{device_id}, fallback_{fallback}
  {
  }

  auto* get_triton_manager() const noexcept { return manager_; }

 private:
  TRITONBACKEND_MemoryManager* manager_;
  std::int64_t device_id_;
  cuda::mr::any_resource<cuda::mr::device_accessible> fallback_;

 public:
  void* allocate(cuda::stream_ref stream, std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto* ptr = static_cast<void*>(nullptr);
    if (manager_ == nullptr) {
      ptr = fallback_.allocate(stream, bytes, alignment);
    } else {
      triton_check(TRITONBACKEND_MemoryManagerAllocate(
        manager_, &ptr, TRITONSERVER_MEMORY_GPU, device_id_, static_cast<std::uint64_t>(bytes)));
    }
    return ptr;
  }

  void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    if (manager_ == nullptr) {
      fallback_.deallocate(stream, ptr, bytes, alignment);
    } else {
      triton_check(
        TRITONBACKEND_MemoryManagerFree(manager_, ptr, TRITONSERVER_MEMORY_GPU, device_id_));
    }
  }

  void* allocate_sync(std::size_t bytes,
                      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr, std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(const triton_memory_resource& other) const {
    return this == &other;
  }

  bool operator!=(const triton_memory_resource& other) const {
    return !(*this == other);
  }

  constexpr friend void get_property(const triton_memory_resource&, cuda::mr::device_accessible) noexcept {}
};

static_assert(cuda::mr::resource_with<triton_memory_resource, cuda::mr::device_accessible>,
              "triton_memory_resource is not a valid cuda::mr::resource");

}  // namespace rapids
}  // namespace backend
}  // namespace triton
