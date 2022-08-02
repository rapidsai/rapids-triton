/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <rapids_triton/tensor/dtype.hpp>
#include <rapids_triton/triton/logging.hpp>

#include <vector>

namespace triton {
namespace client {

using namespace triton::backend::rapids;

class TritonRequest {
public:
    TritonRequest(std::vector<std::pair<std::string, Dtype>> input_info, std::vector<std::pair<std::string, Dtype>> output_info) :
        input_info_{input_info},
        output_info_{output_info} {}

    ~TritonRequest();

    // call this for every input
    template <typename T>
    void set_input();

    // call this for as many outputs are needed
    template <typename T>
    T* get_output();

private:
    std::vector<std::pair<std::string, Dtype>> input_info_;
    std::vector<std::pair<std::string, Dtype>> output_info_;

};

} // namespace client
} // namespace triton