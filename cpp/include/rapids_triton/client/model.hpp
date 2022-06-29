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

#include "server.hpp"

#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/logging.hpp>
#include <rapids_triton/tensor/dtype.hpp>

// Disable class-memaccess warning to facilitate compilation with gcc>7
// https://github.com/Tencent/rapidjson/issues/1700
#pragma GCC diagnostic push
#if defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
#include <rapidjson/document.h>
#pragma GCC diagnostic pop
#include <rapidjson/error/en.h>
#include <triton/core/tritonserver.h>

#include <string>
#include <thread>
#include <vector>

namespace triton {
namespace client {

using namespace triton::backend::rapids;

class TritonModel {

public:
    TritonModel(const TritonServer& server, const std::string& model_name, const int model_version) {
        // check if model is loaded by server
        is_model_ready(server, model_name, model_version);

        // parse model metadata with rapidjson
        parse_model_metadata(server, model_name, model_version);

        // get names of i/o and their corresponding data types
        parse_io_names_and_dtypes();
    }

    // This will return an object in a thread-safe manner
    // to infer requests from the server async
    auto create_inference_request();

private:
    void is_model_ready(const TritonServer& server, const std::string& model_name, const int model_version) {
        bool is_ready = false;
        int health_iters = 0;
        while (!is_ready) {
            triton_check(
                TRITONSERVER_ServerModelIsReady(
                    server.get(), model_name.c_str(), model_version, &is_ready));
            if (!is_ready) {
                if (++health_iters >= 10) {
                    log_error("model failed to be ready in 10 iterations");
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
    }

    void parse_model_metadata(const TritonServer& server, const std::string& model_name, const int model_version) {
        TRITONSERVER_Message* model_metadata_message;
        triton_check(
            TRITONSERVER_ServerModelMetadata(
                server.get(), model_name.c_str(), 1, &model_metadata_message));
        const char* buffer;
        size_t byte_size;
        triton_check(
            TRITONSERVER_MessageSerializeToJson(
                model_metadata_message, &buffer, &byte_size));

        std::cout << std::string(buffer) << std::endl;

        model_metadata.Parse(buffer, byte_size);
        if (model_metadata.HasParseError()) {
        log_error(
            "error: failed to parse model metadata from JSON: " +
            std::string(GetParseError_En(model_metadata.GetParseError())) +
            " at " + std::to_string(model_metadata.GetErrorOffset()));
        }

        triton_check(
            TRITONSERVER_MessageDelete(model_metadata_message));
    }

    void parse_io_names_and_dtypes() {
        for(const auto& input : model_metadata["inputs"].GetArray()) {
            auto info = std::make_pair(input["name"].GetString(), TRITONSERVER_StringToDataType(input["datatype"].GetString()));
            inputs_info.push_back(info);
        }

        for(const auto& output : model_metadata["outputs"].GetArray()) {
            auto info = std::make_pair(output["name"].GetString(), TRITONSERVER_StringToDataType(output["datatype"].GetString()));
            outputs_info.push_back(info);
        }
    }

    rapidjson::Document model_metadata;
    std::vector<std::pair<std::string, DType>> inputs_info;
    std::vector<std::pair<std::string, DType>> outputs_info;

};

} // namespace client
} // namespace triton