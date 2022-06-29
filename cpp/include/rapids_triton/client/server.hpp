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
#include <rapids_triton/triton/logging.hpp>

#include <triton/core/tritonserver.h>

#include <string>
#include <thread>

namespace triton {
namespace client {

using namespace triton::backend::rapids;

class TritonServer {

public:
    TritonServer(const std::string& model_repo_path, const int verbose_level = 0) {

        // supply options to server
        set_server_options(model_repo_path, verbose_level);

        // check if server ready
        is_server_ready();
    }

    ~TritonServer() {
        triton_check(TRITONSERVER_ServerDelete(server_ptr));
    }

    auto get() const {
        return server_ptr;
    }

private:

    void set_server_options(const std::string& model_repo_path, double min_compute_capability = 6.0, const int verbose_level = 0) {
        TRITONSERVER_ServerOptions* server_options = nullptr;
        triton_check(
            TRITONSERVER_ServerOptionsNew(&server_options));
        triton_check(
            TRITONSERVER_ServerOptionsSetModelRepositoryPath(
                server_options, model_repo_path.c_str()));
        triton_check(
            TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level));
        triton_check(
            TRITONSERVER_ServerOptionsSetBackendDirectory(
                server_options, "/opt/tritonserver/backends"));
        triton_check(
            TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
                server_options, "/opt/tritonserver/repoagents"));
        triton_check(
            TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true));

        #ifndef TRITON_ENABLE_GPU
          min_compute_capability = 0;
        #endif  // TRITON_ENABLE_GPU
        triton_check(
            TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
                server_options, min_compute_capability));

        triton_check(
            TRITONSERVER_ServerNew(&server_ptr, server_options));
        triton_check(
            TRITONSERVER_ServerOptionsDelete(server_options));
    }

    void is_server_ready() {
        size_t health_iters = 0;
        while (true) {
            bool live, ready;
            triton_check(
                TRITONSERVER_ServerIsLive(server_ptr, &live));
            triton_check(
                TRITONSERVER_ServerIsReady(server_ptr, &ready));
            if (live && ready) {
                break;
            }

            if (++health_iters >= 10) {
                log_error("failed to find healthy inference server");
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    TRITONSERVER_Server *server_ptr {nullptr};

};

} // namespace client
} // namespace triton