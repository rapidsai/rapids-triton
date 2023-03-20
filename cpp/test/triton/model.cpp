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

#include <gtest/gtest.h>
#include <string>
#include <rapids_triton/model/shared_state.hpp>

namespace triton {
namespace backend {
namespace rapids {

bool TestBoolString(std::string b){
  auto model_config = std::make_unique<common::TritonJson::Value>(common::TritonJson::ValueType::OBJECT);
  model_config->AddInt("max_batch_size", 1);
  model_config->Add("output", common::TritonJson::Value(common::TritonJson::ValueType::ARRAY));
  auto params = common::TritonJson::Value(common::TritonJson::ValueType::OBJECT);
  auto string_value = common::TritonJson::Value(common::TritonJson::ValueType::OBJECT);
  string_value.AddString("string_value", b);
  params.Add("some_bool", std::move(string_value));
  model_config->Add("parameters",std::move(params));
  SharedModelState s(std::move(model_config));
  return s.get_config_param<bool>("some_bool");
}

TEST(RapidsTriton, bool_param)
{
  EXPECT_TRUE(TestBoolString("true"));
  EXPECT_FALSE(TestBoolString("false"));
  EXPECT_THROW(
    {
      try {
        TestBoolString("True");
      } catch (const TritonException& e) {
        EXPECT_STREQ("Expected 'true' or 'false' for parameter 'some_bool', got: 'True'", e.what());
        throw;
      }
    },
    TritonException);
}
}  // namespace rapids
}  // namespace backend
}  // namespace triton