/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/custom.h"

namespace tensorflow {

void FloatToCustom(const float* src, custom* dst, int64 size) {
  const uint32_t* p = reinterpret_cast<const uint32_t*>(src);
  uint32_t* q = reinterpret_cast<uint32_t*>(dst);
  for (; size != 0; p++, q++, size--) {
    *q = p[0];
  }
}

void CustomToFloat(const custom* src, float* dst, int64 size) {
  const uint32_t* p = reinterpret_cast<const uint32_t*>(src);
  uint32_t* q = reinterpret_cast<uint32_t*>(dst);
  for (; size != 0; p++, q++, size--) {
    q[0] = *p;
  }
}

}  // end namespace tensorflow
