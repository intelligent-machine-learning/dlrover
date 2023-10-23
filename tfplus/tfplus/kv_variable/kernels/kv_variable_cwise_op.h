// Copyright 2023 The TFPlus Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TFPLUS_KV_VARIABLE_KERNELS_KV_VARIABLE_CWISE_OP_H_
#define TFPLUS_KV_VARIABLE_KERNELS_KV_VARIABLE_CWISE_OP_H_

namespace tfplus {

template <typename T>
struct CwiseOperationBase {
  virtual ~CwiseOperationBase() = default;
  virtual T operator()(const T& lhs, const T& rhs) const = 0;
};

template <typename T>
struct CwiseOperationAssign : public CwiseOperationBase<T> {
  T operator()(const T& lhs, const T& rhs) const override { return rhs; }
};

template <typename T>
struct CwiseOperationAdd : public CwiseOperationBase<T> {
  T operator()(const T& lhs, const T& rhs) const override { return lhs + rhs; }
};

template <typename T>
struct CwiseOperationSub : public CwiseOperationBase<T> {
  T operator()(const T& lhs, const T& rhs) const override { return lhs - rhs; }
};

template <typename T>
struct CwiseOperationMul : public CwiseOperationBase<T> {
  T operator()(const T& lhs, const T& rhs) const override { return lhs * rhs; }
};

template <typename T>
struct CwiseOperationDiv : public CwiseOperationBase<T> {
  T operator()(const T& lhs, const T& rhs) const override { return lhs / rhs; }
};

template <typename T>
struct CwiseOperationMin : public CwiseOperationBase<T> {
  T operator()(const T& lhs, const T& rhs) const override {
    return lhs.cwiseMin(rhs);
  }
};

template <typename T>
struct CwiseOperationMax : public CwiseOperationBase<T> {
 public:
  T operator()(const T& lhs, const T& rhs) const override {
    return lhs.cwiseMax(rhs);
  }
};

}  // namespace tfplus

#endif  // TFPLUS_KV_VARIABLE_KERNELS_KV_VARIABLE_CWISE_OP_H_
