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

#ifndef TFPLUS_KV_VARIABLE_KERNELS_KV_VARIABLE_INTERFACE_H_
#define TFPLUS_KV_VARIABLE_KERNELS_KV_VARIABLE_INTERFACE_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tfplus/kv_variable/kernels/mutex.h"

namespace tfplus {
using ::tensorflow::DataType;
using ::tensorflow::mutex;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::string;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;

enum RestoreMode {
  NORMAL = 0,
  MERGE = 1,        // merge value from variable with same name
  REPARTITION = 2,  // repartition restore
  REPARTITION_MERGE = 3
};

enum ScatterUpdateOps {
  SCATTER_UPDATE_ASSIGN,
  SCATTER_UPDATE_ADD,
  SCATTER_UPDATE_SUB,
  SCATTER_UPDATE_MUL,
  SCATTER_UPDATE_DIV,
  SCATTER_UPDATE_MIN,
  SCATTER_UPDATE_MAX
};

constexpr bool DEFAULT_ENABLE_CUTOFF = true;
constexpr float DEFAULT_CUTOFF_VALUE = 1.0e-20;

class KvVariableInterface : public ::tensorflow::ResourceBase {
 public:
  virtual ~KvVariableInterface() = default;
  virtual size_t size() const = 0;
  virtual size_t sum_freq() const = 0;
  virtual TensorShape GetShape() const = 0;
  virtual bool IsInitialized() const = 0;

  /*
  Use the input random_init tensor to set the initialization table.
  */
  virtual Status InitRandomValues(const Tensor& random_init) = 0;

  /*
  Get the shape of intialilzation table. If the initialization table
  has not been set yet, return an error.
  */
  virtual Status GetInitTableShape(TensorShape* shape) const = 0;

  /*
  Copy the content of the initialization table. The input tensor object
  must have the same shape with the initialization table. We can invoke
  GetInitTableShape(...) first to get the shape of the initialization table
  and then allocate a tensor as the input to this function.
  */
  virtual Status GetInitTable(Tensor* tensor) const = 0;

  // FindOrZeros retrieves the tensor with given keys. All zeros will be
  // returned if the given key doesn't exist in the table_. The function is
  // typically used in KvVariableGather OP for model prediction.
  virtual Status FindOrZeros(OpKernelContext* ctx, const Tensor& keys,
                             Tensor* values) const = 0;

  Status FindOrInsert(OpKernelContext* ctx, const Tensor& keys,
                      Tensor* values) {
    return FindOrInsert(ctx, keys, values, nullptr, nullptr);
  }

  Status FindOrInsert(OpKernelContext* ctx, const Tensor& keys, Tensor* values,
                      const Tensor* filter_out) {
    return FindOrInsert(ctx, keys, values, nullptr,
                        const_cast<Tensor*>(filter_out), true);
  }

  Status FindOrInsertFillFilter(OpKernelContext* ctx, const Tensor& keys,
                                Tensor* values, Tensor* filter_out) {
    return FindOrInsert(ctx, keys, values, nullptr, filter_out, false);
  }

  Status FindOrInsertWithCounts(OpKernelContext* ctx, const Tensor& keys,
                                const Tensor& counts, Tensor* values) {
    return FindOrInsert(ctx, keys, values, &counts, nullptr);
  }

  //  FindOrInsert is typically used in KvVariableGather OP for model training.
  //  If the key already in the table_, we return its value, otherwise we will
  //  insert a random vector from random_init_table_ as its initializer vector.
  //  The 3rd param filter_out is used to filter out the low frequency. If the
  //  key is low frequency, we do not insert it, and do not update its gradient
  //  in backward computing.
  virtual Status FindOrInsert(OpKernelContext* ctx, const Tensor& keys,
                              Tensor* values, const Tensor* counts,
                              Tensor* filter_out, bool apply_filter = true) = 0;

  /*
    InsertOrUpdate is used after optimizer complete the backward computing.
    The filter_out are low frequency keys which will not be updated.
    The blacklist are used in group lasso && sparse group lasso optimizer that
    indicate the key has already group sparse and can be removed.
  */
  Status InsertOrUpdate(OpKernelContext* ctx, const Tensor& keys,
                        const Tensor& values) {
    return InsertOrUpdate(ctx, keys, values, nullptr, nullptr);
  }

  Status InsertOrUpdate(OpKernelContext* ctx, const Tensor& keys,
                        const Tensor& values, const Tensor* filter_out) {
    return InsertOrUpdate(ctx, keys, values, filter_out, nullptr);
  }

  virtual Status InsertOrUpdate(OpKernelContext* ctx, const Tensor& keys,
                                const Tensor& values, const Tensor* filter_out,
                                const Tensor* blacklist) = 0;

  virtual Status GetCount(const Tensor& indices, Tensor* counts) = 0;

  virtual Status GetTimeStamp(const Tensor& indices, Tensor* timestamps) = 0;

  /*
  As per the operation given, update the values of the keys.
  Supported operations include ADD, SUB, MUL, DIV, MIN and MAX.
  */
  virtual Status ScatterUpdate(OpKernelContext* ctx, const Tensor& keys,
                               const Tensor& updates,
                               ScatterUpdateOps update_op) = 0;

  /*
    Load values from checkpoint or savedmodel.
  */
  virtual Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                              const Tensor& values,
                              const std::vector<Tensor>& others) = 0;

  /*
    Export values to checkpoint or savedmodel.
    If enable_cutoff is true, we will not export key which all values are
    smaller than cutoff_value.
  */
  virtual Status ExportValues(OpKernelContext* ctx, int first_n,
                              bool enable_cutoff = false,
                              float cutoff_value = 0.0,
                              void* table_handler = nullptr) = 0;

  virtual Status ExportValuesForMultiHash(OpKernelContext* ctx, int first_n,
                                          bool enable_cutoff,
                                          float cutoff_value,
                                          const string& variable_name) = 0;

  virtual Status CountStorageSize(OpKernelContext* ctx) = 0;

  virtual Status FullOrDeltaImport(OpKernelContext* ctx, int first_n,
                                   const Tensor& keys, const Tensor& values,
                                   const std::vector<Tensor>& others) = 0;

  virtual Status FullOrDeltaExport(OpKernelContext* ctx, int first_n,
                                   bool enable_cutoff = false,
                                   float cutoff_value = 0.0,
                                   bool need_full_export = true) = 0;

  virtual bool NeedDeltaInfo() const = 0;

  virtual Status ApplySnapshot(const std::vector<std::string>& src_files) = 0;

  virtual Status LoadRemoteTable(const std::string& table_name) = 0;

  virtual Status Delete(const Tensor& indices) = 0;

  virtual Status DeleteWithTimestamp(OpKernelContext* ctx, int threshold) = 0;

  virtual Status MarkAsDeltaListElements(OpKernelContext *ctx,
                                         const Tensor &keys,
                                         const std::vector<int64_t> &indices)
                                         = 0;

  virtual const bool is_support_delta_export() const = 0;

  virtual const std::string& name() const = 0;

  virtual DataType key_dtype() const = 0;

  virtual DataType value_dtype() const = 0;

  virtual TensorShape key_shape() const { return TensorShape(); }

  virtual TensorShape value_shape() const = 0;

  virtual int embedding_dim() const = 0;

  virtual tf_mutex* mu() const = 0;
};  // KvVariableInterface

}  // namespace tfplus
#endif  // TFPLUS_KV_VARIABLE_KERNELS_KV_VARIABLE_INTERFACE_H_
