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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/types.h"
#include "tfplus/kv_variable/kernels/kv_variable.h"
#include "tfplus/kv_variable/kernels/utility.h"
#include "tfplus/kv_variable/kernels/hybrid_embedding/storage_config.pb.h"

namespace tfplus {
using namespace tensorflow;  // NOLINT(build/namespaces)

template <typename K, typename V>
class CreateKvVariableOp : public OpKernel {
 public:
  explicit CreateKvVariableOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), use_node_name_sharing_(false) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
    init_token_.store(false, std::memory_order_relaxed);
    table_handle_set_.store(false, std::memory_order_relaxed);
  }

  // ~CreateKvVariableOp() override {
  //   if (table_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
  //     KvVariableInterface* table = nullptr;
  //     cinfo_.resource_manager()->template Lookup<KvVariableInterface>(
  //         cinfo_.container(), cinfo_.name(), &table);
  //     core::ScopedUnref unref_me(table);

  //     if (!cinfo_.resource_manager()
  //              ->template Delete<KvVariableInterface>(cinfo_.container(),
  //                                                     cinfo_.name())
  //              .ok()) {
  //       // Do nothing; the resource can have been deleted by session resets.
  //     }
  //   }
  // }

  void Compute(OpKernelContext* ctx) override {
    bool init_token = init_token_.exchange(true, std::memory_order_relaxed);
    if (!init_token) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
      int32 enter_threshold;
      std::string variable_name;
      TensorShape value_shape;
      std::string phstore_path;
      std::string storage_option_string;
      StorageOption storage_option;
      OP_REQUIRES_OK(ctx,
                     GetNodeAttr(this->def(), "value_shape", &value_shape));
      // OP_REQUIRES(ctx, TensorShapeUtils::IsVector(value_shape),
      //             errors::InvalidArgument("Default value must be a vector, "
      //                                     "got shape ",
      //                                     value_shape));
      OP_REQUIRES_OK(
          ctx, GetNodeAttr(this->def(), "enter_threshold", &enter_threshold));

      StorageConfig storage_config;
      storage_config.set_training_storage_size(-1);
      storage_config.set_inference_storage_size(-1);
      storage_option.set_combination(StorageCombination::MEM);
      storage_option.mutable_configs()->insert({MEM, storage_config});
      auto creator =
          [ctx, this, enter_threshold, &value_shape, &storage_option](
              tfplus::KvVariableInterface** ret) {
            KvVariableInterface* container =
                new KvVariable<K, V>(this->cinfo_.name(), value_shape,
                                     enter_threshold, storage_option);

            if (ctx->track_allocations()) {
              ctx->record_persistent_memory_allocation(container->MemoryUsed());
            }
            *ret = container;
            return ::tensorflow::OkStatus();
          };  // NOLINT

      KvVariableInterface* table = nullptr;
      OP_REQUIRES_OK(
          ctx, cinfo_.resource_manager()
                   ->template LookupOrCreate<KvVariableInterface, true>(
                       cinfo_.container(), cinfo_.name(), &table, creator));
      core::ScopedUnref unref_me(table);

      handle_ = MakeResourceHandle<KvVariableInterface>(ctx, cinfo_.container(),
                                                        cinfo_.name());
      table_handle_set_.store(true, std::memory_order_release);
    }
    // Wait for the initialization to complete.
    // Just in case that Arks initiates more than one
    while (!table_handle_set_.load(std::memory_order_acquire)) {
      // do nothing
    }
    Tensor* handle;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
    handle->scalar<ResourceHandle>()() = handle_;
  }

 private:
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;
  std::atomic<bool> init_token_;
  std::atomic<bool> table_handle_set_;
  ResourceHandle handle_;
  TF_DISALLOW_COPY_AND_ASSIGN(CreateKvVariableOp);
};

#define REGISTER_KERNEL(key_dtype, value_dtype)                               \
  REGISTER_KERNEL_BUILDER(Name("KvVariable")                                  \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<key_dtype>("key_dtype")         \
                              .TypeConstraint<value_dtype>("value_dtype"),    \
                          tfplus::CreateKvVariableOp<key_dtype, value_dtype>) \
  REGISTER_KERNEL_BUILDER(Name("KvVariableV2")                                \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<key_dtype>("key_dtype")         \
                              .TypeConstraint<value_dtype>("value_dtype"),    \
                          tfplus::CreateKvVariableOp<key_dtype, value_dtype>) \
  REGISTER_KERNEL_BUILDER(Name("KvVariableV3")                                \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<key_dtype>("key_dtype")         \
                              .TypeConstraint<value_dtype>("value_dtype"),    \
                          tfplus::CreateKvVariableOp<key_dtype, value_dtype>) \
  REGISTER_KERNEL_BUILDER(Name("KvVariableV4")                                \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<key_dtype>("key_dtype")         \
                              .TypeConstraint<value_dtype>("value_dtype"),    \
                          tfplus::CreateKvVariableOp<key_dtype, value_dtype>)

REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(uint64, float);
// REGISTER_KERNEL(string, float);
// REGISTER_KERNEL(string, Eigen::half);
REGISTER_KERNEL(int32, Eigen::half);
REGISTER_KERNEL(int64, Eigen::half);
REGISTER_KERNEL(uint64, Eigen::half);
#undef REGISTER_KERNEL

template <typename T>
class KvVariableShapeOp : public OpKernel {
 public:
  explicit KvVariableShapeOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));

    core::ScopedUnref unref_me(table);
    TensorShape shape = table->GetShape();

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {shape.dims()}, &output));
    for (int i = 0; i < shape.dims(); ++i) {
      output->flat<T>()(i) = shape.dim_size(i);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KvVariableShapeV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("out_type"),
                        KvVariableShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("KvVariableShapeV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("out_type"),
                        KvVariableShapeOp<int64>);

class InitKvVariableOp : public OpKernel {
 public:
  explicit InitKvVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    const Tensor& random_init_table = ctx->input(1);
    OP_REQUIRES_OK(ctx, table->InitRandomValues(random_init_table));
  }
};

#define REGISTER_KERNEL(type)                                                \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("InitKvVariableV2").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      InitKvVariableOp);

REGISTER_KERNEL(int32)
REGISTER_KERNEL(int64)
REGISTER_KERNEL(uint64)
REGISTER_KERNEL(float)
REGISTER_KERNEL(Eigen::half)
#undef REGISTER_KERNEL

class KvVariableIsInitializedOp : public OpKernel {
 public:
  explicit KvVariableIsInitializedOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    auto output_tensor = output->tensor<bool, 0>();

    KvVariableInterface* table = nullptr;
    Status s = LookupResource(ctx, HandleFromInput(ctx, 0), &table);
    if (!s.ok()) {
      output_tensor() = false;
      return;
    }

    core::ScopedUnref unref_me(table);
    output_tensor(0) = table->IsInitialized();
  }
};

REGISTER_KERNEL_BUILDER(Name("KvVariableIsInitializedV2").Device(DEVICE_CPU),
                        KvVariableIsInitializedOp);

class KvVariableSizeOp : public OpKernel {
 public:
  explicit KvVariableSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    auto output_tensor = output->tensor<int64, 0>();

    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    // return the number of elements in the table
    output_tensor() = table->size();
  }
};

REGISTER_KERNEL_BUILDER(Name("KvVariableSizeV2").Device(DEVICE_CPU),
                        KvVariableSizeOp);

class KvVariableStorageSizeOp : public OpKernel {
 public:
  explicit KvVariableStorageSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);
    OP_REQUIRES_OK(ctx, table->CountStorageSize(ctx));
  }
};
REGISTER_KERNEL_BUILDER(Name("KvVariableSizeV3").Device(DEVICE_CPU),
                        KvVariableStorageSizeOp);

class KvVariableFrequencyOp : public OpKernel {
 public:
  explicit KvVariableFrequencyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    auto output_tensor = output->tensor<int64, 0>();

    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    // return the elements freq
    output_tensor() = table->sum_freq();
  }
};

REGISTER_KERNEL_BUILDER(Name("KvVariableFrequency").Device(DEVICE_CPU),
                        KvVariableFrequencyOp);

class DestroyKvVariableOp : public OpKernel {
 public:
  explicit DestroyKvVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("ignore_lookup_error", &ignore_lookup_error_));
  }

  void Compute(OpKernelContext* ctx) override {
    // destroy lookup table
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    // release resources
    Status status = DeleteResource(ctx, HandleFromInput(ctx, 0));
    // check the status
    // if (ignore_lookup_error_ && errors::IsNotFound(status)) {
    //   return;
    // }

    OP_REQUIRES_OK(ctx, status);
  }

 private:
  bool ignore_lookup_error_;
};

REGISTER_KERNEL_BUILDER(Name("DestroyKvVariableOpV2").Device(DEVICE_CPU),
                        DestroyKvVariableOp);

class ReadKvVariableOp : public OpKernel {
 public:
  explicit ReadKvVariableOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("Tkeys", &key_type_));
    OP_REQUIRES_OK(c, c->GetAttr("Tvalues", &value_type_));
  }

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);
    OP_REQUIRES_OK(ctx, table->ExportValues(ctx, 2));
  }

 private:
  DataType key_type_;
  DataType value_type_;
};

// We only register CPU kernels
REGISTER_KERNEL_BUILDER(Name("ReadKvVariableOpV2").Device(DEVICE_CPU),
                        ReadKvVariableOp);

template <typename T, typename Index>
class KvVariableGatherOrZerosOp : public OpKernel {
 public:
  explicit KvVariableGatherOrZerosOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    table_.store(nullptr, std::memory_order_relaxed);
  }

  ~KvVariableGatherOrZerosOp() {
    KvVariableInterface* table =
        table_.exchange(nullptr, std::memory_order_relaxed);
    if (table != nullptr) {
      table->Unref();
    }
  }

  void Compute(OpKernelContext* ctx) override {
    // TODO(mochen.bmc): Potential memory leak

    KvVariableInterface* table = table_.load(std::memory_order_relaxed);
    if (table == nullptr) {
      OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
      table_.store(table, std::memory_order_relaxed);
    }

    // get the indices
    const Tensor& indices = ctx->input(1);

    // get the number of indices
    const int64_t N = indices.NumElements();
    const TensorShape& value_shape = table->value_shape();

    // The result shape is indices.shape + value_shape.
    TensorShape result_shape = indices.shape();
    // OP_REQUIRES(ctx, result_shape.dims() == 1,
    //             errors::InvalidArgument("Shape of incies must be one"));

    // for the shape of the output
    for (int i = 0; i < value_shape.dims(); i++) {
      result_shape.AddDim(value_shape.dim_size(i));
    }

    // allocate the output buffer
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, result_shape, &out));

    // gather the data
    if (N > 0) {
      OP_REQUIRES_OK(ctx, table->FindOrZeros(ctx, indices, out));
    }

    // VLOG(1) << "table " << ctx->op_kernel().name() << " current size "
    //         << table->size();
  }

 private:
  std::atomic<KvVariableInterface*> table_;
};

#define REGISTER_GATHER_OR_ZEROS_FULL(dev, type, index_type)           \
  REGISTER_KERNEL_BUILDER(Name("KvVariableGatherOrZerosV2")            \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("table_handle")              \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          KvVariableGatherOrZerosOp<type, index_type>)

#define REGISTER_GATHER_OR_ZEROS_ALL_INDICES(dev, type) \
  REGISTER_GATHER_OR_ZEROS_FULL(dev, type, int32);      \
  REGISTER_GATHER_OR_ZEROS_FULL(dev, type, int64);      \
  REGISTER_GATHER_OR_ZEROS_FULL(dev, type, uint64);     \
  // REGISTER_GATHER_OR_ZEROS_FULL(dev, type, string)

#define REGISTER_GATHER_OR_ZEROS_CPU(type) \
  REGISTER_GATHER_OR_ZEROS_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_GATHER_OR_ZEROS_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_OR_ZEROS_CPU);
#undef REGISTER_GATHER_OR_ZEROS_CPU
#undef REGISTER_GATHER_OR_ZEROS_ALL_INDICES
#undef REGISTER_GATHER_OR_ZEROS_FULL

template <typename T, typename Index>
class BatchKvVariableGatherOrZerosOp : public OpKernel {
 public:
  explicit BatchKvVariableGatherOrZerosOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
      }

  ~BatchKvVariableGatherOrZerosOp() {
  }

  void Compute(OpKernelContext* ctx) override {
    for (int i = 0; i < N_; i++) {
      KvVariableInterface* table;
      OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, i), &table));
      // bugfix(mochen.bmc): memory leak
      core::ScopedUnref unref_me(table);
      const Tensor& indices = ctx->input(N_ + i);
      const int64_t ne = indices.NumElements();

      // The result shape is indices.shape + value_shape.
      const TensorShape& value_shape = table->value_shape();
      TensorShape result_shape = indices.shape();

      // for the shape of the output
      for (int i = 0; i < value_shape.dims(); i++) {
        result_shape.AddDim(value_shape.dim_size(i));
      }

      // allocate the output buffer
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, result_shape, &out));

      // gather the data
      if (ne > 0) {
        OP_REQUIRES_OK(ctx, table->FindOrZeros(ctx, indices, out));
      }
    }
  }

 private:
  int N_;
};

#define REGISTER_BATCH_GATHER_OR_ZEROS_FULL(dev, type, index_type)     \
  REGISTER_KERNEL_BUILDER(Name("BatchKvVariableGatherOrZerosV2")       \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          BatchKvVariableGatherOrZerosOp<type, index_type>)

#define REGISTER_BATCH_GATHER_OR_ZEROS_ALL_INDICES(dev, type) \
  REGISTER_BATCH_GATHER_OR_ZEROS_FULL(dev, type, int32);      \
  REGISTER_BATCH_GATHER_OR_ZEROS_FULL(dev, type, int64);      \
  REGISTER_BATCH_GATHER_OR_ZEROS_FULL(dev, type, uint64)
  // REGISTER_BATCH_GATHER_OR_ZEROS_FULL(dev, type, string)

#define REGISTER_BATCH_GATHER_OR_ZEROS_CPU(type) \
  REGISTER_BATCH_GATHER_OR_ZEROS_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_BATCH_GATHER_OR_ZEROS_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_BATCH_GATHER_OR_ZEROS_CPU);
#undef REGISTER_BATCH_GATHER_OR_ZEROS_CPU
#undef REGISTER_BATCH_GATHER_OR_ZEROS_ALL_INDICES
#undef REGISTER_BATCH_GATHER_OR_ZEROS_FULL

template <typename T, typename Index>
class KvVariableGatherOrInsertOp : public OpKernel {
 public:
  explicit KvVariableGatherOrInsertOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    // get the indices
    const Tensor& indices = ctx->input(1);

    // get the number of indices
    const int64_t N = indices.NumElements();
    TensorShape value_shape = table->value_shape();

    // The result shape is indices.shape + value_shape.
    TensorShape result_shape = indices.shape();
    // OP_REQUIRES(ctx, result_shape.dims() == 1,
    //             errors::InvalidArgument("Shape of incies must be one"));

    // for the shape of the output
    for (int i = 0; i < value_shape.dims(); i++) {
      result_shape.AddDim(value_shape.dim_size(i));
    }

    // allocate the output buffer
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, result_shape, &out));

    // gather the data
    if (N > 0) {
      OP_REQUIRES_OK(ctx, table->FindOrInsert(ctx, indices, out));
    }

    // VLOG(1) << "table " << ctx->op_kernel().name() << " current size "
    //         << table->size();
  }
};

#define REGISTER_GATHER_INSERT_FULL(dev, type, index_type)             \
  REGISTER_KERNEL_BUILDER(Name("KvVariableGatherOrInsertV2")           \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("table_handle")              \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          KvVariableGatherOrInsertOp<type, index_type>)

#define REGISTER_GATHER_INSERT_ALL_INDICES(dev, type) \
  REGISTER_GATHER_INSERT_FULL(dev, type, int32);      \
  REGISTER_GATHER_INSERT_FULL(dev, type, int64);      \
  REGISTER_GATHER_INSERT_FULL(dev, type, uint64)
  // REGISTER_GATHER_INSERT_FULL(dev, type, string)

#define REGISTER_GATHER_INSERT_CPU(type) \
  REGISTER_GATHER_INSERT_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_GATHER_INSERT_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_INSERT_CPU);
#undef REGISTER_GATHER_INSERT_CPU
#undef REGISTER_GATHER_INSERT_ALL_INDICES
#undef REGISTER_GATHER_INSERT_FULL

template <typename T, typename Index>
class KvVariableGatherOrInsertWithCountsOp : public OpKernel {
 public:
  explicit KvVariableGatherOrInsertWithCountsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    // get the indices
    const Tensor& indices = ctx->input(1);
    const Tensor& counts = ctx->input(2);

    // get the number of indices
    const int64_t N = indices.NumElements();
    TensorShape value_shape = table->value_shape();

    // The result shape is indices.shape + value_shape.
    TensorShape result_shape = indices.shape();
    // OP_REQUIRES(ctx, result_shape.dims() == 1,
    //             errors::InvalidArgument("Shape of incies must be one"));

    // for the shape of the output
    for (int i = 0; i < value_shape.dims(); i++) {
      result_shape.AddDim(value_shape.dim_size(i));
    }

    // allocate the output buffer
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, result_shape, &out));

    // gather the data
    if (N > 0) {
      OP_REQUIRES_OK(ctx,
                     table->FindOrInsertWithCounts(ctx, indices, counts, out));
    }

    // VLOG(1) << "table " << ctx->op_kernel().name() << " current size "
    //         << table->size();
  }
};

#define REGISTER_GATHER_INSERT_COUNTS_FULL(dev, type, index_type) \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("KvVariableGatherOrInsertWithCounts")                  \
          .Device(DEVICE_##dev)                                   \
          .HostMemory("table_handle")                             \
          .TypeConstraint<type>("dtype")                          \
          .TypeConstraint<index_type>("Tindices"),                \
      KvVariableGatherOrInsertWithCountsOp<type, index_type>)

#define REGISTER_GATHER_INSERT_COUNTS_ALL_INDICES(dev, type) \
  REGISTER_GATHER_INSERT_COUNTS_FULL(dev, type, int32);      \
  REGISTER_GATHER_INSERT_COUNTS_FULL(dev, type, int64);      \
  REGISTER_GATHER_INSERT_COUNTS_FULL(dev, type, uint64);
  // REGISTER_GATHER_INSERT_COUNTS_FULL(dev, type, string)

#define REGISTER_GATHER_INSERT_COUNTS_CPU(type) \
  REGISTER_GATHER_INSERT_COUNTS_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_GATHER_INSERT_COUNTS_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_INSERT_COUNTS_CPU);
#undef REGISTER_GATHER_INSERT_COUNTS_CPU
#undef REGISTER_GATHER_INSERT_COUNTS_ALL_INDICES
#undef REGISTER_GATHER_INSERT_COUNTS_FULL

template <typename T, typename Index>
class KvVariableGatherOp : public OpKernel {
 public:
  explicit KvVariableGatherOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    // get the indices
    const Tensor& indices = ctx->input(1);
    const Tensor& tensor = ctx->input(2);
    /*use init value*/
    auto use_init_value = tensor.flat<bool>()(0);
    // get the number of indices
    const int64_t N = indices.NumElements();
    TensorShape value_shape = table->value_shape();

    // The result shape is indices.shape + value_shape.
    TensorShape result_shape = indices.shape();
    // OP_REQUIRES(ctx, result_shape.dims() == 1,
    //             errors::InvalidArgument("Shape of incies must be one"));

    // for the shape of the output
    for (int i = 0; i < value_shape.dims(); i++) {
      result_shape.AddDim(value_shape.dim_size(i));
    }

    // allocate the output buffer
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, result_shape, &out));

    // gather the data
    if (N > 0) {
      if (!use_init_value) {
        OP_REQUIRES_OK(ctx, table->FindOrZeros(ctx, indices, out));
      } else {
        OP_REQUIRES_OK(ctx, table->FindOrInsert(ctx, indices, out));
      }
    }

    // VLOG(1) << "table " << ctx->op_kernel().name() << " current size "
    //         << table->size();
  }
};

#define REGISTER_GATHER_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("KvVariableGatherV2")                   \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("table_handle")              \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          KvVariableGatherOp<type, index_type>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, type, int32);      \
  REGISTER_GATHER_FULL(dev, type, int64);      \
  REGISTER_GATHER_FULL(dev, type, uint64);
  // REGISTER_GATHER_FULL(dev, type, string)

#define REGISTER_GATHER_CPU(type) REGISTER_GATHER_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_GATHER_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_CPU);
#undef REGISTER_GATHER_CPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

template <typename T, typename Index>
class KvVariableInsertOp : public OpKernel {
 public:
  explicit KvVariableInsertOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // get the table
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    const Tensor& indices = ctx->input(1);
    const Tensor& values = ctx->input(2);

    const int64_t N = indices.NumElements();
    if (N > 0) {
      // Insert it to the table if a key does not exist; otherwise, update its
      // value
      table->InsertOrUpdate(ctx, indices, values);
    }
  }
};

#define REGISTER_INSERT_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("KvVariableInsertV2")                   \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("table_handle")              \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          KvVariableInsertOp<type, index_type>)

#define REGISTER_INSERT_ALL_INDICES(dev, type) \
  REGISTER_INSERT_FULL(dev, type, int32);      \
  REGISTER_INSERT_FULL(dev, type, int64);      \
  REGISTER_INSERT_FULL(dev, type, uint64);     \
  // REGISTER_INSERT_FULL(dev, type, string);

#define REGISTER_INSERT_CPU(type) REGISTER_INSERT_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_INSERT_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_INSERT_CPU);
#undef REGISTER_INSERT_CPU
#undef REGISTER_INSERT_ALL_INDICES
#undef REGISTER_INSERT_FULL

template <typename Index>
class KvVariableIncreaseCountOp : public OpKernel {
 public:
  explicit KvVariableIncreaseCountOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    // reserved OP , do not call any more
  }
};

#define REGISTER_INCREASE_FREQUENCY_FULL(dev, index_type)              \
  REGISTER_KERNEL_BUILDER(Name("KvVariableIncreaseCountV2")            \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("table_handle")              \
                              .TypeConstraint<index_type>("Tindices"), \
                          KvVariableIncreaseCountOp<index_type>)

#define REGISTER_INCREASE_FREQUENCY_ALL_INDICES(dev) \
  REGISTER_INCREASE_FREQUENCY_FULL(dev, int32);      \
  REGISTER_INCREASE_FREQUENCY_FULL(dev, int64);      \
  REGISTER_INCREASE_FREQUENCY_FULL(dev, uint64);     \
  // REGISTER_INCREASE_FREQUENCY_FULL(dev, string);

// Registration of the CPU implementations.
REGISTER_INCREASE_FREQUENCY_ALL_INDICES(CPU)

#undef REGISTER_INCREASE_FREQUENCY_ALL_INDICES
#undef REGISTER_INCREASE_FREQUENCY_FULL

// Import the content of KvVariable
class KvVariableImportOp : public OpKernel {
 public:
  explicit KvVariableImportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("first_n", &first_n_));
    VLOG(1) << "first_n: " << first_n_;
  }

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    // get keys and values
    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    std::vector<Tensor> others;

    // check expected input data types
    DataTypeVector expected_inputs = {DT_RESOURCE,
                                      table->key_dtype(),
                                      table->value_dtype(),
                                      table->value_dtype(),
                                      table->key_dtype(),
                                      table->key_dtype(),
                                      DataTypeToEnum<uint16>::v()};
    // initialization table
    others.push_back(ctx->input(3));

    // blacklist
    if (first_n_ > 3) {
      others.push_back(ctx->input(4));
    } else {
      // fill an empty tensor
      others.push_back(Tensor());
    }

    // frequency table
    if (first_n_ > 4) {
      others.push_back(ctx->input(5));
      others.push_back(ctx->input(6));
    } else {
      // fill empty tensors
      others.push_back(Tensor());
      others.push_back(Tensor());
    }

    // OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    size_t before_size = table->size();
    OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values, others));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }

    // VLOG(0) << "table " << table->name() << " load before[" << before_size
    //         << "] after[" << table->size() << "]"
    //         << " load info keys:" << keys.NumElements()
    //         << " values:" << values.DebugString().c_str()
    //         << " blacklist: " << others[1].NumElements();
  }

 private:
  // control that we only import the first input tensors
  int first_n_;
};

REGISTER_KERNEL_BUILDER(Name("KvVariableImport").Device(DEVICE_CPU),
                        KvVariableImportOp);

// Import the content of KvVariable
class KvVariableFullOrDeltaImportOp : public OpKernel {
 public:
  explicit KvVariableFullOrDeltaImportOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("first_n", &first_n_));
    VLOG(1) << "first_n: " << first_n_;
  }

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    // get keys and values
    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    std::vector<Tensor> others;

    // check expected input data types
    DataTypeVector expected_inputs = {DT_RESOURCE,
                                      table->key_dtype(),
                                      table->value_dtype(),
                                      table->value_dtype(),
                                      table->key_dtype(),
                                      table->key_dtype(),
                                      DataTypeToEnum<uint32>::v(),
                                      DataTypeToEnum<bool>::v(),
                                      table->key_dtype()};
    // initialization table
    others.push_back(ctx->input(3));

    // blacklist
    others.push_back(ctx->input(4));

    // frequency table
    if (first_n_ > 4) {
      others.push_back(ctx->input(5));
      others.push_back(ctx->input(6));
    } else {
      // fill empty tensors
      others.push_back(Tensor());
      others.push_back(Tensor());
    }

    others.push_back(ctx->input(7));
    others.push_back(ctx->input(8));
    if (ctx->num_inputs() == 10) {
      others.push_back(ctx->input(9));
      expected_inputs.push_back(DataTypeToEnum<bool>::v());
    }
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }

    size_t before_size = table->size();
    OP_REQUIRES_OK(
        ctx, table->FullOrDeltaImport(ctx, first_n_, keys, values, others));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }

    // VLOG(0) << "table " << table->name()
    //         << " load before[" << before_size << "] after["
    //         << table->size() << "]"
    //         << " load info keys:" << keys.NumElements()
    //         << " values:" << values.DebugString().c_str()
    //         << " blacklist: " << others[1].NumElements()
    //         << " deletekeys: " << others[5].NumElements()
    //         << " is_full: " << others[4].template flat<bool>()(0);
  }

 private:
  // control that we only import the first input tensors
  int first_n_;
};

REGISTER_KERNEL_BUILDER(Name("KvVariableFullOrDeltaImport").Device(DEVICE_CPU),
                        KvVariableFullOrDeltaImportOp);

REGISTER_KERNEL_BUILDER(Name("KvVariableFullOrDeltaImportV2")
                        .Device(DEVICE_CPU),
                        KvVariableFullOrDeltaImportOp);

// Compatiable import op for old version, will be only
// used in loading model for tf serving. So we just
// import keys, values, and init_table, others will
// be ignored.
class KvVariableOldImportOp : public OpKernel {
 public:
  explicit KvVariableOldImportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    // get keys and values
    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    std::vector<Tensor> others;

    // check expected input data types
    DataTypeVector expected_inputs = {DT_RESOURCE, table->key_dtype(),
                                      table->value_dtype(),
                                      table->value_dtype(), table->key_dtype()};
    // initialization table
    others.push_back(ctx->input(3));

    // fill empty tensors
    others.push_back(Tensor());
    others.push_back(Tensor());
    others.push_back(Tensor());

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    size_t before_size = table->size();
    OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values, others));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KvVariableImportV2").Device(DEVICE_CPU),
                        KvVariableOldImportOp);
REGISTER_KERNEL_BUILDER(Name("KvVariableImportV3").Device(DEVICE_CPU),
                        KvVariableOldImportOp);

// Export the content of KvVariable
class KvVariableExportOp : public OpKernel {
 public:
  explicit KvVariableExportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("first_n", &first_n_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("enable_cutoff", &enable_cutoff_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cutoff_value", &cutoff_value_));

    VLOG(1) << "first_n: " << first_n_ << " enable_cutoff: " << enable_cutoff_
            << " cutoff_value: " << cutoff_value_;
  }

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(
        ctx, table->ExportValues(ctx, first_n_, enable_cutoff_, cutoff_value_));
  }

 private:
  int first_n_;
  bool enable_cutoff_;
  float cutoff_value_;
};

REGISTER_KERNEL_BUILDER(Name("KvVariableExport").Device(DEVICE_CPU),
                        KvVariableExportOp);

class KvVariableExportForMultiHashOp : public OpKernel {
 public:
  explicit KvVariableExportForMultiHashOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("first_n", &first_n_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("enable_cutoff", &enable_cutoff_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cutoff_value", &cutoff_value_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("variable_name", &variable_name_));

    VLOG(1) << "first_n: " << first_n_ << " enable_cutoff: " << enable_cutoff_
            << " cutoff_value: " << cutoff_value_;
  }

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(
        ctx, table->ExportValuesForMultiHash(ctx, first_n_, enable_cutoff_,
                                             cutoff_value_, variable_name_));
  }

 private:
  int first_n_;
  bool enable_cutoff_;
  float cutoff_value_;
  std::string variable_name_;
};

REGISTER_KERNEL_BUILDER(Name("KvVariableExportForMultiHash").Device(DEVICE_CPU),
                        KvVariableExportForMultiHashOp);
// A Fake op for KvVariableExportV2 and KvVariableExportV3, which used to only
// load graph, will not run this op. We use KvVariableExport instead.
class FakeKvVariableExportOp : public OpKernel {
 public:
  explicit FakeKvVariableExportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {}
};
REGISTER_KERNEL_BUILDER(Name("KvVariableExportV2").Device(DEVICE_CPU),
                        FakeKvVariableExportOp);
REGISTER_KERNEL_BUILDER(Name("KvVariableExportV3").Device(DEVICE_CPU),
                        FakeKvVariableExportOp);

// Export the content of KvVariable
class KvVariableFullOrDeltaExportOp : public OpKernel {
 public:
  explicit KvVariableFullOrDeltaExportOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("first_n", &first_n_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("enable_cutoff", &enable_cutoff_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cutoff_value", &cutoff_value_));

    VLOG(1) << "first_n: " << first_n_ << " enable_cutoff: " << enable_cutoff_
            << " cutoff_value: " << cutoff_value_;
  }

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);
    const Tensor& tensor = ctx->input(1);
    auto need_full_export = tensor.template flat<bool>()(0);

    OP_REQUIRES_OK(ctx,
                   table->FullOrDeltaExport(ctx, first_n_, enable_cutoff_,
                                            cutoff_value_, need_full_export));
  }

 private:
  int first_n_;
  bool enable_cutoff_;
  float cutoff_value_;
};

REGISTER_KERNEL_BUILDER(Name("KvVariableFullOrDeltaExport").Device(DEVICE_CPU),
                        KvVariableFullOrDeltaExportOp);

template <typename T, typename Index, ScatterUpdateOps OP>
class KvVariableScatterUpdateOP : public OpKernel {
 public:
  explicit KvVariableScatterUpdateOP(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    KvVariableInterface* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    const Tensor& indices = ctx->input(1);
    const Tensor& updates = ctx->input(2);

    const int64_t N = indices.NumElements();
    if (N > 0) {
      OP_REQUIRES_OK(ctx, table->ScatterUpdate(ctx, indices, updates, OP));
    }
  }
};

#define REGISTER_SCATTER_KERNEL_INDEX(type, index_type, dev, name, op) \
  REGISTER_KERNEL_BUILDER(Name(name)                                   \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("table_handle")              \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          KvVariableScatterUpdateOP<type, index_type, op>)

#define REGISTER_SCATTER_KERNEL(type, dev, name, op)          \
  REGISTER_SCATTER_KERNEL_INDEX(type, int32, dev, name, op);  \
  REGISTER_SCATTER_KERNEL_INDEX(type, int64, dev, name, op);  \
  REGISTER_SCATTER_KERNEL_INDEX(type, uint64, dev, name, op); \
  // REGISTER_SCATTER_KERNEL_INDEX(type, string, dev, name, op);

#define REGISTER_SCATTER_ARITHMETIC(type, dev)                    \
  REGISTER_SCATTER_KERNEL(type, dev, "KvVariableScatterAddV2",    \
                          SCATTER_UPDATE_ADD);                    \
  REGISTER_SCATTER_KERNEL(type, dev, "KvVariableScatterSubV2",    \
                          SCATTER_UPDATE_SUB);                    \
  REGISTER_SCATTER_KERNEL(type, dev, "KvVariableScatterMulV2",    \
                          SCATTER_UPDATE_MUL);                    \
  REGISTER_SCATTER_KERNEL(type, dev, "KvVariableScatterDivV2",    \
                          SCATTER_UPDATE_DIV);                    \
  REGISTER_SCATTER_KERNEL(type, dev, "KvVariableScatterUpdateV2", \
                          SCATTER_UPDATE_ASSIGN);

#define REGISTER_SCATTER_MINMAX(type, dev)                     \
  REGISTER_SCATTER_KERNEL(type, dev, "KvVariableScatterMinV2", \
                          SCATTER_UPDATE_MIN);                 \
  REGISTER_SCATTER_KERNEL(type, dev, "KvVariableScatterMaxV2", \
                          SCATTER_UPDATE_MAX);

// Registers CPU kernels.
#define REGISTER_SCATTER_ARITHMETIC_CPU(type) \
  REGISTER_SCATTER_ARITHMETIC(type, CPU);
#define REGISTER_SCATTER_MINMAX_CPU(type) REGISTER_SCATTER_MINMAX(type, CPU);

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ARITHMETIC_CPU);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_SCATTER_MINMAX_CPU);

#undef REGISTER_SCATTER_ARITHMETIC
#undef REGISTER_SCATTER_ARITHMETIC_CPU
#undef REGISTER_SCATTER_MINMAX
#undef REGISTER_SCATTER_MINMAX_CPU
#undef REGISTER_SCATTER_KERNEL
#undef REGISTER_SCATTER_KERNEL_INDEX
}  // namespace tfplus
