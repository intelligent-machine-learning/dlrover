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

#define EIGEN_USE_THREADS
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/bounds_check.h"

#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tfplus/kv_variable/kernels/kv_variable.h"
#include "tfplus/kv_variable/kernels/kv_variable_interface.h"
#include "tfplus/kv_variable/kernels/utility.h"

namespace {

// from tensorflow
// https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/core/kernels/training_ops.cc#L1227
template <typename T>
inline T FtrlCompute(T accum, T linear, T lr, T l1, T l2, T lr_power) {
  T quadratic;
  if (lr_power == static_cast<T>(-0.5)) {
    quadratic = Eigen::numext::sqrt(accum) / lr + static_cast<T>(2) * l2;
  } else {
    quadratic =
        Eigen::numext::pow(accum, -lr_power) / lr + static_cast<T>(2) * l2;
  }
  auto l1_reg_adjust = std::max(std::min(linear, l1), -l1);
  return (l1_reg_adjust - linear) / quadratic;
}
}  // namespace

namespace tfplus {
using namespace tensorflow;  // NOLINT(build/namespaces)

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using SYCLDevice = Eigen::SyclDevice;

// Copy from tensorflow since we cannot use training_op_helpers.h
// https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/training_op_helpers.h#L71
// Utility structure that releases a sequence of borrowed mutexes when it is
// deleted.
struct VariableInputLockHolder {
 public:
  VariableInputLockHolder(
      std::vector<ResourceBase*> vars,
      std::unique_ptr<std::vector<mutex_lock>> locks,
      std::unique_ptr<std::vector<tf_shared_lock>> shared_locks)
      : vars_(std::move(vars)),
        locks_(std::move(locks)),
        shared_locks_(std::move(shared_locks)) {}

  VariableInputLockHolder(VariableInputLockHolder&& other)
      : vars_(std::move(other.vars_)),
        locks_(std::move(other.locks_)),
        shared_locks_(std::move(other.shared_locks_)) {}

  ~VariableInputLockHolder() {
    // Release the locks before unreffing the Vars, because each lock
    // is potentially borrowed from a Var in vars_.
    locks_.reset();
    shared_locks_.reset();
    for (auto var : vars_) {
      var->Unref();
    }
  }

 private:
  std::vector<ResourceBase*> vars_;
  // NOTE: Use a `std::unique_ptr` instead of moving in a vector directly,
  // because a `std::vector<mutex_lock>` is not movable on all platforms.
  std::unique_ptr<std::vector<mutex_lock>> locks_;
  std::unique_ptr<std::vector<tf_shared_lock>> shared_locks_;
};

// Copy from tensorflow since we cannot use training_op_helpers.h
// https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/training_op_helpers.h#L104
// For the dense feature, not implement sparse feature.
tf_mutex* GetTrainingVariableMutex(OpKernelContext* ctx, int input,
                                        bool sparse,
                                        ResourceBase** maybe_resource) {
  *maybe_resource = nullptr;
  if (ctx->input_dtype(input) == DT_RESOURCE) {
    Status s;
    if (sparse) {
      KvVariableInterface* p = nullptr;
      s = LookupResource(ctx, HandleFromInput(ctx, input), &p);
      if (s.ok()) {
        *maybe_resource = p;
        return p->mu();
      }
    } else {
      Var* p = nullptr;
      s = LookupResource(ctx, HandleFromInput(ctx, input), &p);
      if (s.ok()) {
        *maybe_resource = p;
        return p->mu();
      }
    }
    ctx->CtxFailureWithWarning(errors::Internal("Invalid variable reference."));
    return nullptr;
  }
  return ctx->input_ref_mutex(input);
}

// MaybeLockVariableInputMutexesInOrder is a helper function to acquire mutexes
// in address order to mitigate deadlock.  Returns a vector of acquired mutexes.
// Safe to pass duplicates - will only lock each distinct mutex once.  If
// do_lock is false, returns immediately.  Note that this silently doesn't lock
// mutexes for invalid variable references; in all usages this is followed by
// GetInputTensor which will signal a failure.
// Copy from tensorflow since we cannot use training_op_helpers.h
// https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/training_op_helpers.h#L140
VariableInputLockHolder MaybeLockVariableInputMutexesInOrder(
    OpKernelContext* ctx, bool do_lock, const std::vector<int>& input_ids,
    bool sparse = true) {
  bool any_resource = false;
  for (auto i : input_ids) {
    if (ctx->input_dtype(i) == DT_RESOURCE) {
      any_resource = true;
      break;
    }
  }
  if (!do_lock && !any_resource) {
    return VariableInputLockHolder({}, {}, {});
  }
  std::vector<ResourceBase*> vars;
  std::vector<tf_mutex*> mutexes;
  std::vector<int> acquire_order;
  for (auto input : input_ids) {
    ResourceBase* var;
    tf_mutex* mutex = GetTrainingVariableMutex(ctx, input, sparse, &var);
    if (var) {
      vars.push_back(var);
    }
    // Only lock each mutex once if duplicates exist (n^2 but n is 2 or 3).
    if (std::find(mutexes.begin(), mutexes.end(), mutex) == mutexes.end()) {
      acquire_order.push_back(mutexes.size());
      mutexes.push_back(mutex);
    }
  }
  std::sort(acquire_order.begin(), acquire_order.end(),
            [&mutexes](int a, int b) { return mutexes[a] < mutexes[b]; });

  auto locks = absl::make_unique<std::vector<mutex_lock>>();
  auto shared_locks = absl::make_unique<std::vector<tf_shared_lock>>();
  if (!sparse || do_lock) {
    locks->reserve(acquire_order.size());
  } else {
    shared_locks->reserve(acquire_order.size());
  }

  for (auto input : acquire_order) {
    ResourceBase* var;
    tf_mutex* mu = GetTrainingVariableMutex(ctx, input, sparse, &var);
    core::ScopedUnref scoped_unref(var);
    if (mu != nullptr) {
      if (!sparse || do_lock) {
        locks->emplace_back(*static_cast<tf_mutex*>(mu));
      } else {
        shared_locks->emplace_back(*static_cast<tf_mutex*>(mu));
      }
    }
  }
  return VariableInputLockHolder(std::move(vars), std::move(locks),
                                 std::move(shared_locks));
}

template <typename Device, typename T>
struct DenseUpdate {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) = update;
  }
};

template <class T>
typename ::tensorflow::TTypes<T>::Tensor FlatVector(T* data,
                                                    const int64& num_elements) {
  ::Eigen::array<::Eigen::DenseIndex, 1> dims({num_elements});
  return typename ::tensorflow::TTypes<T>::Tensor(data, dims);
}

template <class Tindex, class T>
typename ::tensorflow::TTypes<T>::Tensor GetFlat(KvVariableInterface* table,
                                                 const Tindex& key,
                                                 const int64& num_elements) {
  EVContext<T> var_context;
  static_cast<KvVariable<Tindex, T>*>(table)->FindOrInsertUnsafe(
      key, &var_context, nullptr);
  return FlatVector<T>(var_context.Value(), num_elements);
}

// Copy from tensorflow since we cannot use training_op_helpers.h
// https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/training_op_helpers.h#L195
// This is for use with ResourceVariables to ensure *tensor has a
// reference count of 1 before you update it.
// REQUIRES: If you pass in variable->tensor(), *variable->mu() must be held.
template <typename Device, typename T>
Status PrepareToUpdateVariables(OpKernelContext* ctx, Tensor* tensor) {
  // Tensor's buffer is in use by some read, so we need to copy before
  // updating.
  Tensor tmp;
  if (std::is_same<T, Variant>::value) {
    AllocatorAttributes attr;
    attr.set_on_host(true);
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(tensor->dtype(), tensor->shape(), &tmp, attr));

    const auto elements_in = tensor->flat<Variant>();
    auto elements_out = tmp.flat<Variant>();
    for (int64_t i = 0; i < elements_in.size(); ++i) {
      elements_out(i) = elements_in(i);
    }
  } else {
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(tensor->dtype(), tensor->shape(), &tmp, attr));
    functor::DenseUpdate<Device, T, ASSIGN> copy_functor;
    copy_functor(ctx->eigen_device<Device>(), tmp.flat<T>(),
                  const_cast<const Tensor*>(tensor)->flat<T>());
  }
  *tensor = tmp;
  return OkStatus();
}
// Copy from tensorflow since we cannot use training_op_helpers.h
// https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/training_op_helpers.cc#L23
void MaybeForwardRefInputToRefOutput(OpKernelContext* ctx, int input,
                                     int output) {
  if (ctx->input_dtype(input) != DT_RESOURCE) {
    ctx->forward_ref_input_to_ref_output(input, output);
  }
}

// Copy from tensorflow since we cannot use training_op_helpers.h,
// not implement sparse case.
// https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/training_op_helpers.h#L232
// This gives you `*out`, a tensor you can update, corresponding to a variable
// passed as input index `input`.  This handles the differences between
// reference and resource variables. For reference variables we can just grab
// the tensor, grabbing the lock if lock_held is False.
//
// For resource variables we, if sparse is true, ensure it's in copy-on-read
// mode, and then, regardless of the value of sparse, ensure its refcount is 1
// (by potentially copying its contents). In this case lock_held is ignored.
template <typename Device, typename T>
Status GetInputTensorFromVariable(OpKernelContext* ctx, int input,
                                  bool lock_held, bool sparse, Tensor* out) {
  if (ctx->input_dtype(input) == DT_RESOURCE) {
    Var* var;
    TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, input), &var));
    core::ScopedUnref unref_var(var);
    TF_RETURN_IF_ERROR(PrepareToUpdateVariables<Device, T>(ctx, var->tensor()));
    *out = *var->tensor();
    return ::tensorflow::OkStatus();
  }
  *out = ctx->mutable_input(input, lock_held);
  return ::tensorflow::OkStatus();
}

template <typename Device, typename T, typename Tindex, bool has_l2_shrinkage>
class KvVariableSparseApplyFtrlOp : public OpKernel {
 public:
  explicit KvVariableSparseApplyFtrlOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    /**
     * TODO, set use_exclusive_lock=true can lead to deadlocks
     * fix this problem in another issue
     */
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2});
    // get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum = nullptr,
                        *table_linear = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum));
    core::ScopedUnref unref_me_accum(table_accum);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 2), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    // get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const int lr_power_index = has_l2_shrinkage ? 9 : 8;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr_power.shape()) &&
                    lr_power.scalar<T>()() <= static_cast<T>(0),
                errors::InvalidArgument("lr_power is not a "
                                        "non-positive scalar: ",
                                        lr_power.shape().DebugString()));

    int64_t N = indices.dim_size(0);
    TensorShape var_shape, accum_shape, linear_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    accum_shape = table_accum->value_shape();
    accum_shape.InsertDim(0, N);

    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "kv_varaible and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument(
                    "kv_variable and linear do not have the same shape",
                    var_shape.DebugString(), " ", linear_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Tensor* l2_shrinkage;
    if (has_l2_shrinkage) {
      l2_shrinkage = &ctx->input(8);
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(l2_shrinkage->shape()) &&
              l2_shrinkage->scalar<T>()() >= static_cast<T>(0),
          errors::InvalidArgument("l2 shrinkage regularization strength "
                                  "is not a non-negative scalar: ",
                                  l2_shrinkage->shape().DebugString()));
    }

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_accum,
                     &table_linear, &indices, &grad, &l2_shrinkage, &lr, &l1,
                     &l2, &lr_power](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();

        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l2_shrinkage_scalar;
        if (has_l2_shrinkage) {
          l2_shrinkage_scalar = l2_shrinkage->scalar<T>()();
        }
        T lr_power_scalar = lr_power.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> linear_context;
          EVContext<T> accum_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum)
              ->FindOrInsertUnsafe(key, &accum_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto accum = FlatVector<T>(accum_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);
// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_FTRL(grad_to_use)                                              \
  auto new_accum = accum + grad_to_use.square();                               \
  if (lr_power_scalar == static_cast<T>(-0.5)) {                               \
    linear +=                                                                  \
        grad_to_use - (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;     \
  } else {                                                                     \
    linear += grad_to_use - (new_accum.pow(-lr_power_scalar) -                 \
                             accum.pow(-lr_power_scalar)) /                    \
                                lr_scalar * var;                               \
  }                                                                            \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto x = l1_reg_adjust - linear;                                             \
  if (lr_power_scalar == static_cast<T>(-0.5)) {                               \
    auto y = new_accum.sqrt() / new_accum.constant(lr_scalar) +                \
             linear.constant(static_cast<T>(2) * l2_scalar);                   \
    var = x / y;                                                               \
  } else {                                                                     \
    auto y = new_accum.pow(-lr_power_scalar) / new_accum.constant(lr_scalar) + \
             linear.constant(static_cast<T>(2) * l2_scalar);                   \
    var = x / y;                                                               \
  }                                                                            \
  accum += grad_to_use.square();

          if (has_l2_shrinkage) {
            auto grad_with_shrinkage =
                grad + static_cast<T>(2) * l2_shrinkage_scalar * var;
            COMPUTE_FTRL(grad_with_shrinkage);
          } else {
            COMPUTE_FTRL(grad);
          }
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_FTRL
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableSparseApplyFtrl: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                     \
  REGISTER_KERNEL_BUILDER(                                \
      Name("KvVariableSparseApplyFtrlV2")                 \
          .Device(DEVICE_CPU)                             \
          .TypeConstraint<T>("T")                         \
          .TypeConstraint<Tindices>("Tindices"),          \
      KvVariableSparseApplyFtrlOp<CPUDevice, T, Tindices, \
                                  /*has_l2_shrinkage=*/true>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Sparse Group Lasso + Ftrl-Proximal
template <typename Device, typename T, typename Tindex, bool has_l2_shrinkage>
class KvVariableSparseGroupSparseApplyFtrlOp : public OpKernel {
 public:
  explicit KvVariableSparseGroupSparseApplyFtrlOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum = nullptr,
                        *table_linear = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum));
    core::ScopedUnref unref_me_accum(table_accum);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 2), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Faied to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const Tensor& l21 = ctx->input(8);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l21.shape().DebugString()));
    const int lr_power_index = has_l2_shrinkage ? 10 : 9;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr_power.shape()) &&
                    lr_power.scalar<T>()() <= static_cast<T>(0),
                errors::InvalidArgument("lr_power is not a "
                                        "non-positive scalar: ",
                                        lr_power.shape().DebugString()));

    int64_t N = indices.dim_size(0);
    TensorShape var_shape, accum_shape, linear_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    accum_shape = table_accum->value_shape();
    accum_shape.InsertDim(0, N);

    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "kv_varaible and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument(
                    "kv_variable and linear do not have the same shape",
                    var_shape.DebugString(), " ", linear_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Tensor* l2_shrinkage;
    if (has_l2_shrinkage) {
      l2_shrinkage = &ctx->input(9);
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(l2_shrinkage->shape()) &&
              l2_shrinkage->scalar<T>()() >= static_cast<T>(0),
          errors::InvalidArgument("l2 shrinkage regularization strength "
                                  "is not a non-negative scalar: ",
                                  l2_shrinkage->shape().DebugString()));
    }

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_accum,
                     &table_linear, &indices, &grad, &l2_shrinkage, &lr, &l1,
                     &l2, &l21, &lr_power](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        T l2_shrinkage_scalar;
        if (has_l2_shrinkage) {
          l2_shrinkage_scalar = l2_shrinkage->scalar<T>()();
        }
        T lr_power_scalar = lr_power.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> linear_context;
          EVContext<T> accum_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum)
              ->FindOrInsertUnsafe(key, &accum_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto accum = FlatVector<T>(accum_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_FTRL(grad_to_use)                                              \
  auto new_accum = accum + grad_to_use.square();                               \
  if (lr_power_scalar == static_cast<T>(-0.5)) {                               \
    linear +=                                                                  \
        grad_to_use - (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;     \
  } else {                                                                     \
    linear += grad_to_use - (new_accum.pow(-lr_power_scalar) -                 \
                             accum.pow(-lr_power_scalar)) /                    \
                                lr_scalar * var;                               \
  }                                                                            \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - (l21_norm / l1_linear_norm);          \
    if (lr_power_scalar == static_cast<T>(-0.5)) {                             \
      auto y = new_accum.sqrt() / new_accum.constant(lr_scalar) +              \
               linear.constant(static_cast<T>(2) * l2_scalar);                 \
      var = l1_linear * l1_linear_norm / y;                                    \
    } else {                                                                   \
      auto y =                                                                 \
          new_accum.pow(-lr_power_scalar) / new_accum.constant(lr_scalar) +    \
          linear.constant(static_cast<T>(2) * l2_scalar);                      \
      var = l1_linear * l1_linear_norm / y;                                    \
    }                                                                          \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  accum += grad_to_use.square();                                               \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);                               \
  static_cast<KvVariable<Tindex, T>*>(table_accum)                             \
      ->CoverUpdateUnsafe(key, &accum_context);

          if (has_l2_shrinkage) {
            auto grad_with_shrinkage =
                grad + static_cast<T>(2) * l2_shrinkage_scalar * var;
            COMPUTE_FTRL(grad_with_shrinkage);
          } else {
            COMPUTE_FTRL(grad);
          }
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_FTRL
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableSparseGroupSparseApplyFtrl: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("KvVariableSparseGroupSparseApplyFtrlV2")                 \
          .Device(DEVICE_CPU)                                        \
          .TypeConstraint<T>("T")                                    \
          .TypeConstraint<Tindices>("Tindices"),                     \
      KvVariableSparseGroupSparseApplyFtrlOp<CPUDevice, T, Tindices, \
                                             /*has_l2_shrinkage=*/true>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Group Lasso + Ftrl-Proximal
template <typename Device, typename T, typename Tindex, bool has_l2_shrinkage>
class KvVariableGroupSparseApplyFtrlOp : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyFtrlOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum = nullptr,
                        *table_linear = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum));
    core::ScopedUnref unref_me_accum(table_accum);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 2), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const int lr_power_index = has_l2_shrinkage ? 9 : 8;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr_power.shape()) &&
                    lr_power.scalar<T>()() <= static_cast<T>(0),
                errors::InvalidArgument("lr_power is not a "
                                        "non-positive scalar: ",
                                        lr_power.shape().DebugString()));

    int64_t N = indices.dim_size(0);
    TensorShape var_shape, accum_shape, linear_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    accum_shape = table_accum->value_shape();
    accum_shape.InsertDim(0, N);

    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "kv_varaible and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument(
                    "kv_variable and linear do not have the same shape",
                    var_shape.DebugString(), " ", linear_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Tensor* l2_shrinkage;
    if (has_l2_shrinkage) {
      l2_shrinkage = &ctx->input(8);
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(l2_shrinkage->shape()) &&
              l2_shrinkage->scalar<T>()() >= static_cast<T>(0),
          errors::InvalidArgument("l2 shrinkage regularization strength "
                                  "is not a non-negative scalar: ",
                                  l2_shrinkage->shape().DebugString()));
    }

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_accum,
                     &table_linear, &indices, &grad, &l2_shrinkage, &lr, &l1,
                     &l2, &lr_power](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l2_shrinkage_scalar;
        if (has_l2_shrinkage) {
          l2_shrinkage_scalar = l2_shrinkage->scalar<T>()();
        }
        T lr_power_scalar = lr_power.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> linear_context;
          EVContext<T> accum_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum)
              ->FindOrInsertUnsafe(key, &accum_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto accum = FlatVector<T>(accum_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_FTRL(grad_to_use)                                            \
  auto new_accum = accum + grad_to_use.square();                             \
  if (lr_power_scalar == static_cast<T>(-0.5)) {                             \
    linear +=                                                                \
        grad_to_use - (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;   \
  } else {                                                                   \
    linear += grad_to_use - (new_accum.pow(-lr_power_scalar) -               \
                             accum.pow(-lr_power_scalar)) /                  \
                                lr_scalar * var;                             \
  }                                                                          \
  ::Eigen::Tensor<T, 0, ::Eigen::RowMajor> linear_sqrsum =                   \
      linear.square().sum().sqrt();                                          \
  T linear_norm = linear_sqrsum(0);                                          \
  if (linear_norm > l1_scalar) {                                             \
    if (lr_power_scalar == static_cast<T>(-0.5)) {                           \
      auto eta_rec = new_accum.sqrt() / new_accum.constant(lr_scalar);       \
      auto coef = (l1_scalar - linear_norm) /                                \
                  ((eta_rec + static_cast<T>(2) * l2_scalar) * linear_norm); \
      var = coef * linear;                                                   \
    } else {                                                                 \
      auto eta_rec = new_accum.pow(-lr_power_scalar) / lr_scalar;            \
      auto coef = (l1_scalar - linear_norm) /                                \
                  ((eta_rec + static_cast<T>(2) * l2_scalar) * linear_norm); \
      var = coef * linear;                                                   \
    }                                                                        \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(       \
        key, &var_context);                                                  \
  } else {                                                                   \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(     \
        key, &var_context);                                                  \
  }                                                                          \
  accum += grad_to_use.square();                                             \
  accum += grad_to_use.square();                                             \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                          \
      ->CoverUpdateUnsafe(key, &linear_context);                             \
  static_cast<KvVariable<Tindex, T>*>(table_accum)                           \
      ->CoverUpdateUnsafe(key, &accum_context);

          if (has_l2_shrinkage) {
            auto grad_with_shrinkage =
                grad + static_cast<T>(2) * l2_shrinkage_scalar * var;
            COMPUTE_FTRL(grad_with_shrinkage);
          } else {
            COMPUTE_FTRL(grad);
          }
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_FTRL
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }
    VLOG(1) << "KvVariableSparseGroupSparseApplyFtrl: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                          \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("KvVariableGroupSparseApplyFtrlV2")                 \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .TypeConstraint<Tindices>("Tindices"),               \
      KvVariableGroupSparseApplyFtrlOp<CPUDevice, T, Tindices, \
                                       /*has_l2_shrinkage=*/true>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyAdamOp : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyAdamOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3, 4});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum = nullptr,
                        *table_linear = nullptr, *table_m = nullptr,
                        *table_v = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum));
    core::ScopedUnref unref_me_accum(table_accum);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 2), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 3), &table_m));
    core::ScopedUnref unref_me_m(table_m);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 4), &table_v));
    core::ScopedUnref unref_me_v(table_v);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(3)));
    OP_REQUIRES(
        ctx, table_v->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(4)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(5);
    const Tensor& indices = ctx->input(6);
    const Tensor& lr = ctx->input(7);
    const Tensor& beta1_power = ctx->input(8);
    const Tensor& beta2_power = ctx->input(9);
    const Tensor& beta1 = ctx->input(10);
    const Tensor& beta2 = ctx->input(11);
    const Tensor& epsilon = ctx->input(12);
    const Tensor& l1 = ctx->input(13);
    const Tensor& l2 = ctx->input(14);
    const Tensor& l21 = ctx->input(15);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, accum_shape, linear_shape, m_shape, v_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    accum_shape = table_accum->value_shape();
    accum_shape.InsertDim(0, N);

    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);

    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);

    v_shape = table_v->value_shape();
    v_shape.InsertDim(0, N);

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "kv_varaible and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument(
                    "kv_variable and linear do not have the same shape",
                    var_shape.DebugString(), " ", linear_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("kv_variable and m do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("kv_variable and v do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_accum,
                     &table_linear, &table_m, &table_v, &indices, &grad, &lr,
                     &beta1_power, &beta2_power, &beta1, &beta2, &epsilon, &l1,
                     &l2, &l21](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();

        T lr_scalar = lr.scalar<T>()();
        T beta1_power_scalar = beta1_power.scalar<T>()();
        T beta2_power_scalar = beta2_power.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));

          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> linear_context;
          EVContext<T> accum_context;
          EVContext<T> m_context;
          EVContext<T> v_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_m)
              ->FindOrInsertUnsafe(key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_v)
              ->FindOrInsertUnsafe(key, &v_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum)
              ->FindOrInsertUnsafe(key, &accum_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto accum = FlatVector<T>(accum_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(v_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_ADAM(grad_to_use)                                              \
  m = beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use;     \
  v = beta2_scalar * v +                                                       \
      (static_cast<T>(1) - beta2_scalar) * grad_to_use.square();               \
  auto new_accum = v / (static_cast<T>(1) - beta2_power_scalar);               \
  auto epsilon_adjust =                                                        \
      epsilon_scalar /                                                         \
      Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar);             \
  if (beta1_scalar > beta1_power_scalar) {                                     \
    linear += m / (static_cast<T>(1) - beta1_power_scalar) -                   \
              (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;             \
  } else {                                                                     \
    linear +=                                                                  \
        m / (static_cast<T>(1) - beta1_power_scalar) -                         \
        (new_accum.sqrt() - accum.sqrt() + accum.constant(epsilon_adjust)) /   \
            lr_scalar * var;                                                   \
  }                                                                            \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y =                                                                   \
        (new_accum.sqrt() + new_accum.constant(epsilon_adjust)) / lr_scalar +  \
        linear.constant(static_cast<T>(2) * l2_scalar);                        \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  accum = new_accum;                                                           \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);                               \
  static_cast<KvVariable<Tindex, T>*>(table_m)->CoverUpdateUnsafe(key,         \
                                                                  &m_context); \
  static_cast<KvVariable<Tindex, T>*>(table_v)->CoverUpdateUnsafe(key,         \
                                                                  &v_context); \
  static_cast<KvVariable<Tindex, T>*>(table_accum)                             \
      ->CoverUpdateUnsafe(key, &accum_context);
          COMPUTE_ADAM(grad);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_ADAM
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_v->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyAdamOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyAdamV2")   \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyAdamOp<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename T, typename Tindex>
class KvVariableSparseApplyAdagradOp : public OpKernel {
 public:
  explicit KvVariableSparseApplyAdagradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks =
        MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum));
    core::ScopedUnref unref_me_var(table_var);
    core::ScopedUnref unref_me_accum(table_accum);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    const int64_t N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    TensorShape var_shape, accum_shape;
    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    accum_shape = table_accum->value_shape();
    accum_shape.InsertDim(0, N);

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var_shape),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));
    if (N > 0) {
      auto DoWork = [this, ctx, &table_var, &table_accum, inner_dim,
                     &indices, &grad,
                     &lr](int64_t start_i, int64_t limit_i) {
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();
        T lr_scalar = lr.scalar<T>()();
        const int64_t embedding_dim_size = grad.dim_size(1);
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> accum_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_accum)
              ->FindOrInsertUnsafe(key, &accum_context, nullptr);
          auto v = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto a = FlatVector<T>(accum_context.Value(), embedding_dim_size);

          auto g = grad_flat.template chip<0>(i);
          if (update_slots_) {
            a += g.square();
          }

          if (inner_dim > 1) {
            v -= g.constant(lr_scalar) * g * a.rsqrt();
          } else {
            v -= g.constant(lr_scalar) * g / a.sqrt();
          }
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }
    VLOG(1) << "KvVariableSparseApplyAdagradOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
};
#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("KvVariableSparseApplyAdagrad")       \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          KvVariableSparseApplyAdagradOp<T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyAMSGradOp : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyAMSGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3, 4});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_vhat = nullptr,
                        *table_linear = nullptr, *table_m = nullptr,
                        *table_v = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_vhat));
    core::ScopedUnref unref_me_vhat(table_vhat);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 2), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 3), &table_m));
    core::ScopedUnref unref_me_m(table_m);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 4), &table_v));
    core::ScopedUnref unref_me_v(table_v);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_vhat->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(3)));
    OP_REQUIRES(
        ctx, table_v->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(4)));

    // Get gradients, indices and other parameters
    const Tensor& grad = ctx->input(5);
    const Tensor& indices = ctx->input(6);
    const Tensor& lr = ctx->input(7);
    const Tensor& beta1_power = ctx->input(8);
    const Tensor& beta2_power = ctx->input(9);
    const Tensor& beta1 = ctx->input(10);
    const Tensor& beta2 = ctx->input(11);
    const Tensor& epsilon = ctx->input(12);
    const Tensor& l1 = ctx->input(13);
    const Tensor& l2 = ctx->input(14);
    const Tensor& l21 = ctx->input(15);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, vhat_shape, linear_shape, m_shape, v_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    vhat_shape = table_vhat->value_shape();
    vhat_shape.InsertDim(0, N);

    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);

    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);

    v_shape = table_v->value_shape();
    v_shape.InsertDim(0, N);

    OP_REQUIRES(ctx, var_shape.IsSameSize(vhat_shape),
                errors::InvalidArgument("kv_variable and vhat do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        vhat_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument("kv_variable and linear do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        linear_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("kv_variable and m do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("kv_variable and v do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); ++d) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(ctx, grad.dim_size(0) == N,
                errors::InvalidArgument("grad must be the same size "
                                        "as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument("Inner dimension should be "
                                        "greater than zero."));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_vhat,
                     &table_linear, &table_m, &table_v, &indices, &grad, &lr,
                     &beta1_power, &beta2_power, &beta1, &beta2, &epsilon, &l1,
                     &l2, &l21](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto indices_flat = indices.flat<Tindex>();
        auto grad_flat = grad.flat_outer_dims<T>();

        T lr_scalar = lr.scalar<T>()();
        T beta1_power_scalar = beta1_power.scalar<T>()();
        T beta2_power_scalar = beta2_power.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; ++i) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> vhat_context;
          EVContext<T> m_context;
          EVContext<T> v_context;
          EVContext<T> linear_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_vhat)
              ->FindOrInsertUnsafe(key, &vhat_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_m)
              ->FindOrInsertUnsafe(key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_v)
              ->FindOrInsertUnsafe(key, &v_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto vhat = FlatVector<T>(vhat_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(v_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_AMSGrad(grad_to_use)                                           \
  m = beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use;     \
  v = beta2_scalar * v +                                                       \
      (static_cast<T>(1) - beta2_scalar) * grad_to_use.square();               \
  auto new_vhat = vhat.cwiseMax(v / (static_cast<T>(1) - beta2_power_scalar)); \
  linear += m / (static_cast<T>(1) - beta1_power_scalar) -                     \
            (new_vhat.sqrt() - vhat.sqrt()) / lr_scalar * var;                 \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y =                                                                   \
        (new_vhat.sqrt() + new_vhat.constant(epsilon_scalar)) / lr_scalar +    \
        linear.constant(static_cast<T>(2) * l2_scalar);                        \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  vhat = new_vhat;                                                             \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);                               \
  static_cast<KvVariable<Tindex, T>*>(table_m)->CoverUpdateUnsafe(key,         \
                                                                  &m_context); \
  static_cast<KvVariable<Tindex, T>*>(table_v)->CoverUpdateUnsafe(key,         \
                                                                  &v_context); \
  static_cast<KvVariable<Tindex, T>*>(table_vhat)                              \
      ->CoverUpdateUnsafe(key, &vhat_context);
          COMPUTE_AMSGrad(grad);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_AMSGrad
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_vhat->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_v->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyAMSGradOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyAMSGrad")  \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyAMSGradOp<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename T, typename Tindex>
class KvVariableSparseApplyAdadeltaOp : public OpKernel {
 public:
  explicit KvVariableSparseApplyAdadeltaOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks =
        MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum_grad = nullptr,
                        *table_accum_update = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum_grad));
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &table_accum_update));
    core::ScopedUnref unref_me_var(table_var);
    core::ScopedUnref unref_me_accum(table_accum_grad);
    core::ScopedUnref unref_me_accum_update(table_accum_update);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum_grad->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, table_accum_update->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& lr = ctx->input(3);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& rho = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    const Tensor& epsilon = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& grad = ctx->input(6);
    const Tensor& indices = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const int64_t N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    TensorShape var_shape, accum_grad_shape, accum_update_shape;
    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    accum_grad_shape = table_accum_grad->value_shape();
    accum_grad_shape.InsertDim(0, N);

    accum_update_shape = table_accum_update->value_shape();
    accum_update_shape.InsertDim(0, N);

    OP_REQUIRES(
        ctx, var_shape.IsSameSize(accum_grad_shape),
        errors::InvalidArgument("var and accum_grad do not have the same shape",
                                var_shape.DebugString(), " ",
                                accum_grad_shape.DebugString()));
    OP_REQUIRES(
        ctx, var_shape.IsSameSize(accum_update_shape),
        errors::InvalidArgument(
            "var and accum_update do not have the same shape",
            var_shape.DebugString(), " ", accum_update_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var_shape),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
    }

    if (N > 0) {
      auto DoWork = [this, ctx, &table_var, &table_accum_grad,
                     &table_accum_update, &indices, &grad, &lr, &rho,
                     &epsilon](int64_t start_i, int64_t limit_i) {
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();
        const int64_t embedding_dim_size = grad.dim_size(1);

        const T lr_scalar = lr.scalar<T>()();
        const T rho_scalar = rho.scalar<T>()();
        const T epsilon_scalar = epsilon.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; ++i) {
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> accum_grad_context;
          EVContext<T> accum_update_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_accum_grad)
              ->FindOrInsertUnsafe(key, &accum_grad_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum_update)
              ->FindOrInsertUnsafe(key, &accum_update_context, nullptr);
          auto v = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto accum_ =
              FlatVector<T>(accum_grad_context.Value(), embedding_dim_size);
          auto accum_update_ =
              FlatVector<T>(accum_update_context.Value(), embedding_dim_size);
          auto grad_ = grad_flat.template chip<0>(i);

          accum_ = accum_ * accum_.constant(rho_scalar) +
                   grad_.square() * grad_.constant(T(1) - rho_scalar);
          const auto update =
              (accum_update_ + accum_update_.constant(epsilon_scalar)).sqrt() *
              (accum_ + accum_.constant(epsilon_scalar)).rsqrt() * grad_;
          v -= update * update.constant(lr_scalar);
          accum_update_ =
              accum_update_ * accum_update_.constant(rho_scalar) +
              update.square() * update.constant(static_cast<T>(1) - rho_scalar);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum_grad->MarkAsDeltaListElements(ctx, indices,
                                                  train_deltalist);
        table_accum_update->MarkAsDeltaListElements(ctx, indices,
                                                    train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }
    VLOG(1) << "KvVariableSparseApplyAdadeltaOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("KvVariableSparseApplyAdadelta")      \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          KvVariableSparseApplyAdadeltaOp<T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyAdadeltaOp : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyAdadeltaOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum_grad = nullptr,
                        *table_accum_update = nullptr, *table_linear = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum_grad));
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &table_accum_update));
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 3), &table_linear));
    core::ScopedUnref unref_me_var(table_var);
    core::ScopedUnref unref_me_accum(table_accum_grad);
    core::ScopedUnref unref_me_accum_update(table_accum_update);
    core::ScopedUnref unref_me_linear(table_linear);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum_grad->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, table_accum_update->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(3)));

    const Tensor& lr = ctx->input(4);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& rho = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    const Tensor& epsilon = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& grad = ctx->input(7);
    const Tensor& indices = ctx->input(8);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    const Tensor& l1 = ctx->input(9);
    const Tensor& l2 = ctx->input(10);
    const Tensor& l21 = ctx->input(11);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    TensorShape var_shape, accum_grad_shape, accum_update_shape, linear_shape;
    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    accum_grad_shape = table_accum_grad->value_shape();
    accum_grad_shape.InsertDim(0, N);

    accum_update_shape = table_accum_update->value_shape();
    accum_update_shape.InsertDim(0, N);

    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);

    OP_REQUIRES(
        ctx, var_shape.IsSameSize(accum_grad_shape),
        errors::InvalidArgument("var and accum_grad do not have the same shape",
                                var_shape.DebugString(), " ",
                                accum_grad_shape.DebugString()));
    OP_REQUIRES(
        ctx, var_shape.IsSameSize(accum_update_shape),
        errors::InvalidArgument(
            "var and accum_update do not have the same shape",
            var_shape.DebugString(), " ", accum_update_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument(
                    "var and linear do not have the same shape",
                    var_shape.DebugString(), " ", linear_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var_shape),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_accum_grad,
                     &table_accum_update, &table_linear, &lr, &rho, &epsilon,
                     &grad, &indices, &l1, &l2,
                     &l21](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();

        T lr_scalar = lr.scalar<T>()();
        T rho_scalar = rho.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> accum_grad_context;
          EVContext<T> accum_update_context;
          EVContext<T> linear_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum_grad)
              ->FindOrInsertUnsafe(key, &accum_grad_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum_update)
              ->FindOrInsertUnsafe(key, &accum_update_context, nullptr);

          auto var = FlatVector<T>(var_context.Value(),
                                 embedding_dim_size);
          auto accum = FlatVector<T>(accum_grad_context.Value(),
                                 embedding_dim_size);
          auto accum_update = FlatVector<T>(accum_update_context.Value(),
                                 embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_ADADELTA(grad_to_use)                                          \
  auto new_accum = accum * rho_scalar +                                        \
                   (static_cast<T>(1) - rho_scalar) * grad_to_use.square();    \
  auto m = (accum_update + accum_update.constant(epsilon_scalar)).sqrt() *     \
           grad_to_use;                                                        \
  linear += m - (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;           \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y =                                                                   \
        (new_accum + new_accum.constant(epsilon_scalar)).sqrt() / lr_scalar +  \
        linear.constant(static_cast<T>(2) * l2_scalar);                        \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  accum = new_accum;                                                           \
  accum_update = accum_update * rho_scalar +                                   \
                 (static_cast<T>(1) - rho_scalar) * m.square() /               \
                     (new_accum + new_accum.constant(epsilon_scalar));         \
  static_cast<KvVariable<Tindex, T>*>(table_accum_grad)                        \
      ->CoverUpdateUnsafe(key, &accum_grad_context);                           \
  static_cast<KvVariable<Tindex, T>*>(table_accum_update)                      \
      ->CoverUpdateUnsafe(key, &accum_update_context);                         \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);
          COMPUTE_ADADELTA(grad);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_ADADELTA
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum_grad->MarkAsDeltaListElements(ctx, indices,
                                                  train_deltalist);
        table_accum_update->MarkAsDeltaListElements(ctx, indices,
                                                    train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyAdadeltaOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyAdadelta") \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyAdadeltaOp<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyMomentumOp : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyMomentumOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum = nullptr,
                        *table_linear = nullptr, *table_m = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum));
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 2), &table_linear));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 3), &table_m));
    core::ScopedUnref unref_me_var(table_var);
    core::ScopedUnref unref_me_accum(table_accum);
    core::ScopedUnref unref_me_linear(table_linear);
    core::ScopedUnref unref_me_m(table_m);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(3)));

    const Tensor& lr = ctx->input(4);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(5);
    const Tensor& indices = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    const Tensor& momentum = ctx->input(7);
    OP_REQUIRES(ctx, IsLegacyScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    const Tensor& l1 = ctx->input(8);
    const Tensor& l2 = ctx->input(9);
    const Tensor& l21 = ctx->input(10);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    TensorShape var_shape, accum_shape, linear_shape, m_shape;
    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    accum_shape = table_accum->value_shape();
    accum_shape.InsertDim(0, N);

    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);

    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument(
                    "var and linear do not have the same shape",
                    var_shape.DebugString(), " ", linear_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var_shape),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_accum,
                     &table_linear, &table_m, &lr, &grad, &indices, &momentum,
                     &l1, &l2, &l21](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);

        auto indices_flat = indices.flat<Tindex>();
        auto grad_flat = grad.flat_outer_dims<T>();

        T lr_scalar = lr.scalar<T>()();
        T momentum_scalar = momentum.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> linear_context;
          EVContext<T> m_context;
          EVContext<T> accum_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_m)
              ->FindOrInsertUnsafe(key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum)
              ->FindOrInsertUnsafe(key, &accum_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto accum = FlatVector<T>(accum_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_MOMENTUM(grad_to_use)                                          \
  m = m * m.constant(momentum_scalar) + grad_to_use;                           \
  auto new_m = m;                                                              \
  if (use_nesterov_) {                                                         \
    new_m = new_m * new_m.constant(momentum_scalar) + grad_to_use;             \
  }                                                                            \
  auto new_accum = accum.constant(static_cast<T>(1));                          \
  linear += new_m - (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;       \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y = new_accum.sqrt() / lr_scalar +                                    \
             linear.constant(static_cast<T>(2) * l2_scalar);                   \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  accum = new_accum;                                                           \
  static_cast<KvVariable<Tindex, T>*>(table_m)->CoverUpdateUnsafe(key,         \
                                                                  &m_context); \
  static_cast<KvVariable<Tindex, T>*>(table_accum)                             \
      ->CoverUpdateUnsafe(key, &accum_context);                                \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);
          COMPUTE_MOMENTUM(grad);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_MOMENTUM
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyMomentumOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyMomentum") \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyMomentumOp<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyAdaHessianOp : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyAdaHessianOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3, 4});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum = nullptr,
                        *table_linear = nullptr, *table_m = nullptr,
                        *table_v = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum));
    core::ScopedUnref unref_me_accum(table_accum);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 2), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 3), &table_m));
    core::ScopedUnref unref_me_m(table_m);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 4), &table_v));
    core::ScopedUnref unref_me_v(table_v);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(3)));
    OP_REQUIRES(
        ctx, table_v->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(4)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(5);
    const Tensor& indices = ctx->input(6);
    const Tensor& hessian = ctx->input(7);
    const Tensor& lr = ctx->input(8);
    const Tensor& beta1_power = ctx->input(9);
    const Tensor& beta2_power = ctx->input(10);
    const Tensor& beta1 = ctx->input(11);
    const Tensor& beta2 = ctx->input(12);
    const Tensor& epsilon = ctx->input(13);
    const Tensor& l1 = ctx->input(14);
    const Tensor& l2 = ctx->input(15);
    const Tensor& l21 = ctx->input(16);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, accum_shape, linear_shape, m_shape, v_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    accum_shape = table_accum->value_shape();
    accum_shape.InsertDim(0, N);

    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);

    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);

    v_shape = table_v->value_shape();
    v_shape.InsertDim(0, N);

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "kv_varaible and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument(
                    "kv_variable and linear do not have the same shape",
                    var_shape.DebugString(), " ", linear_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("kv_variable and m do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("kv_variable and v do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, hessian.dim_size(0) == N,
                errors::InvalidArgument("hessian must be the same size as "
                                        "indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_accum,
                     &table_linear, &table_m, &table_v, &indices, &grad,
                     &hessian, &lr, &beta1_power, &beta2_power, &beta1, &beta2,
                     &epsilon, &l1, &l2, &l21]
                     (int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto hessian_flat = hessian.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();

        T lr_scalar = lr.scalar<T>()();
        T beta1_power_scalar = beta1_power.scalar<T>()();
        T beta2_power_scalar = beta2_power.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> accum_context;
          EVContext<T> m_context;
          EVContext<T> v_context;
          EVContext<T> linear_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum)
              ->FindOrInsertUnsafe(key, &accum_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_m)
              ->FindOrInsertUnsafe(key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_v)
              ->FindOrInsertUnsafe(key, &v_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto accum = FlatVector<T>(accum_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(v_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);

          auto grad = grad_flat.template chip<0>(i);
          auto hessian = hessian_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_ADAHESSIAN(grad_to_use, hessian)                               \
  m = beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use;     \
  v = beta2_scalar * v +                                                       \
      (static_cast<T>(1) - beta2_scalar) * hessian.square();                   \
  auto new_accum = v / (static_cast<T>(1) - beta2_power_scalar);               \
  linear += m / (static_cast<T>(1) - beta1_power_scalar) -                     \
            (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;               \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y =                                                                   \
        (new_accum.sqrt() + new_accum.constant(epsilon_scalar)) / lr_scalar +  \
        linear.constant(static_cast<T>(2) * l2_scalar);                        \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  accum = new_accum;                                                           \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);                               \
  static_cast<KvVariable<Tindex, T>*>(table_m)->CoverUpdateUnsafe(key,         \
                                                                  &m_context); \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);                               \
  static_cast<KvVariable<Tindex, T>*>(table_v)->CoverUpdateUnsafe(key,         \
                                                                  &v_context); \
  static_cast<KvVariable<Tindex, T>*>(table_accum)                             \
      ->CoverUpdateUnsafe(key, &accum_context);
          COMPUTE_ADAHESSIAN(grad, hessian);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_ADAHESSIAN
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_v->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyAdaHessianOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)              \
  REGISTER_KERNEL_BUILDER(                         \
      Name("KvVariableGroupSparseApplyAdaHessian") \
          .Device(DEVICE_CPU)                      \
          .TypeConstraint<T>("T")                  \
          .TypeConstraint<Tindices>("Tindices"),   \
      KvVariableGroupSparseApplyAdaHessianOp<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
struct ApplyAdaHessian {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstFlat hessian,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon) {
    const T alpha = lr() * Eigen::numext::sqrt(T(1) - beta2_power()) /
                    (T(1) - beta1_power());

    m.device(d) += (grad - m) * (T(1) - beta1());
    v.device(d) += (hessian.square() - v) * (T(1) - beta2());
    var.device(d) -= (m * alpha) / (v.sqrt() + epsilon());
  }
};

template <typename Device, typename T>
class ApplyAdaHessianOp : public OpKernel {
 public:
  explicit ApplyAdaHessianOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();

    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2}, false);

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& hessian = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1_power = ctx->input(6);
    const Tensor& beta2_power = ctx->input(7);
    const Tensor& beta1 = ctx->input(8);
    const Tensor& beta2 = ctx->input(9);
    const Tensor& epsilon = ctx->input(10);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not "
                                        "have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not "
                                        "have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(hessian.shape()),
        errors::InvalidArgument("var and hessian do not have the same shape",
                                var.shape().DebugString(), " ",
                                hessian.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    ApplyAdaHessian<Device, T>()(device, var.flat<T>(), m.flat<T>(),
                                 v.flat<T>(), grad.flat<T>(), hessian.flat<T>(),
                                 lr.scalar<T>(), beta1_power.scalar<T>(),
                                 beta2_power.scalar<T>(), beta1.scalar<T>(),
                                 beta2.scalar<T>(), epsilon.scalar<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);

    VLOG(1) << "ApplyAdaHessianOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("ApplyAdaHessian").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdaHessianOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdaHessian")                \
                              .HostMemory("var")                         \
                              .HostMemory("m")                           \
                              .HostMemory("v")                           \
                              .Device(DEVICE_##D)                        \
                              .TypeConstraint<T>("T"),                   \
                          ApplyAdaHessianOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyAdaBeliefOp : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyAdaBeliefOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3, 4});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum = nullptr,
                        *table_linear = nullptr, *table_m = nullptr,
                        *table_v = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum));
    core::ScopedUnref unref_me_accum(table_accum);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 2), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 3), &table_m));
    core::ScopedUnref unref_me_m(table_m);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 4), &table_v));
    core::ScopedUnref unref_me_v(table_v);
    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(3)));
    OP_REQUIRES(
        ctx, table_v->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(4)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(5);
    const Tensor& indices = ctx->input(6);
    const Tensor& lr = ctx->input(7);
    const Tensor& beta1_power = ctx->input(8);
    const Tensor& beta2_power = ctx->input(9);
    const Tensor& beta1 = ctx->input(10);
    const Tensor& beta2 = ctx->input(11);
    const Tensor& epsilon = ctx->input(12);
    const Tensor& l1 = ctx->input(13);
    const Tensor& l2 = ctx->input(14);
    const Tensor& l21 = ctx->input(15);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, accum_shape, linear_shape, m_shape, v_shape;
    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);
    accum_shape = table_accum->value_shape();
    accum_shape.InsertDim(0, N);
    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);
    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);
    v_shape = table_v->value_shape();
    v_shape.InsertDim(0, N);
    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "kv_varaible and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument(
                    "kv_variable and linear do not have the same shape",
                    var_shape.DebugString(), " ", linear_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("kv_variable and m do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("kv_variable and v do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));
    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));
    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_accum,
                     &table_linear, &table_m, &table_v, &indices, &grad, &lr,
                     &beta1_power, &beta2_power, &beta1, &beta2, &epsilon, &l1,
                     &l2, &l21](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();
        T lr_scalar = lr.scalar<T>()();
        T beta1_power_scalar = beta1_power.scalar<T>()();
        T beta2_power_scalar = beta2_power.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> accum_context;
          EVContext<T> m_context;
          EVContext<T> v_context;
          EVContext<T> linear_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum)
              ->FindOrInsertUnsafe(key, &accum_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_m)->FindOrInsertUnsafe(
              key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_v)->FindOrInsertUnsafe(
              key, &v_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto accum = FlatVector<T>(accum_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(v_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_ADABELIEF(grad_to_use)                                         \
  m = beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use;     \
  v = beta2_scalar * v +                                                       \
      (static_cast<T>(1) - beta2_scalar) * (grad_to_use - m).square();         \
  auto new_accum = v / (static_cast<T>(1) - beta2_power_scalar);               \
  linear += m / (static_cast<T>(1) - beta1_power_scalar) -                     \
            (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;               \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y =                                                                   \
        (new_accum.sqrt() + new_accum.constant(epsilon_scalar)) / lr_scalar +  \
        linear.constant(static_cast<T>(2) * l2_scalar);                        \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  accum = new_accum;                                                           \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);                               \
  static_cast<KvVariable<Tindex, T>*>(table_accum)                             \
      ->CoverUpdateUnsafe(key, &accum_context);                                \
  static_cast<KvVariable<Tindex, T>*>(table_m)->CoverUpdateUnsafe(key,         \
                                                                  &m_context); \
  static_cast<KvVariable<Tindex, T>*>(table_v)->CoverUpdateUnsafe(key,         \
                                                                  &v_context);
          COMPUTE_ADABELIEF(grad);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_ADABELIEF
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_v->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyAdaBeliefOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyAdaBelief")\
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyAdaBeliefOp<CPUDevice, T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
struct ApplyAdaBelief {
  void operator()(const Device& d,
                  typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon) {
    const T adjust = Eigen::numext::sqrt(T(1) - beta2_power()) /
                    (T(1) - beta1_power());
    m.device(d) += (grad - m) * (T(1) - beta1());
    v.device(d) += ((grad - m).square() - v) * (T(1) - beta2());
    var.device(d) -= (m * lr() * adjust) / (v.sqrt() + epsilon());
  }
};

template <typename Device, typename T>
class ApplyAdaBeliefOp : public OpKernel {
 public:
  explicit ApplyAdaBeliefOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }
  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder(
      ctx, use_exclusive_lock_, {0, 1, 2}, sparse);
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));
    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& beta1_power = ctx->input(5);
    const Tensor& beta2_power = ctx->input(6);
    const Tensor& beta1 = ctx->input(7);
    const Tensor& beta2 = ctx->input(8);
    const Tensor& epsilon = ctx->input(9);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not "
                                        "have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not "
                                        "have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Device& device = ctx->template eigen_device<Device>();
    ApplyAdaBelief<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), v.flat<T>(), grad.flat<T>(),
        lr.scalar<T>(), beta1_power.scalar<T>(), beta2_power.scalar<T>(),
        beta1.scalar<T>(), beta2.scalar<T>(), epsilon.scalar<T>());
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
    VLOG(1) << "ApplyAdaBeliefOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)             \
  REGISTER_KERNEL_BUILDER(                 \
      Name("ApplyAdaBelief")               \
          .Device(DEVICE_##D)              \
          .TypeConstraint<T>("T"),         \
      ApplyAdaBeliefOp<D##Device, T>);     \
  REGISTER_KERNEL_BUILDER(                 \
      Name("ResourceApplyAdaBelief")       \
          .HostMemory("var")               \
          .HostMemory("m")                 \
          .HostMemory("v")                 \
          .Device(DEVICE_##D)              \
          .TypeConstraint<T>("T"),         \
      ApplyAdaBeliefOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);
TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyLambOp : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyLambOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3, 4});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum = nullptr,
                        *table_linear = nullptr, *table_m = nullptr,
                        *table_v = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum));
    core::ScopedUnref unref_me_accum(table_accum);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 2), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 3), &table_m));
    core::ScopedUnref unref_me_m(table_m);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 4), &table_v));
    core::ScopedUnref unref_me_v(table_v);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(3)));
    OP_REQUIRES(
        ctx, table_v->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(4)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(5);
    const Tensor& indices = ctx->input(6);
    const Tensor& lr = ctx->input(7);
    const Tensor& beta1_power = ctx->input(8);
    const Tensor& beta2_power = ctx->input(9);
    const Tensor& beta1 = ctx->input(10);
    const Tensor& beta2 = ctx->input(11);
    const Tensor& epsilon = ctx->input(12);
    const Tensor& l1 = ctx->input(13);
    const Tensor& l2 = ctx->input(14);
    const Tensor& l21 = ctx->input(15);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, accum_shape, linear_shape, m_shape, v_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    accum_shape = table_accum->value_shape();
    accum_shape.InsertDim(0, N);

    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);

    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);

    v_shape = table_v->value_shape();
    v_shape.InsertDim(0, N);

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "kv_varaible and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument(
                    "kv_variable and linear do not have the same shape",
                    var_shape.DebugString(), " ", linear_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("kv_variable and m do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("kv_variable and v do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_accum,
                     &table_linear, &table_m, &table_v, &indices, &grad, &lr,
                     &beta1_power, &beta2_power, &beta1, &beta2, &epsilon, &l1,
                     &l2, &l21](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();

        T lr_scalar = lr.scalar<T>()();
        T beta1_power_scalar = beta1_power.scalar<T>()();
        T beta2_power_scalar = beta2_power.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));

          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> accum_context;
          EVContext<T> m_context;
          EVContext<T> v_context;
          EVContext<T> linear_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum)
              ->FindOrInsertUnsafe(key, &accum_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_m)
              ->FindOrInsertUnsafe(key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_v)
              ->FindOrInsertUnsafe(key, &v_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto accum = FlatVector<T>(accum_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(v_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_LAMB(grad_to_use)                                              \
  m = beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use;     \
  v = beta2_scalar * v +                                                       \
      (static_cast<T>(1) - beta2_scalar) * grad_to_use.square();               \
  auto new_m = m / (static_cast<T>(1) - beta1_power_scalar);                   \
  auto new_accum = v / (static_cast<T>(1) - beta2_power_scalar);               \
  auto r = new_m / (new_accum.sqrt() + new_accum.constant(epsilon_scalar));    \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar r_norm_t = r.square().sum().sqrt();                             \
  TensorScalar var_norm_t = var.square().sum().sqrt();                         \
  T r_norm = static_cast<T>(r_norm_t(0));                                      \
  T var_norm = static_cast<T>(var_norm_t(0));                                  \
  T ratio = static_cast<T>(1);                                                 \
  if (r_norm > static_cast<T>(0) && var_norm > static_cast<T>(0)) {            \
    ratio = var_norm / (r_norm + static_cast<T>(1e-8));                        \
  }                                                                            \
  linear += new_m * new_m.constant(ratio) -                                    \
            (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;               \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y =                                                                   \
        (new_accum.sqrt() + new_accum.constant(epsilon_scalar)) / lr_scalar +  \
        linear.constant(static_cast<T>(2) * l2_scalar);                        \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  accum = new_accum;                                                           \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);                               \
  static_cast<KvVariable<Tindex, T>*>(table_accum)                             \
      ->CoverUpdateUnsafe(key, &accum_context);                                \
  static_cast<KvVariable<Tindex, T>*>(table_m)->CoverUpdateUnsafe(key,         \
                                                                  &m_context); \
  static_cast<KvVariable<Tindex, T>*>(table_v)->CoverUpdateUnsafe(key,         \
                                                                  &v_context);
          COMPUTE_LAMB(grad);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_LAMB
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_v->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyLambOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyLamb")     \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyLambOp<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

// TF_CALL_half(REGISTER_CPU_KERNELS);
// TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
// TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
struct ApplyLamb {
  void operator()(const Device& d,
                  typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon) {
    const T adjust = Eigen::numext::sqrt(T(1) - beta2_power()) /
                    (T(1) - beta1_power());

    m.device(d) += (grad - m) * (T(1) - beta1());
    v.device(d) += (grad.square() - v) * (T(1) - beta2());

    auto r = (m * adjust) / (v.sqrt() + epsilon());
    using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;
    TensorScalar r_norm_t = r.square().sum().sqrt();
    TensorScalar var_norm_t = var.square().sum().sqrt();
    T r_norm = T(r_norm_t(0));
    T var_norm = T(var_norm_t(0));
    T ratio = T(1);
    if (r_norm > T(0) && var_norm > T(0)) {
      ratio = var_norm / (r_norm + T(1e-8));
    }
    var.device(d) -= (m * lr() * adjust * ratio) / (v.sqrt() + epsilon());
  }
};

template <typename Device, typename T>
class ApplyLambOp : public OpKernel {
 public:
  explicit ApplyLambOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();

    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder(
      ctx, use_exclusive_lock_, {0, 1, 2}, sparse);

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& beta1_power = ctx->input(5);
    const Tensor& beta2_power = ctx->input(6);
    const Tensor& beta1 = ctx->input(7);
    const Tensor& beta2 = ctx->input(8);
    const Tensor& epsilon = ctx->input(9);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not "
                                        "have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not "
                                        "have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    ApplyLamb<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), v.flat<T>(), grad.flat<T>(),
        lr.scalar<T>(), beta1_power.scalar<T>(), beta2_power.scalar<T>(),
        beta1.scalar<T>(), beta2.scalar<T>(), epsilon.scalar<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);

    VLOG(1) << "ApplyLambOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)             \
  REGISTER_KERNEL_BUILDER(                 \
      Name("ApplyLamb")              \
          .Device(DEVICE_##D)              \
          .TypeConstraint<T>("T"),         \
      ApplyLambOp<D##Device, T>);    \
  REGISTER_KERNEL_BUILDER(                 \
      Name("ResourceApplyLamb")      \
          .HostMemory("var")               \
          .HostMemory("m")                 \
          .HostMemory("v")                 \
          .Device(DEVICE_##D)              \
          .TypeConstraint<T>("T"),         \
      ApplyLambOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyLambHessianOp : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyLambHessianOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3, 4});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_accum = nullptr,
                        *table_linear = nullptr, *table_m = nullptr,
                        *table_v = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_accum));
    core::ScopedUnref unref_me_accum(table_accum);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 2), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 3), &table_m));
    core::ScopedUnref unref_me_m(table_m);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 4), &table_v));
    core::ScopedUnref unref_me_v(table_v);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_accum->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(3)));
    OP_REQUIRES(
        ctx, table_v->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(4)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(5);
    const Tensor& indices = ctx->input(6);
    const Tensor& hessian = ctx->input(7);
    const Tensor& lr = ctx->input(8);
    const Tensor& beta1_power = ctx->input(9);
    const Tensor& beta2_power = ctx->input(10);
    const Tensor& beta1 = ctx->input(11);
    const Tensor& beta2 = ctx->input(12);
    const Tensor& epsilon = ctx->input(13);
    const Tensor& l1 = ctx->input(14);
    const Tensor& l2 = ctx->input(15);
    const Tensor& l21 = ctx->input(16);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, accum_shape, linear_shape, m_shape, v_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    accum_shape = table_accum->value_shape();
    accum_shape.InsertDim(0, N);

    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);

    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);

    v_shape = table_v->value_shape();
    v_shape.InsertDim(0, N);

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "kv_varaible and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument(
                    "kv_variable and linear do not have the same shape",
                    var_shape.DebugString(), " ", linear_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("kv_variable and m do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("kv_variable and v do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(
        ctx, hessian.dim_size(0) == N,
        errors::InvalidArgument(
          "hessian must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_accum,
                     &table_linear, &table_m, &table_v, &indices, &grad,
                     &hessian, &lr, &beta1_power, &beta2_power, &beta1, &beta2,
                     &epsilon, &l1, &l2, &l21]
                     (int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto hessian_flat = hessian.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();

        T lr_scalar = lr.scalar<T>()();
        T beta1_power_scalar = beta1_power.scalar<T>()();
        T beta2_power_scalar = beta2_power.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));

          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> accum_context;
          EVContext<T> m_context;
          EVContext<T> v_context;
          EVContext<T> linear_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_accum)
              ->FindOrInsertUnsafe(key, &accum_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_m)
              ->FindOrInsertUnsafe(key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_v)
              ->FindOrInsertUnsafe(key, &v_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto accum = FlatVector<T>(accum_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(v_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);
          auto hessian = hessian_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_LAMBHESSIAN(grad_to_use, hessian)                              \
  m = beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use;     \
  v = beta2_scalar * v +                                                       \
      (static_cast<T>(1) - beta2_scalar) * hessian.square();                   \
  auto new_m = m / (static_cast<T>(1) - beta1_power_scalar);                   \
  auto new_accum = v / (static_cast<T>(1) - beta2_power_scalar);               \
  auto r = new_m / (new_accum.sqrt() + new_accum.constant(epsilon_scalar));    \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar r_norm_t = r.square().sum().sqrt();                             \
  TensorScalar var_norm_t = var.square().sum().sqrt();                         \
  T r_norm = static_cast<T>(r_norm_t(0));                                      \
  T var_norm = static_cast<T>(var_norm_t(0));                                  \
  T ratio = static_cast<T>(1);                                                 \
  if (r_norm > static_cast<T>(0) && var_norm > static_cast<T>(0)) {            \
    ratio = var_norm / (r_norm + static_cast<T>(1e-8));                        \
  }                                                                            \
  linear += new_m * new_m.constant(ratio) -                                    \
            (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;               \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y =                                                                   \
        (new_accum.sqrt() + new_accum.constant(epsilon_scalar)) / lr_scalar +  \
        linear.constant(static_cast<T>(2) * l2_scalar);                        \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  accum = new_accum;                                                           \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);                               \
  static_cast<KvVariable<Tindex, T>*>(table_m)->CoverUpdateUnsafe(key,         \
                                                                  &m_context); \
  static_cast<KvVariable<Tindex, T>*>(table_v)->CoverUpdateUnsafe(key,         \
                                                                  &v_context); \
  static_cast<KvVariable<Tindex, T>*>(table_accum)                             \
      ->CoverUpdateUnsafe(key, &accum_context);
          COMPUTE_LAMBHESSIAN(grad, hessian);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_LAMBHESSIAN
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_accum->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_v->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyLambHessianOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyLambHessian")     \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyLambHessianOp<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

// TF_CALL_half(REGISTER_CPU_KERNELS);
// TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
// TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
struct ApplyLambHessian {
  void operator()(const Device& d,
                  typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstFlat hessian,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon) {
    const T adjust = Eigen::numext::sqrt(T(1) - beta2_power()) /
                    (T(1) - beta1_power());

    m.device(d) += (grad - m) * (T(1) - beta1());
    v.device(d) += (hessian.square() - v) * (T(1) - beta2());

    auto r = (m * adjust) / (v.sqrt() + epsilon());
    using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;
    TensorScalar r_norm_t = r.square().sum().sqrt();
    TensorScalar var_norm_t = var.square().sum().sqrt();
    T r_norm = T(r_norm_t(0));
    T var_norm = T(var_norm_t(0));
    T ratio = T(1);
    if (r_norm > T(0) && var_norm > T(0)) {
      ratio = var_norm / (r_norm + T(1e-8));
    }
    var.device(d) -= (m * lr() * adjust * ratio) / (v.sqrt() + epsilon());
  }
};

template <typename Device, typename T>
class ApplyLambHessianOp : public OpKernel {
 public:
  explicit ApplyLambHessianOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();

    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder(
      ctx, use_exclusive_lock_, {0, 1, 2}, sparse);

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& hessian = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1_power = ctx->input(6);
    const Tensor& beta2_power = ctx->input(7);
    const Tensor& beta1 = ctx->input(8);
    const Tensor& beta2 = ctx->input(9);
    const Tensor& epsilon = ctx->input(10);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not "
                                        "have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not "
                                        "have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(hessian.shape()),
        errors::InvalidArgument("var and hessian do not have the same shape",
                                var.shape().DebugString(), " ",
                                hessian.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    ApplyLambHessian<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), v.flat<T>(), grad.flat<T>(),
        hessian.flat<T>(), lr.scalar<T>(), beta1_power.scalar<T>(),
        beta2_power.scalar<T>(), beta1.scalar<T>(), beta2.scalar<T>(),
        epsilon.scalar<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);

    VLOG(1) << "ApplyLambHessianOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)             \
  REGISTER_KERNEL_BUILDER(                 \
      Name("ApplyLambHessian")              \
          .Device(DEVICE_##D)              \
          .TypeConstraint<T>("T"),         \
      ApplyLambHessianOp<D##Device, T>);    \
  REGISTER_KERNEL_BUILDER(                 \
      Name("ResourceApplyLambHessian")      \
          .HostMemory("var")               \
          .HostMemory("m")                 \
          .HostMemory("v")                 \
          .Device(DEVICE_##D)              \
          .TypeConstraint<T>("T"),         \
      ApplyLambHessianOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
struct ApplyAdaDQH {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon) {
    const T alpha = lr() * Eigen::numext::sqrt(T(1) - beta2_power()) /
                    (T(1) - beta1_power());

    T beta;
    if (beta1() > beta1_power()) {
      beta = T(1) - beta1_power() / beta1();
    } else {
      beta = T(1);
    }
    auto m_old = m / beta;
    auto m_new = grad * (T(1) - beta1()) + m * beta1();
    auto h = m_new / (T(1) - beta1_power()) - m_old;
    v.device(d) += (h.square() - v) * (T(1) - beta2());
    var.device(d) -= m_new * alpha / v.sqrt().cwiseMax(epsilon() *
                                    Eigen::numext::sqrt(T(1) - beta2_power()));
    m.device(d) += (grad - m) * (T(1) - beta1());
  }
};

template <typename Device, typename T>
class ApplyAdaDQHOp: public OpKernel {
 public:
  explicit ApplyAdaDQHOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();

    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2}, false);

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));

    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& beta1_power = ctx->input(5);
    const Tensor& beta2_power = ctx->input(6);
    const Tensor& beta1 = ctx->input(7);
    const Tensor& beta2 = ctx->input(8);
    const Tensor& epsilon = ctx->input(9);

    const Device& device = ctx->template eigen_device<Device>();
    ApplyAdaDQH<Device, T>()(device, var.flat<T>(), m.flat<T>(),
                            v.flat<T>(), grad.flat<T>(),
                            lr.scalar<T>(), beta1_power.scalar<T>(),
                            beta2_power.scalar<T>(), beta1.scalar<T>(),
                            beta2.scalar<T>(), epsilon.scalar<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);

    VLOG(1) << "ApplyAdaDQHOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                      \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ApplyAdaDQH").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdaDQHOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdaDQH")                \
                              .HostMemory("var")                    \
                              .HostMemory("m")                      \
                              .HostMemory("v")                      \
                              .Device(DEVICE_##D)                   \
                              .TypeConstraint<T>("T"),              \
                          ApplyAdaDQHOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableSparseApplyAdaDQHOp : public OpKernel {
 public:
  explicit KvVariableSparseApplyAdaDQHOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_m = nullptr,
                        *table_v = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_m));
    core::ScopedUnref unref_me_m(table_m);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &table_v));
    core::ScopedUnref unref_me_v(table_v);
    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_v->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1_power = ctx->input(6);
    const Tensor& beta2_power = ctx->input(7);
    const Tensor& beta1 = ctx->input(8);
    const Tensor& beta2 = ctx->input(9);
    const Tensor& epsilon = ctx->input(10);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, m_shape, v_shape;
    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);
    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);
    v_shape = table_v->value_shape();
    v_shape.InsertDim(0, N);
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("kv_variable and m do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("kv_variable and v do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));
    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_m,
                     &table_v, &indices, &grad, &lr,
                     &beta1_power, &beta2_power, &beta1, &beta2,
                     &epsilon](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();
        T lr_scalar = lr.scalar<T>()();
        T beta1_power_scalar = beta1_power.scalar<T>()();
        T beta2_power_scalar = beta2_power.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> m_context;
          EVContext<T> v_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_m)
              ->FindOrInsertUnsafe(key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_v)
              ->FindOrInsertUnsafe(key, &v_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(v_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_ADADQH(grad_to_use)                                            \
  const T alpha = lr_scalar * Eigen::numext::sqrt(static_cast<T>(1) -          \
    beta2_power_scalar) / (static_cast<T>(1) - beta1_power_scalar);            \
  T beta;                                                                      \
  if (beta1_scalar > beta1_power_scalar) {                                     \
    beta = static_cast<T>(1) - beta1_power_scalar / beta1_scalar;              \
  } else {                                                                     \
    beta = static_cast<T>(1);                                                  \
  }                                                                            \
  auto m_old = m / beta;                                                       \
  auto m_new = beta1_scalar * m +                                              \
               (static_cast<T>(1) - beta1_scalar) * grad_to_use;               \
  auto h = m_new / (static_cast<T>(1) - beta1_power_scalar) - m_old;           \
  v = beta2_scalar * v +                                                       \
      (static_cast<T>(1) - beta2_scalar) * h.square();                         \
  var -= m_new * alpha / v.sqrt().cwiseMax(epsilon_scalar *                    \
              Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar));    \
  m = beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use;
          COMPUTE_ADADQH(grad);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_ADADQH
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_v->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableSparseApplyAdaDQHOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableSparseApplyAdaDQH")\
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableSparseApplyAdaDQHOp<CPUDevice, T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class SparseApplyAdaDQHOp : public OpKernel {
 public:
  explicit SparseApplyAdaDQHOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();

    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2}, false);

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));

    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(m.shape()),
        errors::InvalidArgument("var and m do not have the same shape",
                                var.shape().DebugString(), " ",
                                m.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(v.shape()),
        errors::InvalidArgument("var and v do not have the same shape",
                                var.shape().DebugString(), " ",
                                v.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));


    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1_power = ctx->input(6);
    const Tensor& beta2_power = ctx->input(7);
    const Tensor& beta1 = ctx->input(8);
    const Tensor& beta2 = ctx->input(9);
    const Tensor& epsilon = ctx->input(10);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      auto indices_vec = indices.vec<Tindex>();
      auto var_flat = var.flat_outer_dims<T>();
      auto m_flat = m.flat_outer_dims<T>();
      auto v_flat = v.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      const T lr_scalar = lr.scalar<T>()();
      const T beta1_power_scalar = beta1_power.scalar<T>()();
      const T beta2_power_scalar = beta2_power.scalar<T>()();
      const T beta1_scalar = beta1.scalar<T>()();
      const T beta2_scalar = beta2.scalar<T>()();
      const T epsilon_scalar = epsilon.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        // Validate all the indices are in range
        const Tindex index = internal::SubtleMustCopy(indices_vec(i));
        OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                   errors::InvalidArgument(
                     strings::StrCat("Index ", index, " at offset ", i,
                                     " in indices is out of range")));

        auto var_ = var_flat.template chip<0>(index);
        auto m_ = m_flat.template chip<0>(index);
        auto v_ = v_flat.template chip<0>(index);
        auto grad_ = grad_flat.template chip<0>(i);

        const T alpha = lr_scalar * Eigen::numext::sqrt(T(1) - \
          beta2_power_scalar) / (T(1) - beta1_power_scalar);
        T beta;
        if (beta1_scalar > beta1_power_scalar) {
          beta = T(1) - beta1_power_scalar / beta1_scalar;
        } else {
          beta = T(1);
        }
        auto m_old = m_ / beta;
        auto m_new = beta1_scalar * m_ + (T(1) - beta1_scalar) * grad_;
        auto h = m_new / (T(1) - beta1_power_scalar) - m_old;
        v_ = beta2_scalar * v_ + (T(1) - beta2_scalar) * h.square();
        var_ -= m_new * alpha / v_.sqrt().cwiseMax(epsilon_scalar);
        m_ = beta1_scalar * m_ + (T(1) - beta1_scalar) * grad_;
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);

    VLOG(1) << "SparseApplyAdaDQHOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)             \
  REGISTER_KERNEL_BUILDER(                        \
      Name("SparseApplyAdaDQH")                    \
          .Device(DEVICE_CPU)                     \
          .TypeConstraint<T>("T")                 \
          .TypeConstraint<Tindices>("Tindices"),  \
      SparseApplyAdaDQHOp<CPUDevice, T, Tindices>);\
  REGISTER_KERNEL_BUILDER(                        \
      Name("ResourceSparseApplyAdaDQH")            \
          .Device(DEVICE_CPU)                     \
          .TypeConstraint<T>("T")                 \
          .TypeConstraint<Tindices>("Tindices"),  \
      SparseApplyAdaDQHOp<CPUDevice, T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyAdaDQHOp : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyAdaDQHOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_m = nullptr,
                        *table_v = nullptr, *table_linear = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_m));
    core::ScopedUnref unref_me_m(table_m);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &table_v));
    core::ScopedUnref unref_me_v(table_v);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 3), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);
    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_v->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(3)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(4);
    const Tensor& indices = ctx->input(5);
    const Tensor& lr = ctx->input(6);
    const Tensor& beta1_power = ctx->input(7);
    const Tensor& beta2_power = ctx->input(8);
    const Tensor& beta1 = ctx->input(9);
    const Tensor& beta2 = ctx->input(10);
    const Tensor& epsilon = ctx->input(11);
    const Tensor& l1 = ctx->input(12);
    const Tensor& l2 = ctx->input(13);
    const Tensor& l21 = ctx->input(14);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, m_shape, v_shape, linear_shape;
    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);
    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);
    v_shape = table_v->value_shape();
    v_shape.InsertDim(0, N);
    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("kv_variable and m do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("kv_variable and v do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument("kv_variable and linear do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        linear_shape.DebugString()));
    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_m, &table_v,
                     &table_linear, &indices, &grad, &lr, &beta1_power,
                     &beta2_power, &beta1, &beta2, &epsilon, &l1, &l2,
                     &l21](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();
        T lr_scalar = lr.scalar<T>()();
        T beta1_power_scalar = beta1_power.scalar<T>()();
        T beta2_power_scalar = beta2_power.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> linear_context;
          EVContext<T> m_context;
          EVContext<T> v_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_m)
              ->FindOrInsertUnsafe(key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_v)
              ->FindOrInsertUnsafe(key, &v_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(v_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_ADADQH(grad_to_use)                                            \
  const T alpha =                                                              \
      Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar) /            \
      (static_cast<T>(1) - beta1_power_scalar);                                \
  T beta, gamma;                                                               \
  const T epsilon_adjust =                                                     \
      epsilon_scalar *                                                         \
      Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar);             \
  if (beta1_scalar > beta1_power_scalar) {                                     \
    beta = static_cast<T>(1) - beta1_power_scalar / beta1_scalar;              \
    gamma = epsilon_adjust;                                                    \
  } else {                                                                     \
    beta = static_cast<T>(1);                                                  \
    gamma = static_cast<T>(0);                                                 \
  }                                                                            \
  auto m_old = m / beta;                                                       \
  auto m_new =                                                                 \
      beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use;     \
  auto h = m_new / (static_cast<T>(1) - beta1_power_scalar) - m_old;           \
  auto v_new =                                                                 \
      beta2_scalar * v + (static_cast<T>(1) - beta2_scalar) * h.square();      \
  linear += m_new * alpha - (v_new.sqrt().cwiseMax(epsilon_adjust) -           \
                             v.sqrt().cwiseMax(gamma)) /                       \
                                lr_scalar * var;                               \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y = v_new.sqrt().cwiseMax(epsilon_adjust) / lr_scalar +               \
             linear.constant(static_cast<T>(2) * l2_scalar);                   \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  v = v_new;                                                                   \
  m = m_new;                                                                   \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);                               \
  static_cast<KvVariable<Tindex, T>*>(table_m)->CoverUpdateUnsafe(key,         \
                                                                  &m_context); \
  static_cast<KvVariable<Tindex, T>*>(table_v)->CoverUpdateUnsafe(key,         \
                                                                  &v_context);
          COMPUTE_ADADQH(grad);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_ADADQH
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_v->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyAdaDQHOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyAdaDQH")   \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyAdaDQHOp<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyAdaDQHV2Op : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyAdaDQHV2Op(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_m = nullptr,
                        *table_v = nullptr, *table_linear = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_m));
    core::ScopedUnref unref_me_m(table_m);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &table_v));
    core::ScopedUnref unref_me_v(table_v);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 3), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);
    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_v->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(3)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(4);
    const Tensor& indices = ctx->input(5);
    const Tensor& lr = ctx->input(6);
    const Tensor& beta1_power = ctx->input(7);
    const Tensor& beta2_power = ctx->input(8);
    const Tensor& beta1 = ctx->input(9);
    const Tensor& beta2 = ctx->input(10);
    const Tensor& epsilon = ctx->input(11);
    const Tensor& l1 = ctx->input(12);
    const Tensor& l2 = ctx->input(13);
    const Tensor& l21 = ctx->input(14);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, m_shape, v_shape, linear_shape;
    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);
    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);
    v_shape = table_v->value_shape();
    v_shape.InsertDim(0, N);
    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("kv_variable and m do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("kv_variable and v do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument("kv_variable and linear do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        linear_shape.DebugString()));
    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const int64_t first_dim_size = indices.dim_size(0);
    const int64_t embedding_dim_size = grad.dim_size(1);
    T lr_scalar = lr.scalar<T>()();
    T beta1_power_scalar = beta1_power.scalar<T>()();
    T beta2_power_scalar = beta2_power.scalar<T>()();
    T beta1_scalar = beta1.scalar<T>()();
    T beta2_scalar = beta2.scalar<T>()();
    T epsilon_scalar = epsilon.scalar<T>()();
    T l1_scalar = l1.scalar<T>()() * lr_scalar;
    T l2_scalar = l2.scalar<T>()() * lr_scalar;
    T l21_scalar = l21.scalar<T>()() * lr_scalar;
    const T alpha =
      lr_scalar * Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar) /
      (static_cast<T>(1) - beta1_power_scalar);
    const T epsilon_adjust = epsilon_scalar *
      Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar);
    const T last_epsilon_adjust = epsilon_scalar *
      Eigen::numext::sqrt(static_cast<T>(1)
      - beta2_power_scalar / beta2_scalar);
    auto l21_norm = l21_scalar *
      Eigen::numext::sqrt(static_cast<T>(inner_dim));

    if (N > 0) {
      auto DoWork = [this, ctx, &table_var, &table_m, &table_v, &table_linear,
                     &indices, &grad, &beta1_power_scalar,
                     &beta2_power_scalar, &beta1_scalar, &beta2_scalar,
                     &epsilon_adjust, &last_epsilon_adjust, &l1_scalar,
                     &l2_scalar, &l21_norm, &alpha, first_dim_size,
                     embedding_dim_size](int64_t start_i, int64_t limit_i) {
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> linear_context;
          EVContext<T> m_context;
          EVContext<T> v_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_m)
              ->FindOrInsertUnsafe(key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_v)
              ->FindOrInsertUnsafe(key, &v_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(v_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_ADADQH(grad_to_use)                                            \
  T beta;                                                                      \
  if (beta1_scalar > beta1_power_scalar) {                                     \
    beta = static_cast<T>(1) - beta1_power_scalar / beta1_scalar;              \
  } else {                                                                     \
    beta = static_cast<T>(1);                                                  \
  }                                                                            \
  auto m_old = m / beta;                                                       \
  auto m_new =                                                                 \
      beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use;     \
  auto h = m_new / (static_cast<T>(1) - beta1_power_scalar) - m_old;           \
  auto v_new =                                                                 \
      beta2_scalar * v + (static_cast<T>(1) - beta2_scalar) * h.square();      \
  auto accum_new = v_new.sqrt().cwiseMax(epsilon_adjust);                      \
  auto accum = v.sqrt().cwiseMax(last_epsilon_adjust);                         \
  linear += m_new * alpha - (accum_new - accum) * var;                         \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y = v_new.sqrt().cwiseMax(epsilon_adjust) +                           \
             linear.constant(static_cast<T>(2) * l2_scalar);                   \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  v = v_new;                                                                   \
  m = m_new;                                                                   \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);                               \
  static_cast<KvVariable<Tindex, T>*>(table_m)->CoverUpdateUnsafe(key,         \
                                                                  &m_context); \
  static_cast<KvVariable<Tindex, T>*>(table_v)->CoverUpdateUnsafe(key,         \
                                                                  &v_context);

          COMPUTE_ADADQH(grad);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_ADADQH
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_v->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };
      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyAdaDQHV2Op: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyAdaDQHV2") \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyAdaDQHV2Op<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyAdamV2Op : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyAdamV2Op(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_linear = nullptr,
                        *table_m = nullptr, *table_v = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &table_m));
    core::ScopedUnref unref_me_m(table_m);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 3), &table_v));
    core::ScopedUnref unref_me_v(table_v);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));
    OP_REQUIRES(
        ctx, table_v->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(3)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(4);
    const Tensor& indices = ctx->input(5);
    const Tensor& lr = ctx->input(6);
    const Tensor& beta1_power = ctx->input(7);
    const Tensor& beta2_power = ctx->input(8);
    const Tensor& beta1 = ctx->input(9);
    const Tensor& beta2 = ctx->input(10);
    const Tensor& epsilon = ctx->input(11);
    const Tensor& l1 = ctx->input(12);
    const Tensor& l2 = ctx->input(13);
    const Tensor& l21 = ctx->input(14);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, linear_shape, m_shape, v_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);

    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);

    v_shape = table_v->value_shape();
    v_shape.InsertDim(0, N);

    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument(
                    "kv_variable and linear do not have the same shape",
                    var_shape.DebugString(), " ", linear_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("kv_variable and m do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("kv_variable and v do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_linear, &table_m,
                     &table_v, &indices, &grad, &lr, &beta1_power, &beta2_power,
                     &beta1, &beta2, &epsilon, &l1, &l2,
                     &l21](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();

        T lr_scalar = lr.scalar<T>()();
        T beta1_power_scalar = beta1_power.scalar<T>()();
        T beta2_power_scalar = beta2_power.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }

        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));

          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> linear_context;
          EVContext<T> m_context;
          EVContext<T> v_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_m)
              ->FindOrInsertUnsafe(key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_v)
              ->FindOrInsertUnsafe(key, &v_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(v_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_ADAM(grad_to_use)                                              \
  const T alpha =                                                              \
      Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar) /            \
      (static_cast<T>(1) - beta1_power_scalar);                                \
  m = beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use;     \
  auto new_v = beta2_scalar * v +                                              \
               (static_cast<T>(1) - beta2_scalar) * grad_to_use.square();      \
  if (beta1_scalar > beta1_power_scalar) {                                     \
    linear += alpha * m - (new_v.sqrt() - v.sqrt()) / lr_scalar * var;         \
  } else {                                                                     \
    linear +=                                                                  \
        alpha * m - (new_v.sqrt() - v.sqrt() + v.constant(epsilon_scalar)) /   \
                        lr_scalar * var;                                       \
  }                                                                            \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y = (new_v.sqrt() + new_v.constant(epsilon_scalar)) / lr_scalar +     \
             linear.constant(static_cast<T>(2) * l2_scalar);                   \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  static_cast<KvVariable<Tindex, T>*>(table_linear)                            \
      ->CoverUpdateUnsafe(key, &linear_context);                               \
  static_cast<KvVariable<Tindex, T>*>(table_m)->CoverUpdateUnsafe(key,         \
                                                                  &m_context); \
  static_cast<KvVariable<Tindex, T>*>(table_v)->CoverUpdateUnsafe(key,         \
                                                                  &v_context); \
  v = new_v;
          COMPUTE_ADAM(grad);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_ADAM
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_linear->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_v->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyAdamV2Op: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyAdamNewV2")\
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyAdamV2Op<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyAdamV3Op : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyAdamV3Op(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_m_v_linear = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &table_m_v_linear));
    core::ScopedUnref unref_me_linear(table_m_v_linear);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_m_v_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(2);
    const Tensor& indices = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& beta1_power = ctx->input(5);
    const Tensor& beta2_power = ctx->input(6);
    const Tensor& beta1 = ctx->input(7);
    const Tensor& beta2 = ctx->input(8);
    const Tensor& epsilon = ctx->input(9);
    const Tensor& l1 = ctx->input(10);
    const Tensor& l2 = ctx->input(11);
    const Tensor& l21 = ctx->input(12);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, m_v_linear_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    m_v_linear_shape = table_m_v_linear->value_shape();
    m_v_linear_shape.InsertDim(0, N);

    auto check_dim_size = [this](const TensorShape& var_shape,
                                 const TensorShape& m_v_linear_shape) {
      if (m_v_linear_shape.dims() != var_shape.dims()) return false;
      for (int d = 0; d < var_shape.dims(); d++) {
        auto var_dim_size = var_shape.dim_size(d);
        auto opt_dim_size = m_v_linear_shape.dim_size(d);
        if (var_dim_size != opt_dim_size && var_dim_size * 3 != opt_dim_size)
          return false;
      }
      return true;
    };

    OP_REQUIRES(
        ctx, check_dim_size(var_shape, m_v_linear_shape),
        errors::InvalidArgument(
            "kv_variable and linear do not have the same shape",
            var_shape.DebugString(), " ", m_v_linear_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    T lr_scalar = lr.scalar<T>()();
    T beta1_power_scalar = beta1_power.scalar<T>()();
    T beta2_power_scalar = beta2_power.scalar<T>()();
    T beta1_scalar = beta1.scalar<T>()();
    T beta2_scalar = beta2.scalar<T>()();
    T epsilon_scalar = epsilon.scalar<T>()();
    T l1_scalar = l1.scalar<T>()();
    T l2_scalar = l2.scalar<T>()();
    T l21_scalar = l21.scalar<T>()();
    const int64_t first_dim_size = indices.dim_size(0);
    const int64_t embedding_dim_size = grad.dim_size(1);
    size_t value_bytes = embedding_dim_size * sizeof(T);
    const T alpha =
        Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar) /
        (static_cast<T>(1) - beta1_power_scalar);
    auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim));

    if (N > 0) {
      auto DoWork = [this, ctx, &table_var, &table_m_v_linear, &indices, &grad,
                     &lr_scalar, &beta1_power_scalar, &beta2_power_scalar,
                     &beta1_scalar, &beta2_scalar, &epsilon_scalar, &l1_scalar,
                     &l2_scalar, &alpha, &l21_norm, value_bytes,
                     embedding_dim_size,
                     first_dim_size](int64_t start_i, int64_t limit_i) {
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();
        // prepare phstore variables
        std::vector<int64> train_deltalist;
        train_deltalist.reserve(limit_i - start_i);
        std::unique_ptr<T, void (*)(T*)> buf_var(
            static_cast<T*>(AllocateRaw(value_bytes)), DeallocateRaw<T>);
        std::unique_ptr<T, void (*)(T*)> buf_opt(
            static_cast<T*>(AllocateRaw(value_bytes * 3)), DeallocateRaw<T>);
        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context(buf_var.get(), false);
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          EVContext<T> opt_value_context(buf_opt.get(), false);
          static_cast<KvVariable<Tindex, T>*>(table_m_v_linear)
              ->FindOrInsertUnsafe(key, &opt_value_context, nullptr);
          auto m = FlatVector<T>(opt_value_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(opt_value_context.Value() + embedding_dim_size,
                                 embedding_dim_size);
          auto linear =
              FlatVector<T>(opt_value_context.Value() + 2 * embedding_dim_size,
                            embedding_dim_size);
          auto grad_value = grad_flat.template chip<0>(i);
// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_ADAM(grad_to_use)                                          \
  m = beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use; \
  auto new_v = beta2_scalar * v +                                          \
               (static_cast<T>(1) - beta2_scalar) * grad_to_use.square();  \
  auto new_v_sqrt = new_v.sqrt();                                          \
  if (beta1_scalar > beta1_power_scalar) {                                 \
    linear += alpha * m - (new_v_sqrt - v.sqrt()) / lr_scalar * var;       \
  } else {                                                                 \
    linear +=                                                              \
        alpha * m - (new_v_sqrt - v.sqrt() + v.constant(epsilon_scalar)) / \
                        lr_scalar * var;                                   \
  }                                                                        \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);    \
  auto l1_linear = l1_reg_adjust - linear;                                 \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;               \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();         \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                  \
  if (l1_linear_norm > l21_norm) {                                         \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;        \
    auto y = (new_v_sqrt + new_v.constant(epsilon_scalar)) / lr_scalar +   \
             linear.constant(static_cast<T>(2) * l2_scalar);               \
    var = l1_linear * l1_linear_norm / y;                                  \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(     \
        key, &var_context);                                                \
  } else {                                                                 \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(   \
        key, &var_context);                                                \
  }                                                                        \
  v = new_v;                                                               \
  static_cast<KvVariable<Tindex, T>*>(table_m_v_linear)                    \
      ->CoverUpdateUnsafe(key, &opt_value_context);
          COMPUTE_ADAM(grad_value);
          train_deltalist.push_back(i);
        }
#undef COMPUTE_ADAM
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m_v_linear->MarkAsDeltaListElements(ctx, indices,
                                                  train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyAdamV3Op: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyAdamV3")\
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyAdamV3Op<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class SparseApplyAdaBeliefOp : public OpKernel {
 public:
  explicit SparseApplyAdaBeliefOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();

    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2}, false);

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));

    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(m.shape()),
        errors::InvalidArgument("var and m do not have the same shape",
                                var.shape().DebugString(), " ",
                                m.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(v.shape()),
        errors::InvalidArgument("var and v do not have the same shape",
                                var.shape().DebugString(), " ",
                                v.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));


    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1_power = ctx->input(6);
    const Tensor& beta2_power = ctx->input(7);
    const Tensor& beta1 = ctx->input(8);
    const Tensor& beta2 = ctx->input(9);
    const Tensor& epsilon = ctx->input(10);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      auto indices_vec = indices.vec<Tindex>();
      auto var_flat = var.flat_outer_dims<T>();
      auto m_flat = m.flat_outer_dims<T>();
      auto v_flat = v.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      const T lr_scalar = lr.scalar<T>()();
      const T beta1_power_scalar = beta1_power.scalar<T>()();
      const T beta2_power_scalar = beta2_power.scalar<T>()();
      const T beta1_scalar = beta1.scalar<T>()();
      const T beta2_scalar = beta2.scalar<T>()();
      const T epsilon_scalar = epsilon.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        // Validate all the indices are in range
        const Tindex index = internal::SubtleMustCopy(indices_vec(i));
        OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                   errors::InvalidArgument(
                     strings::StrCat("Index ", index, " at offset ", i,
                                     " in indices is out of range")));

        auto var_ = var_flat.template chip<0>(index);
        auto m_ = m_flat.template chip<0>(index);
        auto v_ = v_flat.template chip<0>(index);
        auto grad_ = grad_flat.template chip<0>(i);

        const T alpha = lr_scalar * Eigen::numext::sqrt(T(1) - \
          beta2_power_scalar) / (T(1) - beta1_power_scalar);
        m_ = beta1_scalar * m_ + (T(1) - beta1_scalar) * grad_;
        v_ = beta2_scalar * v_ + (T(1) - beta2_scalar) * (grad_ - m_).square();
        var_ -= m_ * alpha / (v_.sqrt() + epsilon_scalar);
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);

    VLOG(1) << "SparseApplyAdaBeliefOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                 \
  REGISTER_KERNEL_BUILDER(                            \
      Name("SparseApplyAdaBelief")                    \
          .Device(DEVICE_CPU)                         \
          .TypeConstraint<T>("T")                     \
          .TypeConstraint<Tindices>("Tindices"),      \
      SparseApplyAdaBeliefOp<CPUDevice, T, Tindices>);\
  REGISTER_KERNEL_BUILDER(                            \
      Name("ResourceSparseApplyAdaBelief")            \
          .Device(DEVICE_CPU)                         \
          .TypeConstraint<T>("T")                     \
          .TypeConstraint<Tindices>("Tindices"),      \
      SparseApplyAdaBeliefOp<CPUDevice, T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class SparseApplyAdaHessianOp : public OpKernel {
 public:
  explicit SparseApplyAdaHessianOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();

    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2}, false);

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));

    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(m.shape()),
        errors::InvalidArgument("var and m do not have the same shape",
                                var.shape().DebugString(), " ",
                                m.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(v.shape()),
        errors::InvalidArgument("var and v do not have the same shape",
                                var.shape().DebugString(), " ",
                                v.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));


    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    const Tensor& hessian = ctx->input(5);
    const Tensor& lr = ctx->input(6);
    const Tensor& beta1_power = ctx->input(7);
    const Tensor& beta2_power = ctx->input(8);
    const Tensor& beta1 = ctx->input(9);
    const Tensor& beta2 = ctx->input(10);
    const Tensor& epsilon = ctx->input(11);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      auto indices_vec = indices.vec<Tindex>();
      auto var_flat = var.flat_outer_dims<T>();
      auto m_flat = m.flat_outer_dims<T>();
      auto v_flat = v.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      auto hessian_flat = hessian.flat_outer_dims<T>();
      const T lr_scalar = lr.scalar<T>()();
      const T beta1_power_scalar = beta1_power.scalar<T>()();
      const T beta2_power_scalar = beta2_power.scalar<T>()();
      const T beta1_scalar = beta1.scalar<T>()();
      const T beta2_scalar = beta2.scalar<T>()();
      const T epsilon_scalar = epsilon.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        // Validate all the indices are in range
        const Tindex index = internal::SubtleMustCopy(indices_vec(i));
        OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                   errors::InvalidArgument(
                     strings::StrCat("Index ", index, " at offset ", i,
                                     " in indices is out of range")));

        auto var_ = var_flat.template chip<0>(index);
        auto m_ = m_flat.template chip<0>(index);
        auto v_ = v_flat.template chip<0>(index);
        auto grad_ = grad_flat.template chip<0>(i);
        auto h_ = hessian_flat.template chip<0>(i);

        const T alpha = lr_scalar * Eigen::numext::sqrt(T(1) - \
          beta2_power_scalar) / (T(1) - beta1_power_scalar);
        m_ = beta1_scalar * m_ + (T(1) - beta1_scalar) * grad_;
        v_ = beta2_scalar * v_ + (T(1) - beta2_scalar) * h_.square();
        var_ -= m_ * alpha / (v_.sqrt() + epsilon_scalar);
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);

    VLOG(1) << "SparseApplyAdaHessianOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                  \
  REGISTER_KERNEL_BUILDER(                             \
      Name("SparseApplyAdaHessian")                    \
          .Device(DEVICE_CPU)                          \
          .TypeConstraint<T>("T")                      \
          .TypeConstraint<Tindices>("Tindices"),       \
      SparseApplyAdaHessianOp<CPUDevice, T, Tindices>);\
  REGISTER_KERNEL_BUILDER(                             \
      Name("ResourceSparseApplyAdaHessian")            \
          .Device(DEVICE_CPU)                          \
          .TypeConstraint<T>("T")                      \
          .TypeConstraint<Tindices>("Tindices"),       \
      SparseApplyAdaHessianOp<CPUDevice, T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableComputeGroupAdaDQHHPOp : public OpKernel {
 public:
  explicit KvVariableComputeGroupAdaDQHHPOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 3});

    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_m = nullptr,
                        *table_v = nullptr, *table_linear = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_m));
    core::ScopedUnref unref_me_m(table_m);
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &table_v));
    core::ScopedUnref unref_me_v(table_v);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 3), &table_linear));
    core::ScopedUnref unref_me_linear(table_linear);
    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_m->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, table_v->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));
    OP_REQUIRES(
        ctx, table_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(3)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(4);
    const Tensor& indices = ctx->input(5);
    const Tensor& lr = ctx->input(6);
    const Tensor& beta1_power = ctx->input(7);
    const Tensor& beta2_power = ctx->input(8);
    const Tensor& beta1 = ctx->input(9);
    const Tensor& beta2 = ctx->input(10);
    const Tensor& epsilon = ctx->input(11);
    const Tensor& l1 = ctx->input(12);
    const Tensor& l2 = ctx->input(13);
    const Tensor& l21 = ctx->input(14);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, m_shape, v_shape, linear_shape;
    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);
    m_shape = table_m->value_shape();
    m_shape.InsertDim(0, N);
    v_shape = table_v->value_shape();
    v_shape.InsertDim(0, N);
    linear_shape = table_linear->value_shape();
    linear_shape.InsertDim(0, N);
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("kv_variable and m do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("kv_variable and v do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
                errors::InvalidArgument("kv_variable and linear do not "
                                        "have the same shape",
                                        var_shape.DebugString(), " ",
                                        linear_shape.DebugString()));
    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    // Create hypergradient of lr and eps
    Tensor *lr_hg = nullptr, *eps_hg = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, grad.shape(), &lr_hg));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, grad.shape(), &eps_hg));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_m,
                     &table_v, &table_linear, &indices, &grad, &lr,
                     &beta1_power, &beta2_power, &beta1, &beta2, &epsilon,
                     &l1, &l2, &l21,
                     &lr_hg, &eps_hg](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();
        auto lr_hg_flat = lr_hg->flat_outer_dims<T>();
        auto eps_hg_flat = eps_hg->flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();
        T beta1_power_scalar = beta1_power.scalar<T>()();
        T beta2_power_scalar = beta2_power.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();

        for (int64_t i = start_i; i < limit_i; ++i) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> m_context;
          EVContext<T> v_context;
          EVContext<T> linear_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_linear)
              ->FindOrInsertUnsafe(key, &linear_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_m)
              ->FindOrInsertUnsafe(key, &m_context, nullptr);
          static_cast<KvVariable<Tindex, T>*>(table_v)
              ->FindOrInsertUnsafe(key, &v_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(m_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(v_context.Value(), embedding_dim_size);
          auto linear =
              FlatVector<T>(linear_context.Value(), embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);
          auto lr_hg = lr_hg_flat.template chip<0>(i);
          auto eps_hg = eps_hg_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_GROUPADADQHHG(grad_to_use)                                     \
  const T epsilon_adjust = epsilon_scalar *                                    \
    Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar / beta2_scalar);\
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  auto y = v.sqrt().cwiseMax(epsilon_adjust);                                  \
  auto deno = (y + y.constant(static_cast<T>(2) * l2_scalar *                  \
    lr_scalar)).square();                                                      \
  auto indicator = (v.constant(epsilon_adjust) >= v.sqrt()).select(            \
    v.constant(static_cast<T>(1)), v.constant(static_cast<T>(0)));             \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    lr_hg = y / deno * l1_linear_norm * l1_linear;                             \
    eps_hg = -lr_scalar * Eigen::numext::sqrt(static_cast<T>(1) -              \
      beta2_power_scalar / beta2_scalar) / y * indicator                       \
      * l1_linear_norm * l1_linear;                                            \
  } else {                                                                     \
    lr_hg.setZero();                                                           \
    eps_hg.setZero(); /*set to zero*/                                          \
  }
          COMPUTE_GROUPADADQHHG(grad);
        }
#undef COMPUTE_GROUPADADQHHG
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableComputeGroupAdaDQHHPOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableComputeAdaDQHHG")          \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableComputeGroupAdaDQHHPOp<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS)
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
struct ComputeAdaDQHHG {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  const bool& sam, typename TTypes<T>::Flat delta,
                  typename TTypes<T>::ConstScalar alpha,
                  typename TTypes<T>::Flat lr_hg,
                  typename TTypes<T>::Flat eps_hg) {
    const T adjust_bias = Eigen::numext::sqrt(T(1) - beta2_power() / beta2()) /
                    (T(1) - beta1_power() / beta1());
    const T epsilon_adjust = epsilon() *
                    Eigen::numext::sqrt(T(1) - beta2_power() / beta2());

    auto deno = v.sqrt().cwiseMax(epsilon_adjust);
    auto indicator = (v.constant(epsilon_adjust) >= v.sqrt()).select(
      v.constant(T(1)), v.constant(T(0)));

    lr_hg = -adjust_bias * m / deno;
    eps_hg = lr() * adjust_bias * m / deno.square() * indicator;
    if (sam) {
      lr_hg -= (T(1) - alpha()) * delta;
    }
  }
};

template <typename Device, typename T>
class ComputeAdaDQHHGOp: public OpKernel {
 public:
  explicit ComputeAdaDQHHGOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();

    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2, 11}, false);

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));

    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));

    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));

    Tensor delta;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 11, use_exclusive_lock_, sparse, &delta));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(2)));

    OP_REQUIRES(
        ctx, delta.IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(11)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& beta1_power = ctx->input(5);
    const Tensor& beta2_power = ctx->input(6);
    const Tensor& beta1 = ctx->input(7);
    const Tensor& beta2 = ctx->input(8);
    const Tensor& epsilon = ctx->input(9);
    const Tensor& sam = ctx->input(10);
    const Tensor& alpha = ctx->input(12);

    // Create hypergradient of lr and eps
    Tensor *lr_hg = NULL, *eps_hg = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, grad.shape(), &lr_hg));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, grad.shape(), &eps_hg));

    const Device& device = ctx->template eigen_device<Device>();
    ComputeAdaDQHHG<Device, T>()(device, var.flat<T>(), m.flat<T>(),
                                 v.flat<T>(), grad.flat<T>(),
                                 lr.scalar<T>(), beta1_power.scalar<T>(),
                                 beta2_power.scalar<T>(), beta1.scalar<T>(),
                                 beta2.scalar<T>(), epsilon.scalar<T>(),
                                 sam.scalar<bool>()(), delta.flat<T>(),
                                 alpha.scalar<T>(), lr_hg->flat<T>(),
                                 eps_hg->flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 2);

    VLOG(1) << "ComputeAdaDQHHGOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("ComputeAdaDQHHG").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ComputeAdaDQHHGOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceComputeAdaDQHHG")                \
                              .HostMemory("var")                         \
                              .HostMemory("m")                           \
                              .HostMemory("v")                           \
                              .HostMemory("delta")                       \
                              .Device(DEVICE_##D)                        \
                              .TypeConstraint<T>("T"),                   \
                          ComputeAdaDQHHGOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyRectifiedAdamOp : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyRectifiedAdamOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_opt = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &table_opt));
    core::ScopedUnref unref_me_linear(table_opt);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_opt->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(2);
    const Tensor& indices = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& beta1_power = ctx->input(5);
    const Tensor& beta2_power = ctx->input(6);
    const Tensor& beta1 = ctx->input(7);
    const Tensor& beta2 = ctx->input(8);
    const Tensor& epsilon = ctx->input(9);
    const Tensor& l1 = ctx->input(10);
    const Tensor& l2 = ctx->input(11);
    const Tensor& l21 = ctx->input(12);
    const Tensor& r_t = ctx->input(13);
    const Tensor& tractable = ctx->input(14);
    const Tensor& amsgrad = ctx->input(15);
    const Tensor& use_nesterov = ctx->input(16);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, opt_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    opt_shape = table_opt->value_shape();
    opt_shape.InsertDim(0, N);

    auto check_dim_size = [this](const TensorShape& var_shape,
                                 const TensorShape& opt_shape) {
      if (opt_shape.dims() != var_shape.dims()) return false;
      for (int d = 0; d < var_shape.dims(); d++) {
        auto var_dim_size = var_shape.dim_size(d);
        auto opt_dim_size = opt_shape.dim_size(d);
        if (var_dim_size != opt_dim_size && var_dim_size * 5 != opt_dim_size)
          return false;
      }
      return true;
    };

    OP_REQUIRES(ctx, check_dim_size(var_shape, opt_shape),
                errors::InvalidArgument(
                    "kv_variable and opt_shape do not have the same shape",
                    var_shape.DebugString(), " ", opt_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      auto DoWork = [this, ctx, inner_dim, &table_var, &table_opt, &indices,
                     &grad, &lr, &beta1_power, &beta2_power, &beta1, &beta2,
                     &epsilon, &l1, &l2, &l21, &r_t, &tractable, &amsgrad,
                     &use_nesterov](int64_t start_i, int64_t limit_i) {
        const int64_t first_dim_size = indices.dim_size(0);
        const int64_t embedding_dim_size = grad.dim_size(1);
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();

        T lr_scalar = lr.scalar<T>()();
        T beta1_power_scalar = beta1_power.scalar<T>()();
        T beta2_power_scalar = beta2_power.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l21_scalar = l21.scalar<T>()();
        T r_t_scalar = r_t.scalar<T>()();
        bool tractable_flag = tractable.scalar<bool>()();
        bool amsgrad_flag = amsgrad.scalar<bool>()();
        bool use_nesterov_flag = use_nesterov.scalar<bool>()();
        std::vector<int64> train_deltalist;
        auto need_delta_info = table_var->NeedDeltaInfo();
        if (need_delta_info) {
          train_deltalist.reserve(limit_i - start_i);
        }
        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));

          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context;
          EVContext<T> opt_context;
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          static_cast<KvVariable<Tindex, T>*>(table_opt)->FindOrInsertUnsafe(
              key, &opt_context, nullptr);
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          auto m = FlatVector<T>(opt_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(opt_context.Value() + embedding_dim_size,
                                 embedding_dim_size);
          auto linear = FlatVector<T>(
              opt_context.Value() + embedding_dim_size * 2, embedding_dim_size);
          auto vhat = FlatVector<T>(
              opt_context.Value() + embedding_dim_size * 3, embedding_dim_size);
          auto vamsgrad = FlatVector<T>(
              opt_context.Value() + embedding_dim_size * 4, embedding_dim_size);
          auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_RECTFIED_ADAM(grad_to_use)                                     \
  const T alpha = Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar); \
  m = beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use;     \
  auto new_v = beta2_scalar * v +                                              \
               (static_cast<T>(1) - beta2_scalar) * grad_to_use.square();      \
  auto m_corr = m;                                                             \
  if (use_nesterov_flag) {                                                     \
    m_corr =                                                                   \
        grad_to_use * (static_cast<T>(1) - beta1_scalar) + beta1_scalar * m;   \
  }                                                                            \
                                                                               \
  if (!tractable_flag) {                                                       \
    auto radam_m = m_corr / (static_cast<T>(1) - beta1_power_scalar);          \
    auto radam_v = v.constant(static_cast<T>(1)) / lr_scalar;                  \
    auto radam_v_old = vhat;                                                   \
    APPLY_RECTIFIED_ADAM(radam_m, radam_v, radam_v_old);                       \
  } else if (amsgrad_flag) {                                                   \
    auto radam_m = r_t_scalar * m / (static_cast<T>(1) - beta1_power_scalar);  \
    vamsgrad = new_v.cwiseMax(vamsgrad);                                       \
                                                                               \
    auto radam_v =                                                             \
        (vamsgrad.sqrt() / alpha + v.constant(epsilon_scalar)) / lr_scalar;    \
    auto radam_v_old = vhat;                                                   \
    APPLY_RECTIFIED_ADAM(radam_m, radam_v, radam_v_old);                       \
  } else {                                                                     \
    auto radam_m = r_t_scalar * m / (static_cast<T>(1) - beta1_power_scalar);  \
    auto radam_v =                                                             \
        (new_v.sqrt() / alpha + v.constant(epsilon_scalar)) / lr_scalar;       \
    auto radam_v_old = vhat;                                                   \
    APPLY_RECTIFIED_ADAM(radam_m, radam_v, radam_v_old);                       \
  }

#define APPLY_RECTIFIED_ADAM(radam_m, radam_v, radam_v_old)                    \
  linear += radam_m - (radam_v - radam_v_old) * var;                           \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);        \
  auto l1_linear = l1_reg_adjust - linear;                                     \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;                   \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();             \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                      \
  auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim)); \
  if (l1_linear_norm > l21_norm) {                                             \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;            \
    auto y = radam_v + linear.constant(static_cast<T>(2) * l2_scalar);         \
    var = l1_linear * l1_linear_norm / y;                                      \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(         \
        key, &var_context);                                                    \
  } else {                                                                     \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(       \
        key, &var_context);                                                    \
  }                                                                            \
  vhat = radam_v;                                                              \
  v = new_v;                                                                   \
  static_cast<KvVariable<Tindex, T>*>(table_opt)->CoverUpdateUnsafe(           \
      key, &opt_context);
          COMPUTE_RECTFIED_ADAM(grad);
          if (need_delta_info) {
            train_deltalist.push_back(i);
          }
        }
#undef COMPUTE_RECTFIED_ADAM
#undef APPLY_RECTIFIED_ADAM
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_opt->MarkAsDeltaListElements(ctx, indices, train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyRectifiedAdamOp: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)                \
  REGISTER_KERNEL_BUILDER(                           \
      Name("KvVariableGroupSparseApplyRectifiedAdam")\
          .Device(DEVICE_CPU)                        \
          .TypeConstraint<T>("T")                    \
          .TypeConstraint<Tindices>("Tindices"),     \
      KvVariableGroupSparseApplyRectifiedAdamOp<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
//  REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvVariableGroupSparseApplyAdamV4Op : public OpKernel {
 public:
  explicit KvVariableGroupSparseApplyAdamV4Op(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    uint64_t stime = tensorflow::Env::Default()->NowMicros();
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1});
    // Get the KvVariable handle
    KvVariableInterface *table_var = nullptr, *table_m_v_linear = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &table_var));
    core::ScopedUnref unref_me_var(table_var);
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &table_m_v_linear));
    core::ScopedUnref unref_me_linear(table_m_v_linear);

    OP_REQUIRES(
        ctx, table_var->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(0)));
    OP_REQUIRES(
        ctx, table_m_v_linear->IsInitialized(),
        errors::FailedPrecondition("Failed to use uninitialized variables: ",
                                   requested_input(1)));

    // Get gradients and indices
    const Tensor& grad = ctx->input(2);
    const Tensor& indices = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& beta1_power = ctx->input(5);
    const Tensor& beta2_power = ctx->input(6);
    const Tensor& beta1 = ctx->input(7);
    const Tensor& beta2 = ctx->input(8);
    const Tensor& epsilon = ctx->input(9);
    const Tensor& l1 = ctx->input(10);
    const Tensor& l2 = ctx->input(11);
    const Tensor& l21 = ctx->input(12);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l1.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l2.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l21.shape()) &&
                    l21.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l21 regularization strength is not "
                                        "a non-negative scalar: ",
                                        l21.shape().DebugString()));

    const int64_t N = indices.dim_size(0);
    TensorShape var_shape, m_v_linear_shape;

    var_shape = table_var->value_shape();
    var_shape.InsertDim(0, N);

    m_v_linear_shape = table_m_v_linear->value_shape();
    m_v_linear_shape.InsertDim(0, N);

    auto check_dim_size = [this](const TensorShape& var_shape,
                                 const TensorShape& m_v_linear_shape) {
      if (m_v_linear_shape.dims() != var_shape.dims()) return false;
      for (int d = 0; d < var_shape.dims(); d++) {
        auto var_dim_size = var_shape.dim_size(d);
        auto opt_dim_size = m_v_linear_shape.dim_size(d);
        if (var_dim_size != opt_dim_size && var_dim_size * 3 != opt_dim_size)
          return false;
      }
      return true;
    };

    OP_REQUIRES(
        ctx, check_dim_size(var_shape, m_v_linear_shape),
        errors::InvalidArgument(
            "kv_variable and linear do not have the same shape",
            var_shape.DebugString(), " ", m_v_linear_shape.DebugString()));

    int64_t inner_dim = 1;
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    T lr_scalar = lr.scalar<T>()();
    T beta1_power_scalar = beta1_power.scalar<T>()();
    T beta2_power_scalar = beta2_power.scalar<T>()();
    T beta1_scalar = beta1.scalar<T>()();
    T beta2_scalar = beta2.scalar<T>()();
    T epsilon_scalar = epsilon.scalar<T>()();
    T l1_scalar = l1.scalar<T>()() * lr_scalar;
    T l2_scalar = l2.scalar<T>()() * lr_scalar;
    T l21_scalar = l21.scalar<T>()() * lr_scalar;
    const int64_t first_dim_size = indices.dim_size(0);
    const int64_t embedding_dim_size = grad.dim_size(1);
    size_t value_bytes = embedding_dim_size * sizeof(T);
    const T alpha =
      lr_scalar * Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar) /
      (static_cast<T>(1) - beta1_power_scalar);
    auto l21_norm = l21_scalar * Eigen::numext::sqrt(static_cast<T>(inner_dim));

    if (N > 0) {
      auto DoWork = [this, ctx, &table_var, &table_m_v_linear, &indices, &grad,
                     &beta1_power_scalar, &beta2_power_scalar,
                     &beta1_scalar, &beta2_scalar, &epsilon_scalar, &l1_scalar,
                     &l2_scalar, &alpha, &l21_norm,
                     value_bytes, embedding_dim_size,
                     first_dim_size](int64_t start_i, int64_t limit_i) {
        auto grad_flat = grad.flat_outer_dims<T>();
        auto indices_flat = indices.flat<Tindex>();
        // prepare phstore variables
        std::vector<int64> train_deltalist;
        train_deltalist.reserve(limit_i - start_i);
        std::unique_ptr<T, void (*)(T*)> buf_var(
            static_cast<T*>(AllocateRaw(value_bytes)), DeallocateRaw<T>);
        std::unique_ptr<T, void (*)(T*)> buf_opt(
            static_cast<T*>(AllocateRaw(value_bytes * 3)), DeallocateRaw<T>);
        for (int64_t i = start_i; i < limit_i; i++) {
          OP_REQUIRES(ctx, FastBoundsCheck(i, first_dim_size),
                      errors::InvalidArgument(strings::StrCat(
                          "Index ", i, " out of range ", first_dim_size)));
          Tindex key = indices_flat(i);
          bool should_filter = false;
          EVContext<T> var_context(buf_var.get(), false);
          auto var_lock =
              static_cast<KvVariable<Tindex, T>*>(table_var)->GetScopedKeyLock(
                  key, LockType::WRITE_LOCK);
          static_cast<KvVariable<Tindex, T>*>(table_var)->FindOrInsertUnsafe(
              key, &var_context, &should_filter);
          if (should_filter) {
            continue;
          }
          auto var = FlatVector<T>(var_context.Value(), embedding_dim_size);
          EVContext<T> opt_value_context(buf_opt.get(), false);
          static_cast<KvVariable<Tindex, T>*>(table_m_v_linear)
              ->FindOrInsertUnsafe(key, &opt_value_context, nullptr);
          auto m = FlatVector<T>(opt_value_context.Value(), embedding_dim_size);
          auto v = FlatVector<T>(opt_value_context.Value() + embedding_dim_size,
                                 embedding_dim_size);
          auto linear =
              FlatVector<T>(opt_value_context.Value() + 2 * embedding_dim_size,
                            embedding_dim_size);
          auto grad_value = grad_flat.template chip<0>(i);
// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_ADAM(grad_to_use)                                          \
  m = beta1_scalar * m + (static_cast<T>(1) - beta1_scalar) * grad_to_use; \
  auto new_v = beta2_scalar * v +                                          \
               (static_cast<T>(1) - beta2_scalar) * grad_to_use.square();  \
  auto new_v_sqrt = new_v.sqrt();                                          \
  if (beta1_scalar > beta1_power_scalar) {                                 \
    linear += alpha * m - (new_v_sqrt - v.sqrt()) * var;                   \
  } else {                                                                 \
    linear += alpha * m - (new_v_sqrt +                                    \
              new_v_sqrt.constant(epsilon_scalar)) * var;                  \
  }                                                                        \
  auto l1_reg_adjust = linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar);    \
  auto l1_linear = l1_reg_adjust - linear;                                 \
  using TensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;               \
  TensorScalar l1_linear_norm_t = l1_linear.square().sum().sqrt();         \
  T l1_linear_norm = static_cast<T>(l1_linear_norm_t(0));                  \
  if (l1_linear_norm > l21_norm) {                                         \
    l1_linear_norm = static_cast<T>(1) - l21_norm / l1_linear_norm;        \
    auto y = new_v_sqrt + new_v.constant(epsilon_scalar) +                 \
             linear.constant(static_cast<T>(2) * l2_scalar);               \
    var = l1_linear * l1_linear_norm / y;                                  \
    static_cast<KvVariable<Tindex, T>*>(table_var)->CoverUpdateUnsafe(     \
        key, &var_context);                                                \
  } else {                                                                 \
    static_cast<KvVariable<Tindex, T>*>(table_var)->MarkBlacklistUnsafe(   \
        key, &var_context);                                                \
  }                                                                        \
  v = new_v;                                                               \
  static_cast<KvVariable<Tindex, T>*>(table_m_v_linear)                    \
      ->CoverUpdateUnsafe(key, &opt_value_context);
          COMPUTE_ADAM(grad_value);
          train_deltalist.push_back(i);
        }
#undef COMPUTE_ADAM
        table_var->MarkAsDeltaListElements(ctx, indices, train_deltalist);
        table_m_v_linear->MarkAsDeltaListElements(ctx, indices,
                                                  train_deltalist);
      };

      const int64_t cost = 5000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }

    VLOG(1) << "KvVariableGroupSparseApplyAdamV4Op: "
            << ::tensorflow::Env::Default()->NowMicros() - stime << " ms";
  }

 private:
  bool use_exclusive_lock_;
};
#define REGISTER_KERNELS(T, Tindices)            \
  REGISTER_KERNEL_BUILDER(                       \
      Name("KvVariableGroupSparseApplyAdamV4")   \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<T>("T")                \
          .TypeConstraint<Tindices>("Tindices"), \
      KvVariableGroupSparseApplyAdamV4Op<CPUDevice, T, Tindices>);

#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);   \
  REGISTER_KERNELS(T, uint64);  \
  // REGISTER_KERNELS(T, string);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS
}  // NOLINT(readability/fn_size) namespace tfplus
