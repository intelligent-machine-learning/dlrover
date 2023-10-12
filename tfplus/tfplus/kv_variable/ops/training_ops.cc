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
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeAndType;
using ::tensorflow::shape_inference::ShapeHandle;
using namespace tensorflow;  // NOLINT(build/namespaces)

// Code from
// https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/core/ops/training_ops.cc#L25
static ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

// Code from
// https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/core/ops/training_ops.cc#L37
static Status HandleGradAndIndicesInputs(InferenceContext* c, bool sparse,
                                         int grad_idx, ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape(c, grad_idx);
  if (!sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, grad, s));
    return ::tensorflow::OkStatus();
  }
  // Indices is a vector where indices.dim[0].rank == grad[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(grad_idx + 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused));

  // Trailing part of grad matches trailing part of *s.
  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(grad, 0, c->UnknownDim(), &grad_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, grad_unknown_first, s));

  return ::tensorflow::OkStatus();
}

static Status HandleHessAndIndicesInputs(InferenceContext* c, bool sparse,
                                         int hess_idx, ShapeHandle* s) {
  ShapeHandle hess = ShapeOrHandleShape(c, hess_idx);
  if (!sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, hess, s));
    return ::tensorflow::OkStatus();
  }
  // Indices is a vector where indices.dim[0].rank == hess[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(hess_idx - 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(hess, 0), &unused));

  // Trailing part of hess matches trailing part of *s.
  ShapeHandle hess_unknown_first;
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(hess, 0, c->UnknownDim(), &hess_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, hess_unknown_first, s));

  return ::tensorflow::OkStatus();
}

// Code from
// https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/core/ops/training_ops.cc#L461
static Status ApplyFtrlShapeFn(InferenceContext* c, bool l21) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // linear
  TF_RETURN_IF_ERROR(HandleGradAndIndicesInputs(c, true, 3 /* grad_idx */, &s));
  int idx = 5;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  if (l21) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l21
  }
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2_shrinkage
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr_power
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableSparseApplyFtrlV2")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) { return ApplyFtrlShapeFn(c, false); });

REGISTER_OP("KvVariableGroupSparseApplyFtrlV2")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) { return ApplyFtrlShapeFn(c, false); });

REGISTER_OP("KvVariableSparseGroupSparseApplyFtrlV2")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) { return ApplyFtrlShapeFn(c, true); });

static Status GroupApplyAdamShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // linear
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 4), &s));  // v
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 5 /*grad_idx */, &s));  // grad
  int idx = 7;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l21
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyAdamV2")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyAdamShapeFn(c);
    });

static Status ApplyAdagradShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 3 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableSparseApplyAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradShapeFn(c);
    });

static Status GroupApplyAMSGradShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // vhat
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // linear
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 4), &s));  // v
  TF_RETURN_IF_ERROR(
    HandleGradAndIndicesInputs(c, true, 5 /*grad_idx */, &s));          // grad
  int idx = 7;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l21
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyAMSGrad")
    .Input("var: resource")
    .Input("vhat: resource")
    .Input("linear: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyAMSGradShapeFn(c);
    });

static Status ApplyAdadeltaShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape(c, 2), &s));            // accum update
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));  // rho
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 6 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableSparseApplyAdadelta")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("accum_update: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdadeltaShapeFn(c);
    });

static Status GroupApplyAdadeltaShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(
    c->Merge(s, ShapeOrHandleShape(c, 2), &s));             // accum_update
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // linear
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // rho
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 7 /*grad_idx */, &s));        // grad
  int idx = 9;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));   // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));   // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));   // l21
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyAdadelta")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("accum_update: resource")
    .Input("linear: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyAdadeltaShapeFn(c);
    });

static Status GroupApplyMomentumShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // linear
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // m
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 5 /*grad_idx */, &s));        // grad
  int idx = 7;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));   // momentum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));   // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));   // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));   // l21
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyMomentum")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("m: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("momentum: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyMomentumShapeFn(c);
    });

static Status GroupApplyAdaHessianShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // linear
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 4), &s));  // v
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 5 /*grad_idx */, &s));  // grad
  int idx = sparse ? 7 : 6;
  // hessian
  TF_RETURN_IF_ERROR(
      HandleHessAndIndicesInputs(c, sparse, idx++ /*hess_idx */, &s));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l21
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyAdaHessian")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("hessian: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyAdaHessianShapeFn(c, true);
    });

static Status ApplyAdaHessianShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 3 /*grad_idx */, &s));  // grad
  int idx = sparse ? 5 : 4;
  // hessian
  TF_RETURN_IF_ERROR(
      HandleHessAndIndicesInputs(c, sparse, idx++ /*hess_idx */, &s));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("ApplyAdaHessian")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("grad: T")
    .Input("hessian: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaHessianShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceApplyAdaHessian")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("hessian: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaHessianShapeFn(c, false /* sparse */);
    });

REGISTER_OP("SparseApplyAdaHessian")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("hessian: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaHessianShapeFn(c, true);
    });

REGISTER_OP("ResourceSparseApplyAdaHessian")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("hessian: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaHessianShapeFn(c, true);
    });

static Status GroupApplyAdaBeliefShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // linear
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 4), &s));  // v
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 5 /*grad_idx */, &s));  // grad
  int idx = 7;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l21
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyAdaBelief")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyAdaBeliefShapeFn(c);
    });

static Status ApplyAdaBeliefShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 3 /*grad_idx */, &s));  // grad
  int idx = sparse ? 5 : 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("ApplyAdaBelief")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("grad: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaBeliefShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceApplyAdaBelief")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaBeliefShapeFn(c, false /* sparse */);
    });

REGISTER_OP("SparseApplyAdaBelief")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaBeliefShapeFn(c, true);
    });

REGISTER_OP("ResourceSparseApplyAdaBelief")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaBeliefShapeFn(c, true);
    });

static Status GroupApplyLambShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // linear
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 4), &s));  // v
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 5 /*grad_idx */, &s));  // grad
  int idx = 7;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l21
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyLamb")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyLambShapeFn(c);
    });

static Status ApplyLambShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 3 /*grad_idx */, &s));  // grad
  int idx = sparse ? 5 : 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("ApplyLamb")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("grad: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyLambShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceApplyLamb")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyLambShapeFn(c, false /* sparse */);
    });

REGISTER_OP("KvVariableGroupSparseApplyLambHessian")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("hessian: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyAdaHessianShapeFn(c, true);
    });

REGISTER_OP("ApplyLambHessian")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("grad: T")
    .Input("hessian: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaHessianShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceApplyLambHessian")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("hessian: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaHessianShapeFn(c, false /* sparse */);
    });

static Status ApplyAdaDQHShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 3 /*grad_idx */, &s));  // grad
  int idx = sparse ? 5 : 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("ApplyAdaDQH")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("grad: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaDQHShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceApplyAdaDQH")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaDQHShapeFn(c, false /* sparse */);
    });

REGISTER_OP("KvVariableSparseApplyAdaDQH")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaDQHShapeFn(c, true);
    });

REGISTER_OP("SparseApplyAdaDQH")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaDQHShapeFn(c, true);
    });

REGISTER_OP("ResourceSparseApplyAdaDQH")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaDQHShapeFn(c, true);
    });

static Status GroupApplyAdaDQHShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // linear
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 4 /*grad_idx */, &s));  // grad
  int idx = 6;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l21
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyAdaDQH")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyAdaDQHShapeFn(c);
    });

static Status GroupApplyAdamV2ShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // linear
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // v
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 4 /*grad_idx */, &s));  // grad
  int idx = 6;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l21
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyAdamNewV2")
    .Input("var: resource")
    .Input("linear: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyAdamV2ShapeFn(c);
    });

static Status GroupApplyAdamV3ShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 0), &s));  // m_v_linear
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 0), &s));
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 0), &s));
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 2 /*grad_idx */, &s));  // grad
  int idx = 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l21
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyAdamV3")
    .Input("var: resource")
    .Input("m_v_linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyAdamV3ShapeFn(c);
    });

REGISTER_OP("KvVariableComputeAdaDQHHG")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Output("lr_hg: T")
    .Output("eps_hg: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false");

REGISTER_OP("ComputeAdaDQHHG")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("grad: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("sam: bool")
    .Input("delta: Ref(T)")
    .Input("alpha: T")
    .Output("lr_hg: T")
    .Output("eps_hg: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false");

REGISTER_OP("ResourceComputeAdaDQHHG")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("grad: T")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("sam: bool")
    .Input("delta: resource")
    .Input("alpha: T")
    .Output("lr_hg: T")
    .Output("eps_hg: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false");

static Status GroupApplyRectifiedAdamShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 0), &s));  // opt
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 2 /*grad_idx */, &s));  // grad
  int idx = 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l21
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // r_t
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // tractable
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // amsgrad
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // use_nesterov
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyRectifiedAdam")
    .Input("var: resource")
    .Input("opt: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Input("r_t: T")
    .Input("tractable: bool")
    .Input("amsgrad: bool")
    .Input("use_nesterov: bool")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyRectifiedAdamShapeFn(c);
    });

static Status GroupApplyAdaDQHV2ShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // linear
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, true, 4 /*grad_idx */, &s));  // grad
  int idx = 6;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l21
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return ::tensorflow::OkStatus();
}

REGISTER_OP("KvVariableGroupSparseApplyAdaDQHV2")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyAdaDQHV2ShapeFn(c);
    });

REGISTER_OP("KvVariableGroupSparseApplyAdamV4")
    .Input("var: resource")
    .Input("m_v_linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("beat1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l21: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return GroupApplyAdamV3ShapeFn(c);
    });
