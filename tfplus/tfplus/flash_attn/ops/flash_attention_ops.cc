// Copyright 2023 The TF-plus Authors. All Rights Reserved.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("FMHAForward")
    .Input("query: dtype")  // (B x S, H, K)
    .Input("key: dtype")    // (B x S, H, K)
    .Input("value: dtype")  // (B x S, H, K)
    .Input("cu_seqlens_q: int32")
    .Input("cu_seqlens_k: int32")
    .Input("max_seqlen_q: int32")
    .Input("max_seqlen_k: int32")
    .Output("output: dtype")
    .Output("softmax_lse: float")
    .Output("return_sm: dtype")
    .Output("rng_state: uint64")
    .Attr("dtype: type")
    .Attr("p_dropout: float")
    .Attr("softmax_scale: float")
    .Attr("zero_tensors: bool")
    .Attr("is_causal: bool")
    .Attr("return_softmax: bool")
    .Attr("num_splits: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle query_shape;
      shape_inference::ShapeHandle key_shape;
      shape_inference::ShapeHandle value_shape;
      // check ranks
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &query_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &key_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &value_shape));
      c->set_output(0, c->input(0));
      return OkStatus();
    });

REGISTER_OP("FMHABackward")
    .Input("query: dtype")
    .Input("key: dtype")
    .Input("value: dtype")
    .Input("cu_seqlens_q: int32")
    .Input("cu_seqlens_k: int32")
    .Input("fwd_out: dtype")
    .Input("gradient: dtype")
    .Input("softmax_lse: float")
    .Input("max_seqlen_q: int32")
    .Input("max_seqlen_k: int32")
    .Input("rng_state: uint64")
    .Output("dq: dtype")
    .Output("dk: dtype")
    .Output("dv: dtype")
    // .Output("return_sm: dtype")
    .Attr("dtype: type")
    .Attr("p_dropout: float")
    .Attr("softmax_scale: float")
    .Attr("zero_tensors: bool")
    .Attr("is_causal: bool")
    .Attr("return_softmax: bool")
    .Attr("num_splits: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      return OkStatus();
    });

