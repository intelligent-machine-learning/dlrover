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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeAndType;
using ::tensorflow::shape_inference::ShapeHandle;
using ::tensorflow::shape_inference::DimensionHandle;
using namespace tensorflow;  // NOLINT(build/namespaces)

namespace {
Status ScalarOutput(InferenceContext* c) {
  c->set_output(0, c->Scalar());
  return ::tensorflow::OkStatus();
}

Status KvVariableScatterUpdateShape(InferenceContext* c) {
  return ::tensorflow::OkStatus();
}
}  // namespace

REGISTER_OP("KvVariable")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("key_shape: shape = {}")
    .Attr("value_shape: shape")
    // The followng attributes are subject to changes in the future
    .Attr("enter_threshold: int = 0")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      // Set the output
      c->set_output(0, c->Scalar());

      // Set the handle shape and type
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("value_dtype", &dtype));

      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("value_shape", &shape));

      if (shape.dims() == 0) {
        return shape_inference::UnknownShape(c);
      }
      // Insert one dim as the first dim
      shape.InsertDim(0, InferenceContext::kUnknownDim);

      // Form the final shape
      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(
          c->MakeShapeFromPartialTensorShape(shape, &output_shape));
      c->set_output_handle_shapes_and_types(
          0, std::vector<ShapeAndType>{{output_shape, dtype}});

      return ::tensorflow::OkStatus();
    });

// KvVariableV2 is only compatiable with our old version.
// It will only be used to load model for serving.
REGISTER_OP("KvVariableV2")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("key_shape: shape = {}")
    .Attr("value_shape: shape")
    .Attr("initial_num_buckets: int = 131072")  // 2^17
    .Attr("max_load_factor: float = 0.8")
    /*The followng attributes are subject to changes in the future*/
    .Attr("enter_threshold: int = 5")
    .Attr("total_iteration: int = 100000")
    .Attr("worker_num: int = 16")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      /*set the output*/
      c->set_output(0, c->Scalar());

      /*set the handle shape and type*/
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("value_dtype", &dtype));

      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("value_shape", &shape));

      if (shape.dims() == 0) {
        printf("Unknown shape for KvVariableV2\n");
        return shape_inference::UnknownShape(c);
      }
      /*insert one dim as the first dim*/
      shape.InsertDim(0, InferenceContext::kUnknownDim);

      /*form the final shape*/
      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(
          c->MakeShapeFromPartialTensorShape(shape, &output_shape));
      c->set_output_handle_shapes_and_types(
          0, std::vector<ShapeAndType>{{output_shape, dtype}});

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableV3")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("key_shape: shape = {}")
    .Attr("value_shape: shape")
    // The followng attributes are subject to changes in the future
    .Attr("enter_threshold: int = 0")
    .Attr("phstore_path: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      // Set the output
      c->set_output(0, c->Scalar());

      // Set the handle shape and type
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("value_dtype", &dtype));

      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("value_shape", &shape));

      if (shape.dims() == 0) {
        return shape_inference::UnknownShape(c);
      }
      // Insert one dim as the first dim
      shape.InsertDim(0, InferenceContext::kUnknownDim);

      // Form the final shape
      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(
          c->MakeShapeFromPartialTensorShape(shape, &output_shape));
      c->set_output_handle_shapes_and_types(
          0, std::vector<ShapeAndType>{{output_shape, dtype}});

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableV4")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("key_shape: shape = {}")
    .Attr("value_shape: shape")
    .Attr("storage_option: string")
    // The followng attributes are subject to changes in the future
    .Attr("enter_threshold: int = 0")
    // default value is mem
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      // Set the output
      c->set_output(0, c->Scalar());

      // Set the handle shape and type
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("value_dtype", &dtype));

      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("value_shape", &shape));

      if (shape.dims() == 0) {
        return shape_inference::UnknownShape(c);
      }
      // Insert one dim as the first dim
      shape.InsertDim(0, InferenceContext::kUnknownDim);

      // Form the final shape
      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(
          c->MakeShapeFromPartialTensorShape(shape, &output_shape));
      c->set_output_handle_shapes_and_types(
          0, std::vector<ShapeAndType>{{output_shape, dtype}});

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableShapeV2")
    .Input("table_handle: resource")
    .Output("output: out_type")
    .Attr("out_type: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      return ::tensorflow::OkStatus();
    });

REGISTER_OP("InitKvVariableV2")
    .Input("table_handle: resource")
    .Input("random_initializer: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      // The second input must be 2-D
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &handle));

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableIsInitializedV2")
    .Input("table_handle: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(ScalarOutput);

REGISTER_OP("KvVariableSizeV2")
    .Input("table_handle: resource")
    .Output("size: T")
    .Attr("T: {int32, int64} = DT_INT64")
    .SetShapeFn(ScalarOutput);

REGISTER_OP("KvVariableSizeV3")
    .Input("table_handle: resource")
    .Output("sizes: T")
    .Attr("T: {int32, int64} = DT_INT64");

REGISTER_OP("KvVariableFrequency")
    .Input("table_handle: resource")
    .Output("size: T")
    .Attr("T: {int32, int64} = DT_INT64")
    .SetShapeFn(ScalarOutput);

REGISTER_OP("ReadKvVariableOpV2")
    .Input("table_handle: resource")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 1, &values));
      ShapeHandle keys = c->Vector(c->Dim(values, 0));
      c->set_output(0, keys);
      c->set_output(1, values);
      return ::tensorflow::OkStatus();
    });

REGISTER_OP("DestroyKvVariableOpV2")
    .Input("table_handle: resource")
    .Attr("ignore_lookup_error: bool = true")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

// KvVariableGatherV2 is only compatiable with our old version.
// It will only be used to load model for serving.
REGISTER_OP("KvVariableGatherV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Input("use_init_value: bool")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32, int64, uint64}")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableGatherOrZerosV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("BatchKvVariableGatherOrZerosV2")
    .Input("table_handles: N * resource")
    .Input("indices: N * Tindices")
    .Output("output: N * dtype")
    .Attr("N: int >= 1")
    .Attr("dtype: type")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableGatherOrInsertV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableGatherOrInsertWithCounts")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Input("counts: int32")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableInsertV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Input("values: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn([](InferenceContext*) { return ::tensorflow::OkStatus(); });

REGISTER_OP("KvVariableIncreaseCountV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Input("counts: int32")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn([](InferenceContext*) { return ::tensorflow::OkStatus(); });

REGISTER_OP("KvVariableGetCountV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Output("output: dtype")
    .Attr("dtype: {int32} = DT_INT32")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return ::tensorflow::OkStatus();
    });


REGISTER_OP("KvVariableImport")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Input("init_table: Tout")
    .Input("blacklist: Tin")
    .Input("freq_keys: Tin")
    .Input("freq_values: uint16")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .Attr("first_n: int=6")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      // Check first n
      int first_n;
      TF_RETURN_IF_ERROR(c->GetAttr("first_n", &first_n));
      if (first_n < 3 || first_n > 6) {
        // return ::tensorflow::errors::InvalidArgument(
        //     "For KvVariable importing, first_n must be in [3, 6]");
      }

      return ::tensorflow::OkStatus();
    });

// KvVariableImportV2 and KvVariableImportV3 are only compatiable
// with our old version. There will only be used to load model for serving.
REGISTER_OP("KvVariableImportV2")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Input("init_table: Tout")
    .Input("blacklist: Tin")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableImportV3")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Input("init_table: Tout")
    .Input("blacklist: Tin")
    .Input("freq_keys: Tin")
    .Input("freq_values: uint8")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableExport")
    .Input("table_handle: resource")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Output("init_table: Tvalues")
    .Output("blacklist: Tkeys")
    .Output("freq_keys: Tkeys")
    .Output("freq_values: uint16")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .Attr("enable_cutoff: bool = false")
    .Attr("cutoff_value: float = 0.0")
    .Attr("first_n: int=3")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 2, &values));
      ShapeHandle keys = c->Vector(c->Dim(values, 0));
      c->set_output(0, keys);
      c->set_output(1, values);

      // Check first n
      int first_n;
      TF_RETURN_IF_ERROR(c->GetAttr("first_n", &first_n));
      if (first_n < 2 || first_n > 6) {
        // return ::tensorflow::errors::InvalidArgument(
        //     "For KvVariable exporting, first_n must be in [2, 6]");
      }
      // Initialization table
      c->set_output(2, values);

      // Blacklist
      c->set_output(3, c->Vector(c->UnknownDim()));

      // Frequency table
      c->set_output(4, c->Vector(c->UnknownDim()));
      c->set_output(5, c->Vector(c->UnknownDim()));

      return ::tensorflow::OkStatus();
    });

// KvVariableExportV2 and KvVariableExportV3 are only compatiable
// with our old version. There will only be used when do tf serving.
REGISTER_OP("KvVariableExportV2")
    .Input("table_handle: resource")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Output("init_table: Tvalues")
    .Output("blacklist: Tkeys")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .Attr("enable_cutoff: bool = false")
    .Attr("cutoff_value: float = 0.0")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 2, &values));
      ShapeHandle keys = c->Vector(c->Dim(values, 0));
      c->set_output(0, keys);
      c->set_output(1, values);
      c->set_output(2, values);
      c->set_output(3, c->Vector(c->UnknownDim()));

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableExportV3")
    .Input("table_handle: resource")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Output("init_table: Tvalues")
    .Output("blacklist: Tkeys")
    .Output("freq_keys: Tkeys")
    .Output("freq_values: uint8")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .Attr("enable_cutoff: bool = false")
    .Attr("cutoff_value: float = 0.0")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 2, &values));
      ShapeHandle keys = c->Vector(c->Dim(values, 0));
      c->set_output(0, keys);
      c->set_output(1, values);
      c->set_output(2, values);
      c->set_output(3, c->Vector(c->UnknownDim()));
      c->set_output(4, c->Vector(c->UnknownDim()));
      c->set_output(5, c->Vector(c->UnknownDim()));

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableScatterAddV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn(KvVariableScatterUpdateShape);

REGISTER_OP("KvVariableScatterSubV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn(KvVariableScatterUpdateShape);

REGISTER_OP("KvVariableScatterMulV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn(KvVariableScatterUpdateShape);

REGISTER_OP("KvVariableScatterDivV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn(KvVariableScatterUpdateShape);

REGISTER_OP("KvVariableScatterMinV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn(KvVariableScatterUpdateShape);

REGISTER_OP("KvVariableScatterMaxV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn(KvVariableScatterUpdateShape);

REGISTER_OP("KvVariableScatterUpdateV2")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn(KvVariableScatterUpdateShape);

REGISTER_OP("KvVariableFullOrDeltaImport")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Input("init_table: Tout")
    .Input("blacklist: Tin")
    .Input("freq_keys: Tin")
    .Input("freq_values: uint32")
    .Input("need_full_import: bool")
    .Input("delete_keys: Tin")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .Attr("first_n: int=6")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      // Check first n
      int first_n;
      TF_RETURN_IF_ERROR(c->GetAttr("first_n", &first_n));
      if (first_n < 3 || first_n > 8) {
        // return ::tensorflow::errors::InvalidArgument(
        //     "For KvVariable importing, first_n must be in [3, 8]");
      }

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableFullOrDeltaImportV2")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Input("init_table: Tout")
    .Input("blacklist: Tin")
    .Input("freq_keys: Tin")
    .Input("freq_values: uint32")
    .Input("need_full_import: bool")
    .Input("delete_keys: Tin")
    .Input("is_loading_finished: bool")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .Attr("first_n: int=6")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      // Check first n
      int first_n;
      TF_RETURN_IF_ERROR(c->GetAttr("first_n", &first_n));
      if (first_n < 3 || first_n > 8) {
        // return ::tensorflow::errors::InvalidArgument(
        //     "For KvVariable importing, first_n must be in [3, 8]");
      }

      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableFullOrDeltaExport")
    .Input("table_handle: resource")
    .Input("do_full_export: bool")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Output("init_table: Tvalues")
    .Output("blacklist: Tkeys")
    .Output("freq_keys: Tkeys")
    .Output("freq_values: uint32")
    .Output("need_full_import: bool")
    .Output("delete_keys: Tkeys")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .Attr("enable_cutoff: bool = false")
    .Attr("cutoff_value: float = 0.0")
    .Attr("first_n: int=3")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 2, &values));
      ShapeHandle keys = c->Vector(c->Dim(values, 0));
      c->set_output(0, keys);
      c->set_output(1, values);

      // Check first n
      int first_n;
      TF_RETURN_IF_ERROR(c->GetAttr("first_n", &first_n));
      if (first_n < 2 || first_n > 8) {
        // return ::tensorflow::errors::InvalidArgument(
        //     "For KvVariable exporting, first_n must be in [2, 8]");
      }
      // Initialization table
      c->set_output(2, values);

      // Blacklist
      c->set_output(3, c->Vector(c->UnknownDim()) );

      // Frequency table
      c->set_output(4, c->Vector(c->UnknownDim()));
      c->set_output(5, c->Vector(c->UnknownDim()));

      c->set_output(6, c->Vector(1));
      c->set_output(7, c->Vector(c->UnknownDim()));
      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableDelete")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn([](InferenceContext*) { return ::tensorflow::OkStatus(); });

REGISTER_OP("KvVariableGetTimeStamp")
    .Input("table_handle: resource")
    .Input("indices: Tindices")
    .Output("output: dtype")
    .Attr("dtype: {uint32} = DT_UINT32")
    .Attr("Tindices: {int32, int64, uint64, string}")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return ::tensorflow::OkStatus();
    });

REGISTER_OP("KvVariableDeleteWithTimestamp")
    .Input("table_handle: resource")
    .Output("delete_keys: Tkeys")
    .Attr("Tkeys: type")
    .Attr("threshold: int=7")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return ::tensorflow::OkStatus();
    });

REGISTER_OP("SaveV3")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Input("first_n: int32")
    .Input("do_full_export: bool")
    .Input("tensors: dtypes")
    .Attr("freq_use_uint32: bool = false")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle s;
      DimensionHandle unused_dim;

      // Validate prefix.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      // Validate tensor_names and shapes_and_slices.
      for (int i = 1; i <= 2; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &s));
        TF_RETURN_IF_ERROR(
            c->WithValue(c->Dim(s, 0), c->num_inputs() - 5, &unused_dim));
      }
      return ::tensorflow::OkStatus();
    });
