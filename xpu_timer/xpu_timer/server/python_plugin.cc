// Copyright 2024 The DLRover Authors. All rights reserved.
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

#include "xpu_timer/server/python_plugin.h"

#include <Python.h>

#include "xpu_timer/common/logging.h"

namespace xpu_timer {
namespace server {
using namespace util;

void xpuTimerDaemonSetUpPython() {
  Py_Initialize();
  PyEval_InitThreads();
  PyRun_SimpleString("print('xpu_timer_daemon init python')");
  PyObject* sys_module = PyImport_ImportModule("sys");
  PyObject* io_module = PyImport_ImportModule("io");
  // we close all stdout and stderr, otherwise c api will raise broken pipe
  PyObject* devnull =
      PyObject_CallMethod(io_module, "open", "ss", "/dev/null", "w");

  if (PyObject_SetAttrString(sys_module, "stdout", devnull) < 0) PyErr_Print();

  if (PyObject_SetAttrString(sys_module, "stderr", devnull) < 0) PyErr_Print();

  // Clean up
  Py_DECREF(devnull);
  Py_DECREF(sys_module);
  Py_DECREF(io_module);

  PyEval_ReleaseThread(PyThreadState_Get());
}

static PyObject* convertFrameInfoToDict(
    const SignalFrameRequest::FrameInfo& frame_info_proto) {
  PyObject* frame_info_dict = PyDict_New();
  if (!frame_info_dict) return NULL;

  // Set "address" field
  PyObject* address = PyLong_FromUnsignedLongLong(frame_info_proto.address());
  PyDict_SetItemString(frame_info_dict, "address", address);
  Py_DECREF(address);

  // Set "offset" field
  PyObject* offset = PyLong_FromUnsignedLongLong(frame_info_proto.offset());
  PyDict_SetItemString(frame_info_dict, "offset", offset);
  Py_DECREF(offset);

  // Set "function_name" field
  PyObject* function_name =
      PyUnicode_FromString(frame_info_proto.function_name().c_str());

  PyDict_SetItemString(frame_info_dict, "function_name", function_name);
  Py_DECREF(function_name);

  return frame_info_dict;
}
static PyObject* convertSignalFrameRequestToDict(
    const SignalFrameRequest* signal_frame_proto) {
  PyObject* signal_frame_dict = PyDict_New();
  if (!signal_frame_dict) return NULL;

  // Set "signal" field
  PyObject* signal = PyLong_FromLong(signal_frame_proto->signal());
  PyDict_SetItemString(signal_frame_dict, "signal", signal);
  Py_DECREF(signal);

  // Set "rank" field
  PyObject* rank = PyLong_FromLong(signal_frame_proto->rank());
  PyDict_SetItemString(signal_frame_dict, "rank", rank);
  Py_DECREF(rank);

  // Set "ip" field
  PyObject* ip = PyUnicode_FromString(signal_frame_proto->ip().c_str());
  PyDict_SetItemString(signal_frame_dict, "ip", ip);
  Py_DECREF(ip);

  // Set "pod_name" field
  PyObject* pod_name =
      PyUnicode_FromString(signal_frame_proto->pod_name().c_str());
  PyDict_SetItemString(signal_frame_dict, "pod_name", pod_name);
  Py_DECREF(pod_name);

  // Set "job_name" field
  PyObject* job_name =
      PyUnicode_FromString(signal_frame_proto->job_name().c_str());
  PyDict_SetItemString(signal_frame_dict, "job_name", job_name);
  Py_DECREF(job_name);

  // Set "timestamp" field
  PyObject* timestamp =
      PyLong_FromUnsignedLongLong(signal_frame_proto->timestamp());
  PyDict_SetItemString(signal_frame_dict, "timestamp", timestamp);
  Py_DECREF(timestamp);

  // Set "frame_infos" field as a list of dictionaries
  PyObject* frame_infos_list = PyList_New(0);
  for (const auto& frame_info_proto : signal_frame_proto->frame_infos()) {
    PyObject* frame_info_dict = convertFrameInfoToDict(frame_info_proto);
    if (!frame_info_dict) {
      Py_DECREF(frame_infos_list);
      Py_DECREF(signal_frame_dict);
      return NULL;
    }
    PyList_Append(frame_infos_list, frame_info_dict);
    Py_DECREF(frame_info_dict);
  }
  PyDict_SetItemString(signal_frame_dict, "frame_infos", frame_infos_list);
  Py_DECREF(frame_infos_list);

  return signal_frame_dict;
}

void logPythonError(int rank, const char* context) {
  PyObject *ptype, *pvalue, *ptraceback;
  PyErr_Fetch(&ptype, &pvalue, &ptraceback);
  if (ptype != NULL) {
    PyObject* ptype_str = PyObject_Str(ptype);
    const char* ptype_cstr = PyUnicode_AsUTF8(ptype_str);
    PyObject* pvalue_str = PyObject_Str(pvalue);
    const char* pvalue_cstr = PyUnicode_AsUTF8(pvalue_str);
    XLOG(INFO) << "rank " << rank << " error type " << ptype_cstr << " where "
               << context;
    XLOG(INFO) << "rank " << rank << " error " << pvalue_cstr << " where "
               << context;
    ;

    Py_XDECREF(ptype_str);
    Py_XDECREF(pvalue_str);
  } else {
    XLOG(INFO) << "rank " << rank << " ok" << " where " << context;
    ;
  }

  PyErr_Clear();

  Py_XDECREF(ptype);
  Py_XDECREF(pvalue);
  Py_XDECREF(ptraceback);
}

void runPythonPlugin(const SignalFrameRequest* request) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* main_module = PyImport_ImportModule("py_xpu_timer.run_plugin");
  if (!main_module) {
    logPythonError(request->rank(), "import");
    PyGILState_Release(gstate);
    return;
  }
  XLOG(INFO) << " rank " << request->rank() << " import ok";
  PyObject* signal_frame_dict = convertSignalFrameRequestToDict(request);
  PyObject* func =
      PyObject_GetAttrString(main_module, "xpu_timer_parse_cpp_exception");
  if (func && PyCallable_Check(func)) {
    PyObject* result =
        PyObject_CallFunctionObjArgs(func, signal_frame_dict, NULL);
    if (result == NULL) {
      Py_XDECREF(func);
      XLOG(INFO) << " rank " << request->rank()
                 << " call xpu_timer_parse_cpp_exception err";
      logPythonError(request->rank(), "call fn");
      Py_DECREF(main_module);
      PyGILState_Release(gstate);
      return;
    }
    XLOG(INFO) << " rank " << request->rank()
               << " call xpu_timer_parse_cpp_exception ok";
    Py_DECREF(result);
  } else {
    XLOG(INFO) << "load xpu_timer_parse_exception from "
                  "py_xpu_timer.boot_sys_hook error, never here";
    logPythonError(request->rank(), "never here");
    Py_DECREF(main_module);
    Py_DECREF(signal_frame_dict);
    PyGILState_Release(gstate);
    return;
  }
  Py_XDECREF(func);
  Py_DECREF(main_module);
  Py_DECREF(signal_frame_dict);
  PyGILState_Release(gstate);
}
}  // namespace server
}  // namespace xpu_timer
