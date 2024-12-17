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

#include <Python.h>

static PyObject *original_excepthook = NULL;

static void xpuTimerprintExecption() {
  if (PyErr_Occurred()) {
    PyObject *exc_type, *exc_value, *exc_traceback;
    PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
    PyErr_Display(exc_type, exc_value, exc_traceback);

    Py_XDECREF(exc_type);
    Py_XDECREF(exc_value);
    Py_XDECREF(exc_traceback);

    PyErr_Clear();
  }
}

static void runPlugin(PyObject *exc_type, PyObject *exc_value,
                      PyObject *exc_traceback) {
  PyObject *main_module = PyImport_ImportModule("py_xpu_timer.run_plugin");
  if (!main_module) {
    xpuTimerprintExecption();
    return;
  }
  PyObject *func =
      PyObject_GetAttrString(main_module, "xpu_timer_parse_python_exception");
  if (func && PyCallable_Check(func)) {
    PyObject *result = PyObject_CallFunctionObjArgs(func, exc_type, exc_value,
                                                    exc_traceback, NULL);
    if (result == NULL) {
      xpuTimerprintExecption();
      Py_XDECREF(func);
      Py_DECREF(main_module);
      return;
    }
    Py_DECREF(result);
  } else {
    printf(
        "load xpu_timer_parse_exception from py_xpu_timer.boot_sys_hook error, "
        "never here\n");
    return;
  }
  Py_XDECREF(func);
  Py_DECREF(main_module);
}

static PyObject *myExceptHook(PyObject *self, PyObject *args) {
  PyObject *exc_type, *exc_value, *exc_traceback;

  if (!PyArg_ParseTuple(args, "OOO", &exc_type, &exc_value, &exc_traceback))
    return NULL;
  runPlugin(exc_type, exc_value, exc_traceback);
  PyErr_Display(exc_type, exc_value, exc_traceback);
  Py_RETURN_NONE;
}

static PyMethodDef excepthook_method = {"myExceptHook", myExceptHook,
                                        METH_VARARGS, "Custom excepthook"};

void setSysHook() {
  PyObject *excepthook_func = PyCFunction_New(&excepthook_method, NULL);
  if (excepthook_func == NULL) {
    xpuTimerprintExecption();
    return;
  }
  original_excepthook = PySys_GetObject("excepthook");
  if (original_excepthook != NULL) {
    Py_INCREF(original_excepthook);
  }

  PySys_SetObject("excepthook", excepthook_func);
  Py_DECREF(excepthook_func);

  if (original_excepthook != NULL) {
    Py_DECREF(original_excepthook);
  }
}
