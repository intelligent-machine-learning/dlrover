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
#include <frameobject.h>

#include <string>
#include <vector>

struct stack {
  std::string filename;
  std::string function_name;
  int line;
};

PyCodeObject* getCodeOfFrame(PyFrameObject* frame);
PyFrameObject* getFrameFromThreadState(PyThreadState* tstate);

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 11
#include <internal/pycore_frame.h>
#include <pyframe.h>
#include <pystate.h>

PyCodeObject* getCodeOfFrame(PyFrameObject* frame) {
  return PyFrame_GetCode(frame);
}

PyFrameObject* getFrameFromThreadState(PyThreadState* tstate) {
  return PyThreadState_GetFrame(tstate);
}

#else

PyFrameObject* getFrameFromThreadState(PyThreadState* tstate) {
  return tstate->frame;
}

PyCodeObject* getCodeOfFrame(PyFrameObject* frame) { return frame->f_code; }

#endif

extern "C" {

std::vector<stack> gather_python_callstack() {
  // this function is used for dlopen in libevent_hook.so,
  // not direct link agaist to libevent_hook.so
  // not thread safe, assume launch kernel is single thread
  std::vector<stack> call_stack;

  if (!Py_IsInitialized()) {
    return call_stack;
  }
  // check has gil, avoid recursive lock
  if (!PyGILState_Check()) return call_stack;

  PyThreadState* TState = PyThreadState_Get();

  for (PyFrameObject* frame = getFrameFromThreadState(TState); frame != NULL;
       frame = frame->f_back) {
    stack frame_info;
    int line = PyFrame_GetLineNumber(frame);
    PyCodeObject* code = getCodeOfFrame(frame);
    PyObject* filename =
        PyUnicode_FromString(PyUnicode_AsUTF8(code->co_filename));
    if (filename) {
      frame_info.filename = PyUnicode_AsUTF8(filename);
      Py_DECREF(filename);
    }

    PyObject* name = PyUnicode_FromString(PyUnicode_AsUTF8(code->co_name));
    if (name) {
      frame_info.function_name = PyUnicode_AsUTF8(name);
      Py_DECREF(name);
    }
    frame_info.line = line;
    call_stack.push_back(frame_info);
  }

  return call_stack;
}
}
