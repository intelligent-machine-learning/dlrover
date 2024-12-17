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

#include "xpu_timer/python/py_tracing.h"

#include <Python.h>
#include <frameobject.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>

#include "uthash.h"
#include "xpu_timer/python/py_tracing_data.h"

uint64_t getCodeOfFrame(PyFrameObject* frame);
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 11
#include <pyframe.h>
uint64_t getCodeOfFrame(PyFrameObject* frame) {
  return (int64_t)(uintptr_t)PyFrame_GetCode(frame);
}
#else
uint64_t getCodeOfFrame(PyFrameObject* frame) {
  return (int64_t)(uintptr_t)(frame->f_code);
}
#endif

// clang-format off
/*
                               Working flow of python tracing and basic datasturcure

                                                          XpuTimerPyTracingData
                                                              ┌──────┐
                                                       define │start │
                                                       ┌─────►│end   │
                                                       │      └──────┘
                                                       │                                   xpu_timer::KernelTraceManager
                                                       │ XpuTimerPyTracingDataArray           ┌─────────────────┐
                                                      ┌┴────┬─────┬─────┬─────┬─────────┐     │                 │
                                               define │start│start│start│start│ current │     │  ┌───────┐ tag  │
                                              ┌──────►│end  │end  │end  │end  │ index   │  tag│  │ func  ├──────┼──┐
                                              │       ├─────┴─────┴─────┴─────┼─────────┘ ┌───┼─►│tracing│      │  │
                                              │       └─────512 data total────┘           │   │  └───────┘      │  │
                                              │                                           │   │                 │  │
                                              │                                           │   │                 │  │
                                          ┌───┼────────────────────────────┐              │   │                 │  │
                                          │   │tracing data buffer pool    │  3. request  │   │  ┌───────┐      │  │
          2. pushing tracing              │  ┌┴─────┬──────┬──────┬──────┐ │      data    │tag│  │  gc   │ tag  │  │
      ┌───────────────────────────────────┼─►│data  │data  │data  │data  ├─│──────────────┴───┼─►│tracing┼──────┼──┤
      │                                   │  │array │array │array │array │ │                  │  └───────┘      │  │
      │                                   │  └──────┴──────┴──────┴──────┘ │                  └─────────────────┘  │
      │                                   │                                │                                       │
      │  ┌─────────────────┐              │      empty data buffer pool    │                                       │
      │  │  ┌───────┐      │ 1. get empty │  ┌──────┬──────┬──────┬──────┐ │        4. return tracing buffer       │
      │ tag │ func  │  tag │  buffer      │  │data  │data  │data  │data  ├◄┼───────────────────────────────────────┘
      ├──┼──┼tracing│◄─────┼───┬──────────┼──┼array │array │array │array │ │
      │  │  └───────┘      │   │          │  └──────┴──────┴──────┴──────┘ │
      │  │                 │   │          └────────────────────────────────┘
      │  │ current tracing │   │             xpu_timer::py_tracing_manager::PyTracingManager
      │  │  data array     │   │
      │  │                 │   │
      │  │  ┌───────┐      │   │
      │ tag │  gc   │  tag │   │
      └──┼──┼tracing│◄─────┼───┘
         │  └───────┘      │
         └─────────────────┘
        NamedTracingFunction
*/
// clang-format on

// hashtable entry for each python functions
typedef struct {
  int64_t py_code_address;  // use to check is target python function or not
  const char* function_name;
  int tag_name;
  int is_native;
  UT_hash_handle hh;
} NamedTracingFunction;

typedef struct {
  int tag_name;
  XpuTimerPyTracingDataArray* curr_data;
  int64_t count;              // how many calls occured
  const char* function_name;  // for debug
} NamedTracingData;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// this is array indexed by tag name.
static NamedTracingData* named_tracing_data_array = NULL;
// this is a hash map from uthash, for tracing python functions
// eg. torch.utils.data.dataloader._BaseDataLoaderIter.__next__
static NamedTracingFunction* named_tracing_func_map = NULL;
// flag for control tracing
static int start_tracing = 0;
// total tags
static int tracing_data_count = 0;

// use for get address of __code__ for traced function
// use PyRun_String to run python script to get id
static int runPythonCodeGetAddress(const char* input, char** error_message,
                                   int64_t* code_address, int* is_native);
static uint64_t getMicrosecondTimestamp();
static NamedTracingFunction* isTracedPythonFunction(PyFrameObject* frame);
static NamedTracingData* getTracingData(int name);
static void addTracingData(int name, const char* func_name);
// python profiler interface
// this function is not compatiable with other profiler
static int profiler(PyObject* obj, PyFrameObject* frame, int what,
                    PyObject* arg);

void setSysHook();

uint64_t getMicrosecondTimestamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (uint64_t)tv.tv_sec * 1000000 + (uint64_t)tv.tv_usec;
}

NamedTracingFunction* isTracedPythonFunction(PyFrameObject* frame) {
  uint64_t code_address = getCodeOfFrame(frame);
  NamedTracingFunction* traced_function = NULL;
  HASH_FIND(hh, named_tracing_func_map, &code_address, sizeof(int64_t),
            traced_function);
  return traced_function;
}

// profiler entry for PyEval_SetProfile
static int profiler(PyObject* obj, PyFrameObject* frame, int what,
                    PyObject* arg) {
  NamedTracingFunction* func_data = isTracedPythonFunction(frame);
  if (!func_data) return 0;
  int tag_name = func_data->tag_name;
  if ((what == PyTrace_CALL) && start_tracing) {
    pthread_mutex_lock(&mutex);
    NamedTracingData* tracing_data = getTracingData(tag_name);
    XpuTimerPyTracingDataArray* curr_data = tracing_data->curr_data;
    if (curr_data->cur == PY_TRACING_BUFFER_SIZE) {
      xpu_timer_return_py_tracing_data_array(curr_data, PY_TRACING_READY_POOL,
                                             tag_name);
      tracing_data->curr_data =
          xpu_timer_get_empty_py_tracing_data_array(tag_name);
      curr_data = tracing_data->curr_data;
    }
    curr_data->data[curr_data->cur].start = getMicrosecondTimestamp();
    pthread_mutex_unlock(&mutex);
  } else if (what == PyTrace_RETURN) {
    pthread_mutex_lock(&mutex);
    NamedTracingData* tracing_data = getTracingData(tag_name);
    if (start_tracing) {
      XpuTimerPyTracingDataArray* curr_data = tracing_data->curr_data;
      curr_data->data[curr_data->cur].count = tracing_data->count;
      curr_data->data[curr_data->cur++].end = getMicrosecondTimestamp();
    }
    tracing_data->count++;
    pthread_mutex_unlock(&mutex);
  }
  return 0;
}

int runPythonCodeGetAddress(const char* code, char** error_message,
                            int64_t* address, int* is_native) {
  char* input = strdup(code);
  char python_code[4096];
  PyObject* globals = NULL;
  PyObject* locals = NULL;

  *error_message = NULL;

  snprintf(python_code, sizeof(python_code),
           "if '@' in '%s':\n"
           "    tokens = '%s'.split('@')\n"
           "    if len(tokens) == 3:\n"
           "        exec(f'from {tokens[0]} import {tokens[1]} as mm')\n"
           "        obj = getattr(mm, tokens[2])\n"
           "    elif len(tokens) == 2:\n"
           "        exec(f'from {tokens[0]} import {tokens[1]} as obj')\n"
           "    else:\n"
           "        raise ValueError('Invalid input format')\n"
           "else:\n"
           "    obj = globals().get('%s')\n"
           "    if obj is None:\n"
           "        raise ValueError('Global object not found: %s')\n"

           "while hasattr(obj, '__wrapped__'):\n"
           "    obj = getattr(obj, '__wrapped__')\n"
           "if hasattr(obj, '__code__'):\n"
           "    address = id(obj.__code__)\n"
           "    is_native = 0\n"
           "else:\n"
           "    address = id(obj)\n"
           "    is_native = 1\n",
           input, input, input, input);
  int use_globals = strchr(input, '@') == NULL;
  if (use_globals) {
    globals = PyEval_GetGlobals();
    locals = PyEval_GetLocals();
  } else {
    globals = PyDict_New();
    locals = PyDict_New();
  }

  PyObject* result = PyRun_String(python_code, Py_file_input, globals, locals);

  if (result == NULL) {
    if (PyErr_Occurred()) {
      PyObject *ptype, *pvalue, *ptraceback;
      PyErr_Fetch(&ptype, &pvalue, &ptraceback);
      PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);

      PyObject* py_str = PyObject_Str(pvalue);
      const char* str_error = PyUnicode_AsUTF8(py_str);
      *error_message = strdup(str_error ? str_error : "Unknown error");

      Py_XDECREF(py_str);
      Py_XDECREF(ptype);
      Py_XDECREF(pvalue);
      Py_XDECREF(ptraceback);
    } else {
      *error_message = strdup("Unknown error occurred");
    }
    PyErr_Clear();
    if (!use_globals) {
      Py_DECREF(globals);
      Py_DECREF(locals);
    }
    free(input);
    return 1;
  }

  *address = PyLong_AsLongLong(PyDict_GetItemString(locals, "address"));
  *is_native = PyLong_AsLongLong(PyDict_GetItemString(locals, "is_native"));

  if (!use_globals) {
    Py_DECREF(globals);
    Py_DECREF(locals);
  }
  free(input);

  size_t msg_size =
      snprintf(NULL, 0, "Get __code__ attribute for '%s' OK", code) + 1;
  *error_message = (char*)malloc(msg_size);
  snprintf(*error_message, msg_size, "Get __code__ attribute for '%s' OK",
           code);
  return 0;
}

static NamedTracingData* getTracingData(int name) {
  return named_tracing_data_array + name;
}

static void addTracingData(int name, const char* func_name) {
  NamedTracingData* v = getTracingData(name);
  v->tag_name = name;
  v->curr_data = xpu_timer_get_empty_py_tracing_data_array(name);
  v->function_name = strdup(func_name);
}

static void getGcInfo(XpuTimerPyTracingData* data, PyObject* info) {
  if (!PyDict_Check(info)) return;
  PyObject* collected = PyDict_GetItemString(info, "collected");
  PyObject* uncollectable = PyDict_GetItemString(info, "uncollectable");

  if (collected && PyLong_Check(collected)) {
    data->payload.gc_debug[0] = PyLong_AsLong(collected);
  } else {
    data->payload.gc_debug[0] = -1;
  }

  if (uncollectable && PyLong_Check(uncollectable)) {
    data->payload.gc_debug[1] = PyLong_AsLong(uncollectable);
  } else {
    data->payload.gc_debug[1] = -1;
  }
}

static void gcCallback(PyObject* phase, PyObject* info) {
  // this is in gil, so other thread MUST NOT to lock and get gil
  pthread_mutex_lock(&mutex);
  if (PyUnicode_CompareWithASCIIString(phase, "start") == 0 && start_tracing) {
    NamedTracingData* tracing_data = getTracingData(PY_TRACING_GC);
    XpuTimerPyTracingDataArray* curr_data = tracing_data->curr_data;
    if (curr_data->cur == PY_TRACING_BUFFER_SIZE) {
      xpu_timer_return_py_tracing_data_array(curr_data, PY_TRACING_READY_POOL,
                                             PY_TRACING_GC);
      tracing_data->curr_data =
          xpu_timer_get_empty_py_tracing_data_array(PY_TRACING_GC);
      curr_data = tracing_data->curr_data;
    }
    curr_data->data[curr_data->cur].start = getMicrosecondTimestamp();
    pthread_mutex_unlock(&mutex);
  } else if (PyUnicode_CompareWithASCIIString(phase, "stop") == 0) {
    NamedTracingData* tracing_data = getTracingData(PY_TRACING_GC);
    if (start_tracing) {
      XpuTimerPyTracingDataArray* curr_data = tracing_data->curr_data;
      if (start_tracing) {
        curr_data->data[curr_data->cur].count = tracing_data->count;
        curr_data->data[curr_data->cur].type = PAYLOAD_GC;
        getGcInfo(curr_data->data + curr_data->cur, info);
        curr_data->data[curr_data->cur++].end = getMicrosecondTimestamp();
      }
      curr_data->data[curr_data->cur].count = tracing_data->count;
      curr_data->data[curr_data->cur++].end = getMicrosecondTimestamp();
    }
    tracing_data->count++;
  }
  pthread_mutex_unlock(&mutex);
}

static PyObject* gcCallbackWrapper(PyObject* self, PyObject* args,
                                   PyObject* kwargs) {
  PyObject *phase, *info;
  if (!PyArg_ParseTuple(args, "OO", &phase, &info)) {
    return NULL;
  }
  gcCallback(phase, info);
  Py_RETURN_NONE;
}

static PyTypeObject GcCallbackType = {
    PyVarObject_HEAD_INIT(NULL, 0) "gc_callback", /* tp_name */
    sizeof(PyObject),                             /* tp_basicsize */
    0,                                            /* tp_itemsize */
    0,                                            /* tp_dealloc */
    0,                                            /* tp_vectorcall_offset */
    0,                                            /* tp_getattr */
    0,                                            /* tp_setattr */
    0,                                            /* tp_as_async */
    0,                                            /* tp_repr */
    0,                                            /* tp_as_number */
    0,                                            /* tp_as_sequence */
    0,                                            /* tp_as_mapping */
    0,                                            /* tp_hash  */
    gcCallbackWrapper,                            /* tp_call */
    0,                                            /* tp_str */
    0,                                            /* tp_getattro */
    0,                                            /* tp_setattro */
    0,                                            /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                           /* tp_flags */
    0,                                            /* tp_doc */
    0,                                            /* tp_traverse */
    0,                                            /* tp_clear */
    0,                                            /* tp_richcompare */
    0,                                            /* tp_weaklistoffset */
    0,                                            /* tp_iter */
    0,                                            /* tp_iternext */
    0,                                            /* tp_methods */
    0,                                            /* tp_members */
    0,                                            /* tp_getset */
    0,                                            /* tp_base */
    0,                                            /* tp_dict */
    0,                                            /* tp_descr_get */
    0,                                            /* tp_descr_set */
    0,                                            /* tp_dictoffset */
    0,                                            /* tp_init */
    0,                                            /* tp_alloc */
    0,                                            /* tp_new */
};

XpuTimerPyTracingDataArray* xpu_timer_get_partial_py_tracing_data_array(
    int name) {
  pthread_mutex_lock(&mutex);
  NamedTracingData* tracing_data = getTracingData(name);
  if ((!tracing_data || !tracing_data->curr_data) ||
      (tracing_data->curr_data->cur == 0)) {
    pthread_mutex_unlock(&mutex);
    return NULL;
  }
  XpuTimerPyTracingDataArray* result = tracing_data->curr_data;
  tracing_data->curr_data = xpu_timer_get_empty_py_tracing_data_array(name);
  pthread_mutex_unlock(&mutex);
  return result;
}

void xpu_timer_switch_py_tracing(int set_start_tracing) {
  start_tracing = set_start_tracing;
  printf("Set python tracing to %d\n", start_tracing);
}

int64_t xpu_timer_get_py_tracing_count(int name) {
  NamedTracingData* tracing_data = getTracingData(name);
  if (!tracing_data) return 0;
  return tracing_data->count;
}

void xpu_timer_register_gc(char** error_message) {
  addTracingData(PY_TRACING_GC, "GC");
  PyObject* gc_module = PyImport_ImportModule("gc");
  if (!gc_module) {
    return;
  }

  PyObject* callbacks_list = PyObject_GetAttrString(gc_module, "callbacks");
  if (!callbacks_list || !PyList_Check(callbacks_list)) {
    Py_XDECREF(callbacks_list);
    Py_DECREF(gc_module);
    return;
  }

  PyObject* py_callback = PyObject_New(PyObject, &GcCallbackType);

  if (!py_callback) {
    Py_DECREF(callbacks_list);
    Py_DECREF(gc_module);
    return;
  }

  if (PyList_Append(callbacks_list, py_callback) != 0) {
    Py_DECREF(py_callback);
    Py_DECREF(callbacks_list);
    Py_DECREF(gc_module);
    return;
  }

  Py_DECREF(callbacks_list);
  Py_DECREF(gc_module);
  *error_message = strdup("Import gc Ok");
}

void xpu_timer_register_tracing(const char** names, int count, char** errors) {
  if (!Py_IsInitialized()) {
    Py_Initialize();
  }
  PyGILState_STATE gstate = PyGILState_Ensure();
  // register syshook first
  setSysHook();
  // allocate PyCodeObject for each function
  // +1 is for gc, other python call from 1
  tracing_data_count = count;
  named_tracing_data_array =
      (NamedTracingData*)malloc(sizeof(NamedTracingData) * tracing_data_count);
  memset(named_tracing_data_array, 0,
         sizeof(NamedTracingData) * tracing_data_count);
  // register gc, code is 0
  xpu_timer_register_gc(errors);
  int64_t code_address;
  int is_native;

  for (int i = 1; i < count; i++) {
    int ret = runPythonCodeGetAddress(names[i], errors + i, &code_address,
                                      &is_native);
    if (ret) {
      printf("register function `%s` error\n", names[i]);
      continue;
    }
    printf("register function `%s` at address %ld\n", names[i], code_address);
    addTracingData(i, names[i]);

    NamedTracingFunction* traced_function =
        (NamedTracingFunction*)malloc(sizeof(NamedTracingFunction));
    traced_function->tag_name = i;
    traced_function->function_name = strdup(names[i]);
    traced_function->py_code_address = code_address;
    traced_function->is_native = is_native;

    HASH_ADD(hh, named_tracing_func_map, py_code_address, sizeof(int64_t),
             traced_function);
  }
  PyEval_SetProfile(profiler, NULL);
  PyThreadState* tstate = PyThreadState_Get();
  PyThreadState* thread_array[PY_TRACING_MAX_THREADS];
  memset(thread_array, 0, sizeof(thread_array));
  int thread_count = 0;
  while (tstate != NULL && thread_count < PY_TRACING_MAX_THREADS) {
    thread_array[thread_count++] = tstate;
    printf("Set profiler for thread %ld\n", tstate->thread_id);
    tstate = PyThreadState_Next(tstate);
  }
  for (int i = 0; i < thread_count; i++) {
    PyThreadState_Swap(thread_array[i]);
    PyEval_SetProfile(profiler, NULL);
  }
  PyThreadState_Swap(thread_array[0]);

  PyGILState_Release(gstate);
}

void xpu_timer_tracing_debug() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  for (int i = 0; i < tracing_data_count; i++) {
    NamedTracingData* tracing_data = getTracingData(i);
    if (!tracing_data) continue;
    XpuTimerPyTracingDataArray* curr_data = tracing_data->curr_data;
    for (int j = 0; j < curr_data->cur; j++)
      printf("%s start %ld, end %ld\n", tracing_data->function_name,
             curr_data->data[j].start, curr_data->data[j].end);
    printf("%s count %ld\n", tracing_data->function_name, tracing_data->count);
  }
  PyEval_SetProfile(NULL, NULL);
  PyGILState_Release(gstate);
}
