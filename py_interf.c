
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>

#include "rknng_lib.h"
#include <stdio.h>

#include "constants.h"

static PyObject *rpdiv_knng(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject *rpdiv_knng_generic(PyObject *self, PyObject *args, PyObject *kwargs);

// Define python accessible methods
static PyMethodDef RpdivknngMethods[] = {
    // {"logit", rpdivknng_logit, METH_VARARGS, "compute logit"},
    {"get_knng", rpdiv_knng, METH_VARARGS | METH_KEYWORDS, "Create kNN graph"},
    {"get_knng_generic", rpdiv_knng_generic, METH_VARARGS | METH_KEYWORDS,
     "Create kNN graph using python provided distance function"},
    {NULL, NULL, 0, NULL}};

#define v(x0, x1)                                                                                  \
  (*(npy_float64 *)((PyArray_DATA(py_v) + (x0)*PyArray_STRIDES(py_v)[0] +                          \
                     (x1)*PyArray_STRIDES(py_v)[1])))
#define v_shape(i) (py_v->dimensions[(i)])

// For generic distance functions implemented in python
static PyObject *rpdiv_knng_generic(PyObject *self, PyObject *args, PyObject *kwargs) {
  import_array();

  PyObject *py_v;
  int k, w = 0, maxiter = 100;
  float nndes = 0.0, delta = 0.05;
  char *type = NULL;
  // char *distance = NULL;

  PyObject *ret;
  static char *kwlist[] = {"v", "k", "window", "nndes", "maxiter", "delta", "dtype", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ififss", kwlist, &py_v, &k, &w, &nndes,
                                   &maxiter, &delta, &type)) {

    return NULL;
  }
  if (w <= 0) {
    w = 2 * k;
  }
  if (w <= 20) {
    w = 20;
  }

  printf("DELTA=%f\n", delta);

  ret = __rpdiv_knng_generic(py_v, k, w, nndes, delta, maxiter);
  return ret;
}

// For distance function implemented in C code in dataset.h
static PyObject *rpdiv_knng(PyObject *self, PyObject *args, PyObject *kwargs) {
  import_array();

  PyArrayObject *py_v;
  int k, w = 0, maxiter = 100;
  float nndes = 0.0, delta = 0.05;
  char *type = NULL;
  char *distance = NULL;

  PyObject *ret;
  static char *kwlist[] = {"v",     "k",     "window",   "nndes", "maxiter",
                           "delta", "dtype", "distance", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!i|ififss", kwlist, &PyArray_Type, &py_v, &k, &w,
                                   &nndes, &maxiter, &delta, &type, &distance)) {
    return NULL;
  }
  if (w <= 0) {
    w = 2 * k;
  }
  if (w <= 20) {
    w = 20;
  }

  int dtype = D_L2;
  if (distance != NULL) {
    if (strcmp("l2", distance) == 0) {
      dtype = D_L2;
    } else if (strcmp("l1", distance) == 0) {
      dtype = D_L1;
    } else if (strcmp("cos", distance) == 0) {
      dtype = D_COS;
    } else {
      PyErr_SetString(PyExc_ValueError, "Distance must be one for {l2(default),l1,cos}");
      return NULL;
    }
  }

  // if (type != NULL && distance != NULL) {
  // printf(":%s %s\n", type, distance);
  // }

  ret = __rpdiv_knng(py_v, k, w, nndes, delta, maxiter, dtype);
  return ret;
}

/* This initiates the module using the above definitions. */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "rpdivknng", NULL, -1, RpdivknngMethods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_rpdivknng(void) {
  PyObject *m;
  m = PyModule_Create(&moduledef);
  if (!m) {
    return NULL;
  }
  return m;
}
#else
PyMODINIT_FUNC initrpdivknng(void) {
  PyObject *m;

  m = Py_InitModule("rpdivknng", RpdivknngMethods);
  if (m == NULL) {
    return;
  }
}
#endif
