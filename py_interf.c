
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>

#include "rknng_lib.h"
#include <stdio.h>

/*
 * rpdivknngmodule.c
 * This is the C code for a non-numpy Python extension to
 * define the logit function, where logit(p) = log(p/(1-p)).
 * This function will not work on numpy arrays automatically.
 * numpy.vectorize must be called in python to generate
 * a numpy-friendly function.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

/* This declares the logit function */
// static PyObject *rpdiv_knng(PyObject *self, PyObject *args);
static PyObject *rpdiv_knng(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *rpdiv_knng_o(PyObject *self, PyObject *args, PyObject *kwargs);

/*
 * This tells Python what methods this module has.
 * See the Python-C API for more information.
 */
static PyMethodDef SpamMethods[] = {
    // {"logit", rpdivknng_logit, METH_VARARGS, "compute logit"},
    {"rpdiv_knng", rpdiv_knng, METH_VARARGS | METH_KEYWORDS, "Create kNN graph"},
    {"rpdiv_knng_o", rpdiv_knng_o, METH_VARARGS | METH_KEYWORDS, "Create kNN graph"},
    {NULL, NULL, 0, NULL}};

/*
 * This actually defines the logit function for
 * input args from Python.
 */

#define v(x0, x1)                                                                                  \
  (*(npy_float64 *)((PyArray_DATA(py_v) + (x0)*PyArray_STRIDES(py_v)[0] +                          \
                     (x1)*PyArray_STRIDES(py_v)[1])))
#define v_shape(i) (py_v->dimensions[(i)])

// static PyObject *rpdiv_knng(PyArrayObject *py_v, PyObject *py_k, PyObject *py_w, PyObject
// *py_nndes, PyObject *py_delta, PyObject *py_maxiter) {

static PyObject *rpdiv_knng_o(PyObject *self, PyObject *args, PyObject *kwargs) {
  import_array();
  printf("TEST33\n");

  PyObject *py_v;
  int k, w = 0, maxiter = 100;
  float nndes = 0.2, delta = 0.01;
  char *type = NULL;
  char *distance = NULL;

  PyObject *ret;
  static char *kwlist[] = {"v",     "k",     "window",   "nndes", "maxiter",
                           "delta", "dtype", "distance", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ififss", kwlist, &py_v, &k, &w, &nndes,
                                   &maxiter, &delta, &type, &distance)) {

    return NULL;
  }
  if (w <= 0) {
    w = 2 * k;
  }

  delta = 0.01;

  // double dval = 0.0;
  // PyObject *dargs = Py_BuildValue("(ii)", 20, 30);

  // PyObject *myobject_method = PyObject_GetAttrString(py_v, "distance");
  // // PyObject_CallMethod(py_v, "distance", "i", 20);
  // // PyObject *result = PyObject_Call(myobject_method, dargs, NULL);
  // PyObject *p1;
  // PyObject *p2;

  // PyObject *result;
  // for (int i = 0; i < 300; i++) {
  // p1 = PyLong_FromLong(22 + i);
  // p2 = PyLong_FromLong(23);

  // result = PyObject_CallFunctionObjArgs(myobject_method, p1, p2, NULL);
  // dval = PyFloat_AS_DOUBLE(result);
  // // printf("dval=%f\n", dval);
  // Py_DECREF(p1);
  // Py_DECREF(p2);
  // Py_DECREF(result);
  // // Py_DECREF(dval);
  // }
  // https://stackoverflow.com/questions/16777126/pyobject-callmethod-with-keyword-arguments
  // Py_DECREF(args);
  // Py_DECREF(keywords);
  // Py_DECREF(myobject_method);

  // // Do something with the result
  // Py_DECREF(result);

  // PyLong_AsLong(py_w);

  if (distance != NULL) {
    // if (strcmp("l2", distance) == 0) {
    // } else if (strcmp("mnkw", distance) == 0) {
    // } else if (strcmp("cos", distance) == 0) {
    // } else {
    // PyErr_SetString(PyExc_ValueError, "Distance must be one for {l2(default),l1,cos}");
    // return NULL;
    // }
  }

  printf("Delta=%f\n", delta);
  ret = __rpdiv_knng_o(py_v, k, w, nndes, delta, maxiter);

  // else if(strcmp("lev", distance) == 0) {}
  // else if(strcmp("dice", distance) == 0) {}

  // if (type != NULL && distance != NULL) {
  // printf(":%s %s\n", type, distance);
  // }

  // printf("FOOOO:\n");

  // // ret = __rpdiv_knng(py_v, py_k, py_w, py_nndes, py_delta, py_maxiter);
  // ret = __rpdiv_knng(py_v, k, w, nndes, delta, maxiter);
  // ret = __rpdiv_knng(py_v);
  return ret;
  // return result;
}

static PyObject *rpdiv_knng(PyObject *self, PyObject *args, PyObject *kwargs) {
  import_array();

  PyArrayObject *py_v;
  int k, w = 0, maxiter = 100;
  float nndes = 0.2, delta = 0.01;
  char *type = NULL;
  char *distance = NULL;
  // PyObject *py_k;
  // PyObject *py_w;
  // PyObject *py_nndes;
  // PyObject *py_delta;
  // PyObject *py_maxiter;

  PyObject *ret;
  static char *kwlist[] = {"v",     "k",     "window",   "nndes", "maxiter",
                           "delta", "dtype", "distance", NULL};

  // if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &py_v)) {
  // return NULL;
  // }

  // if (!PyArg_ParseTuple(args, "O!ll", &PyArray_Type, &py_v, &py_k, &py_w)) {
  // if (!PyArg_ParseTuple(args, "O!ll", &PyArray_Type, &py_v, &py_k, &py_w)) {
  // if (!PyArg_ParseTuple(args, "O!iiffi", &PyArray_Type, &py_v, &k, &w, &nndes, &delta, &maxiter))
  // {
  // delta=0.05;
  // if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!iifi|f", kwlist, &PyArray_Type, &py_v, &k,
  // &w, &nndes, &maxiter, &delta)) {
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!i|ififss", kwlist, &PyArray_Type, &py_v, &k, &w,
                                   &nndes, &maxiter, &delta, &type, &distance)) {
    return NULL;
  }
  if (w <= 0) {
    w = 2 * k;
  }

  // PyLong_AsLong(py_w);

  if (distance != NULL) {
    if (strcmp("l2", distance) == 0) {
    } else if (strcmp("mnkw", distance) == 0) {
    } else if (strcmp("cos", distance) == 0) {
    } else {
      PyErr_SetString(PyExc_ValueError, "Distance must be one for {l2(default),l1,cos}");
      return NULL;
    }
  }

  // else if(strcmp("lev", distance) == 0) {}
  // else if(strcmp("dice", distance) == 0) {}

  if (type != NULL && distance != NULL) {
    printf(":%s %s\n", type, distance);
  }

  printf("FOOOO:\n");

  // ret = __rpdiv_knng(py_v, py_k, py_w, py_nndes, py_delta, py_maxiter);
  ret = __rpdiv_knng(py_v, k, w, nndes, delta, maxiter);
  // ret = __rpdiv_knng(py_v);
  return ret;
}

/* This initiates the module using the above definitions. */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "rpdivknng", NULL, -1, SpamMethods, NULL, NULL, NULL, NULL};

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

  m = Py_InitModule("rpdivknng", SpamMethods);
  if (m == NULL) {
    return;
  }
}
#endif
