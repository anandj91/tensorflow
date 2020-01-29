/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <array>

#include "tensorflow/python/lib/core/custom.h"

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/python/lib/core/numpy.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace tensorflow {
namespace {

// Workarounds for Python 2 vs 3 API differences.
#if PY_MAJOR_VERSION < 3

PyObject* MakePyString(const string& s) {
  return PyString_FromString(s.c_str());
}

typedef long HashType;  // NOLINT

bool TfPyInt_Check(PyObject* object) { return PyInt_Check(object); }

PyObject* TfPyInt_FromLong(long x) {  // NOLINT
  return PyInt_FromLong(x);
}

long TfPyInt_AsLong(PyObject* x) {  // NOLINT
  return PyInt_AsLong(x);
}

#else  // PY_MAJOR_VERSION < 3

PyObject* MakePyString(const string& s) {
  return PyUnicode_FromString(s.c_str());
}

bool TfPyInt_Check(PyObject* object) {
  if (!PyLong_Check(object)) {
    return 0;
  }
  int overflow = 0;
  PyLong_AsLongAndOverflow(object, &overflow);
  return (overflow == 0);
}

PyObject* TfPyInt_FromLong(long x) {  // NOLINT
  return PyLong_FromLong(x);
}

long TfPyInt_AsLong(PyObject* x) {  // NOLINT
  return PyLong_AsLong(x);
}

typedef Py_hash_t HashType;

#endif  // PY_MAJOR_VERSION < 3

// Forward declaration.
extern PyTypeObject PyCustom_Type;

// Representation of a Python custom object.
struct PyCustom {
  PyObject_HEAD;  // Python object header
  custom value;
};

// Returns true if 'object' is a PyCustom.
bool PyCustom_Check(PyObject* object) {
  return PyObject_IsInstance(object,
                             reinterpret_cast<PyObject*>(&PyCustom_Type));
}

// Extracts the value of a PyCustom object.
custom PyCustom_Custom(PyObject* object) {
  return reinterpret_cast<PyCustom*>(object)->value;
}

// Constructs a PyCustom object from a custom.
Safe_PyObjectPtr PyCustom_FromCustom(custom x) {
  Safe_PyObjectPtr ref =
      make_safe(PyCustom_Type.tp_alloc(&PyCustom_Type, 0));
  PyCustom* p = reinterpret_cast<PyCustom*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a custom value. Returns true on success,
// returns false and reports a Python error on failure.
bool AsCustom(PyObject* arg, custom* output) {
  if (PyCustom_Check(arg)) {
    *output = PyCustom_Custom(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = custom(d);
    return true;
  }
  if (TfPyInt_Check(arg)) {
    long l = TfPyInt_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = custom(static_cast<float>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = custom(f);
    return true;
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               arg->ob_type->tp_name);
  return false;
}

// Converts a PyCustom into a PyFloat.
PyObject* PyCustom_Float(PyObject* self) {
  custom x = PyCustom_Custom(self);
  return PyFloat_FromDouble(static_cast<double>(x));
}

// Converts a PyCustom into a PyInt.
PyObject* PyCustom_Int(PyObject* self) {
  custom x = PyCustom_Custom(self);
  long y = static_cast<long>(x);  // NOLINT
  return TfPyInt_FromLong(y);
}

// Negates a PyCustom.
PyObject* PyCustom_Negative(PyObject* self) {
  custom x = PyCustom_Custom(self);
  return PyCustom_FromCustom(-x).release();
}

// Binary arithmetic operators on PyCustom values.
#define CUSTOM_BINOP(name, op)                                  \
  PyObject* PyCustom_##name(PyObject* a, PyObject* b) {         \
    custom x, y;                                                \
    if (!AsCustom(a, &x) || !AsCustom(b, &y)) return nullptr; \
    custom z = x op y;                                          \
    return PyCustom_FromCustom(z).release();                  \
  }
CUSTOM_BINOP(Add, +)
CUSTOM_BINOP(Subtract, -)
CUSTOM_BINOP(Multiply, *)
CUSTOM_BINOP(Divide, /)
#undef CUSTOM_BINOP

// Python number methods for PyCustom objects.
PyNumberMethods PyCustom_AsNumber = {
    PyCustom_Add,       // nb_add
    PyCustom_Subtract,  // nb_subtract
    PyCustom_Multiply,  // nb_multiply
#if PY_MAJOR_VERSION < 3
    PyCustom_Divide,  // nb_divide
#endif
    nullptr,              // nb_remainder
    nullptr,              // nb_divmod
    nullptr,              // nb_power
    PyCustom_Negative,  // nb_negative
    nullptr,              // nb_positive
    nullptr,              // nb_absolute
    nullptr,              // nb_nonzero
    nullptr,              // nb_invert
    nullptr,              // nb_lshift
    nullptr,              // nb_rshift
    nullptr,              // nb_and
    nullptr,              // nb_xor
    nullptr,              // nb_or
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_coerce
#endif
    PyCustom_Int,  // nb_int
#if PY_MAJOR_VERSION < 3
    PyCustom_Int,  // nb_long
#else
    nullptr,  // reserved
#endif
    PyCustom_Float,  // nb_float
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_oct
    nullptr,  // nb_hex
#endif

    nullptr,  // nb_inplace_add
    nullptr,  // nb_inplace_subtract
    nullptr,  // nb_inplace_multiply
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_inplace_divide
#endif
    nullptr,  // nb_inplace_remainder
    nullptr,  // nb_inplace_power
    nullptr,  // nb_inplace_lshift
    nullptr,  // nb_inplace_rshift
    nullptr,  // nb_inplace_and
    nullptr,  // nb_inplace_xor
    nullptr,  // nb_inplace_or

    nullptr,            // nb_floor_divide
    PyCustom_Divide,  // nb_true_divide
    nullptr,            // nb_inplace_floor_divide
    nullptr,            // nb_inplace_true_divide
    nullptr,            // nb_index
};

// Constructs a new PyCustom.
PyObject* PyCustom_New(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "expected number as argument to custom constructor");
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  if (PyCustom_Check(arg)) {
    Py_INCREF(arg);
    return arg;
  } else {
    custom value;
    if (!AsCustom(arg, &value)) {
      return nullptr;
    }
    return PyCustom_FromCustom(value).release();
  }
}

// Comparisons on PyCustoms.
PyObject* PyCustom_RichCompare(PyObject* a, PyObject* b, int op) {
  custom x, y;
  if (!AsCustom(a, &x) || !AsCustom(b, &y)) return nullptr;
  bool result;
  switch (op) {
    case Py_LT:
      result = x < y;
      break;
    case Py_LE:
      result = x <= y;
      break;
    case Py_EQ:
      result = x == y;
      break;
    case Py_NE:
      result = x != y;
      break;
    case Py_GT:
      result = x > y;
      break;
    case Py_GE:
      result = x >= y;
      break;
    default:
      LOG(FATAL) << "Invalid op type " << op;
  }
  return PyBool_FromLong(result);
}

// Implementation of repr() for PyCustom.
PyObject* PyCustom_Repr(PyObject* self) {
  custom x = reinterpret_cast<PyCustom*>(self)->value;
  string v = strings::StrCat("custom(", static_cast<float>(x), ")");
  return MakePyString(v);
}

// Implementation of str() for PyCustom.
PyObject* PyCustom_Str(PyObject* self) {
  custom x = reinterpret_cast<PyCustom*>(self)->value;
  string v = strings::StrCat(static_cast<float>(x));
  return MakePyString(v);
}

// Hash function for PyCustom. We use the identity function, which is a weak
// hash function.
HashType PyCustom_Hash(PyObject* self) {
  custom x = reinterpret_cast<PyCustom*>(self)->value;
  return x.value;
}

// Python type for PyCustom objects.
PyTypeObject PyCustom_Type = {
#if PY_MAJOR_VERSION < 3
    PyObject_HEAD_INIT(nullptr) 0,  // ob_size
#else
    PyVarObject_HEAD_INIT(nullptr, 0)
#endif
    "custom",                                // tp_name
    sizeof(PyCustom),                        // tp_basicsize
    0,                                         // tp_itemsize
    nullptr,                                   // tp_dealloc
    nullptr,                                   // tp_print
    nullptr,                                   // tp_getattr
    nullptr,                                   // tp_setattr
    nullptr,                                   // tp_compare / tp_reserved
    PyCustom_Repr,                           // tp_repr
    &PyCustom_AsNumber,                      // tp_as_number
    nullptr,                                   // tp_as_sequence
    nullptr,                                   // tp_as_mapping
    PyCustom_Hash,                           // tp_hash
    nullptr,                                   // tp_call
    PyCustom_Str,                            // tp_str
    nullptr,                                   // tp_getattro
    nullptr,                                   // tp_setattro
    nullptr,                                   // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  // tp_flags
    "custom floating-point values",          // tp_doc
    nullptr,                                   // tp_traverse
    nullptr,                                   // tp_clear
    PyCustom_RichCompare,                    // tp_richcompare
    0,                                         // tp_weaklistoffset
    nullptr,                                   // tp_iter
    nullptr,                                   // tp_iternext
    nullptr,                                   // tp_methods
    nullptr,                                   // tp_members
    nullptr,                                   // tp_getset
    nullptr,                                   // tp_base
    nullptr,                                   // tp_dict
    nullptr,                                   // tp_descr_get
    nullptr,                                   // tp_descr_set
    0,                                         // tp_dictoffset
    nullptr,                                   // tp_init
    nullptr,                                   // tp_alloc
    PyCustom_New,                            // tp_new
    nullptr,                                   // tp_free
    nullptr,                                   // tp_is_gc
    nullptr,                                   // tp_bases
    nullptr,                                   // tp_mro
    nullptr,                                   // tp_cache
    nullptr,                                   // tp_subclasses
    nullptr,                                   // tp_weaklist
    nullptr,                                   // tp_del
    0,                                         // tp_version_tag
};

// Numpy support

PyArray_ArrFuncs NPyCustom_ArrFuncs;

PyArray_Descr NPyCustom_Descr = {
    PyObject_HEAD_INIT(nullptr) & PyCustom_Type,  // typeobj
    // We must register custom with a kind other than "f", because numpy
    // considers two types with the same kind and size to be equal, but
    // float16 != custom.
    'V',  // kind
    // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    'E',                                                  // type
    '=',                                                  // byteorder
    NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,  // hasobject
    0,                                                    // type_num
    sizeof(custom),                                     // elsize
    alignof(custom),                                    // alignment
    nullptr,                                              // subarray
    nullptr,                                              // fields
    nullptr,                                              // names
    &NPyCustom_ArrFuncs,                                // f
};

// Registered numpy type ID. Global variable populated by the registration code.
int npy_custom_ = -1;

// Implementations of NumPy array methods.

PyObject* NPyCustom_GetItem(void* data, void* arr) {
  custom x;
  memcpy(&x, data, sizeof(custom));
  return PyCustom_FromCustom(x).release();
}

int NPyCustom_SetItem(PyObject* item, void* data, void* arr) {
  custom x;
  if (!AsCustom(item, &x)) return -1;
  memcpy(data, &x, sizeof(custom));
  return 0;
}

void ByteSwap16(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[1]);
}

void NPyCustom_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                           npy_intp sstride, npy_intp n, int swap, void* arr) {
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);
  if (!src) {
    return;
  }
  if (swap) {
    for (npy_intp i = 0; i < n; i++) {
      char* r = dst + dstride * i;
      memcpy(r, src + sstride * i, sizeof(uint16_t));
      ByteSwap16(r);
    }
  } else if (dstride == sizeof(uint16_t) && sstride == sizeof(uint16_t)) {
    memcpy(dst, src, n * sizeof(uint16_t));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(uint16_t));
    }
  }
}

void NPyCustom_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(uint16_t));
  if (swap) {
    ByteSwap16(dst);
  }
}

npy_bool NPyCustom_NonZero(void* data, void* arr) {
  custom x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<custom>(0);
}

int NPyCustom_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  custom* const buffer = reinterpret_cast<custom*>(buffer_raw);
  const float start(buffer[0]);
  const float delta = static_cast<float>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<custom>(start + i * delta);
  }
  return 0;
}

// NumPy casts

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
             void* toarr) {
  const From* from = reinterpret_cast<From*>(from_void);
  To* to = reinterpret_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<To>(from[i]);
  }
}

// Registers a cast between custom and type 'T'. 'numpy_type' is the NumPy
// type corresponding to 'T'. If 'cast_is_safe', registers that custom can be
// safely coerced to T.
template <typename T>
bool RegisterCustomCast(int numpy_type, bool cast_is_safe) {
  if (PyArray_RegisterCastFunc(PyArray_DescrFromType(numpy_type), npy_custom_,
                               NPyCast<T, custom>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(&NPyCustom_Descr, numpy_type,
                               NPyCast<custom, T>) < 0) {
    return false;
  }
  if (cast_is_safe && PyArray_RegisterCanCast(&NPyCustom_Descr, numpy_type,
                                              NPY_NOSCALAR) < 0) {
    return false;
  }
  return true;
}

template <typename InType, typename OutType, typename Functor>
void BinaryUFunc(char** args, npy_intp* dimensions, npy_intp* steps,
                 void* data) {
  const char* i0 = args[0];
  const char* i1 = args[1];
  char* o = args[2];
  for (npy_intp k = 0; k < *dimensions; k++) {
    InType x = *reinterpret_cast<const InType*>(i0);
    InType y = *reinterpret_cast<const InType*>(i1);
    *reinterpret_cast<OutType*>(o) = Functor()(x, y);
    i0 += steps[0];
    i1 += steps[1];
    o += steps[2];
  }
}

template <typename Functor>
void CompareUFunc(char** args, npy_intp* dimensions, npy_intp* steps,
                  void* data) {
  BinaryUFunc<custom, npy_bool, Functor>(args, dimensions, steps, data);
}

struct CustomEqFunctor {
  npy_bool operator()(custom a, custom b) { return a == b; }
};
struct CustomNeFunctor {
  npy_bool operator()(custom a, custom b) { return a != b; }
};
struct CustomLtFunctor {
  npy_bool operator()(custom a, custom b) { return a < b; }
};
struct CustomGtFunctor {
  npy_bool operator()(custom a, custom b) { return a > b; }
};
struct CustomLeFunctor {
  npy_bool operator()(custom a, custom b) { return a <= b; }
};
struct CustomGeFunctor {
  npy_bool operator()(custom a, custom b) { return a >= b; }
};

// Initializes the module.
bool Initialize() {
  // It's critical to import umath to avoid crash in open source build.
  import_umath1(false);

  Safe_PyObjectPtr numpy_str = make_safe(MakePyString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  // We hit a mysterious crash if we haven't initialized numpy before this:
  PyCustom_Type.tp_base = &PyGenericArrType_Type;

  if (PyType_Ready(&PyCustom_Type) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_InitArrFuncs(&NPyCustom_ArrFuncs);
  NPyCustom_ArrFuncs.getitem = NPyCustom_GetItem;
  NPyCustom_ArrFuncs.setitem = NPyCustom_SetItem;
  NPyCustom_ArrFuncs.copyswapn = NPyCustom_CopySwapN;
  NPyCustom_ArrFuncs.copyswap = NPyCustom_CopySwap;
  NPyCustom_ArrFuncs.nonzero = NPyCustom_NonZero;
  NPyCustom_ArrFuncs.fill = NPyCustom_Fill;

  Py_TYPE(&NPyCustom_Descr) = &PyArrayDescr_Type;
  npy_custom_ = PyArray_RegisterDataType(&NPyCustom_Descr);
  if (npy_custom_ < 0) return false;

  // Support dtype(custom)
  if (PyDict_SetItemString(PyCustom_Type.tp_dict, "dtype",
                           reinterpret_cast<PyObject*>(&NPyCustom_Descr)) <
      0) {
    return false;
  }

  // Register casts

  // We lie shamelessly and say that a cast from half to custom is safe.
  // Numpy frequently uses the smallest legal representation type for small
  // float constants (e.g., 1.0), which is often float16. Things break if these
  // cannot be converted transparently to custom.
  if (!RegisterCustomCast<Eigen::half>(NPY_HALF, /*cast_is_safe=*/true)) {
    return false;
  }

  if (!RegisterCustomCast<float>(NPY_FLOAT, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterCustomCast<double>(NPY_DOUBLE, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterCustomCast<int32>(NPY_INT32, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCustomCast<int64>(NPY_INT64, /*cast_is_safe=*/false)) {
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterCustomCast<complex64>(NPY_COMPLEX64, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterCustomCast<complex128>(NPY_COMPLEX128,
                                        /*cast_is_safe=*/true)) {
    return false;
  }

  // Register ufuncs
  auto register_ufunc = [&](const char* name, PyUFuncGenericFunction fn,
                            const std::array<int, 3>& types) {
    Safe_PyObjectPtr ufunc_obj =
        make_safe(PyObject_GetAttrString(numpy.get(), name));
    if (!ufunc_obj) {
      return false;
    }
    PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());
    if (types.size() != ufunc->nargs) {
      PyErr_Format(PyExc_AssertionError,
                   "ufunc %s takes %d arguments, loop takes %lu", name,
                   ufunc->nargs, types.size());
      return false;
    }
    if (PyUFunc_RegisterLoopForType(ufunc, npy_custom_, fn,
                                    const_cast<int*>(types.data()),
                                    nullptr) < 0) {
      return false;
    }
    return true;
  };

  // Comparisons
  const std::array<int, 3> compare_types = {
      {npy_custom_, npy_custom_, NPY_BOOL}};

  if (!register_ufunc("equal", CompareUFunc<CustomEqFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("not_equal", CompareUFunc<CustomNeFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("less", CompareUFunc<CustomLtFunctor>, compare_types)) {
    return false;
  }
  if (!register_ufunc("greater", CompareUFunc<CustomGtFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("less_equal", CompareUFunc<CustomLeFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("greater_equal", CompareUFunc<CustomGeFunctor>,
                      compare_types)) {
    return false;
  }
  return true;
}

}  // namespace

void RegisterNumpyCustom() {
  if (npy_custom_ >= 0) {
    // Already initialized.
    return;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load custom module.");
    }
    PyErr_Print();
  }
}

PyObject* CustomPyType() {
  CHECK(PyCustom_Type.tp_base != nullptr);
  Py_INCREF(&PyCustom_Type);
  return reinterpret_cast<PyObject*>(&PyCustom_Type);
}

int CustomNumpyType() {
  CHECK_GE(npy_custom_, 0);
  return npy_custom_;
}

}  // namespace tensorflow
