#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Function that adds 1 to an integer
static PyObject* add_one(PyObject* self, PyObject* args) {
    int input;

    if (!PyArg_ParseTuple(args, "i", &input)) {
        return NULL;
    }

    return PyLong_FromLong(input + 1);
}

// Method definitions
static PyMethodDef ExtraMethods[] = {
    {"add_one", add_one, METH_VARARGS, "Add 1 to an integer."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef extramodule = {
    PyModuleDef_HEAD_INIT,
    "extra",   // Name of the module
    NULL,      // Module documentation
    -1,        // Size of per-interpreter state of the module
    ExtraMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_extra(void) {
    return PyModule_Create(&extramodule);
}
