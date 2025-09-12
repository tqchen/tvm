import torch
from torch.utils import cpp_extension
import time

import tvm_ffi
import tvm_ffi.cpp


def load_to_dlpack():
    cpp_source = """
#include <dlpack/dlpack.h>
#include <ATen/DLConvertor.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
#include "tvm_ffi_python_helpers.h"

using namespace std;
namespace at {
namespace {
DLDataType getDLDataTypeX(const Tensor& t) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = t.element_size() * 8;
  switch (t.scalar_type()) {
    case ScalarType::UInt1:
    case ScalarType::UInt2:
    case ScalarType::UInt3:
    case ScalarType::UInt4:
    case ScalarType::UInt5:
    case ScalarType::UInt6:
    case ScalarType::UInt7:
    case ScalarType::Byte:
    case ScalarType::UInt16:
    case ScalarType::UInt32:
    case ScalarType::UInt64:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    case ScalarType::Int1:
    case ScalarType::Int2:
    case ScalarType::Int3:
    case ScalarType::Int4:
    case ScalarType::Int5:
    case ScalarType::Int6:
    case ScalarType::Int7:
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ScalarType::Int:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Long:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Bool:
      dtype.code = DLDataTypeCode::kDLBool;
      break;
    case ScalarType::ComplexHalf:
    case ScalarType::ComplexFloat:
    case ScalarType::ComplexDouble:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case ScalarType::BFloat16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      break;
    // TODO(#146647): use macro here instead of spelling out each shell dtype
    case ScalarType::Float8_e5m2:
    case ScalarType::Float8_e5m2fnuz:
    case ScalarType::Float8_e4m3fn:
    case ScalarType::Float8_e4m3fnuz:
    case ScalarType::Float8_e8m0fnu:
      TORCH_CHECK_INDEX(false, "float8 types are not supported by dlpack");
      break;
    case ScalarType::Float4_e2m1fn_x2:
      TORCH_CHECK_INDEX(false, "float4 types are not supported by dlpack");
      break;
    case ScalarType::QInt8:
    case ScalarType::QUInt8:
    case ScalarType::QInt32:
    case ScalarType::QUInt4x2:
    case ScalarType::QUInt2x4:
      TORCH_CHECK_INDEX(false, "QUInt/QInt types are not supported by dlpack");
      break;
    case ScalarType::Bits1x8:
    case ScalarType::Bits2x4:
    case ScalarType::Bits4x2:
    case ScalarType::Bits8:
    case ScalarType::Bits16:
      TORCH_CHECK_INDEX(false, "Bit types are not supported by dlpack");
      break;
    case ScalarType::Undefined:
      TORCH_CHECK_INDEX(false, "Undefined is not a valid ScalarType");
    case ScalarType::NumOptions:
      TORCH_CHECK_INDEX(false, "NumOptions is not a valid ScalarType");
  }
  return dtype;
}

DLDevice torchDeviceToDLDevice(at::Device device) {
  DLDevice ctx;

  ctx.device_id = (device.is_cuda() || device.is_privateuseone())
      ? static_cast<int32_t>(static_cast<unsigned char>(device.index()))
      : 0;

  switch (device.type()) {
    case DeviceType::CPU:
      ctx.device_type = DLDeviceType::kDLCPU;
      break;
    case DeviceType::CUDA:
#ifdef USE_ROCM
      // ROCM, if enabled will look like cuda to PyTorch
      // while everyone else should see HIP
      ctx.device_type = DLDeviceType::kDLROCM;
#else
      ctx.device_type = DLDeviceType::kDLCUDA;
#endif
      break;
    case DeviceType::OPENCL:
      ctx.device_type = DLDeviceType::kDLOpenCL;
      break;
    case DeviceType::HIP:
      ctx.device_type = DLDeviceType::kDLROCM;
      break;
    case DeviceType::XPU:
      ctx.device_type = DLDeviceType::kDLOneAPI;
      ctx.device_id = at::detail::getXPUHooks().getGlobalIdxFromDevice(device);
      break;
    case DeviceType::MAIA:
      ctx.device_type = DLDeviceType::kDLMAIA;
      break;
    case DeviceType::PrivateUse1:
      ctx.device_type = DLDeviceType::kDLExtDev;
      break;
    case DeviceType::MPS:
      ctx.device_type = DLDeviceType::kDLMetal;
      break;
    default:
      TORCH_CHECK_INDEX(false, "Cannot pack tensors on " + device.str());
  }

  return ctx;
}

// The templated classes below are needed for supporting both:
//   - DLManagedTensor
//   - DLManagedTensorVersioned
template <class T>
struct ATenDLMTensor {
  Tensor handle;
  T tensor{};
};

template <class T>
void deleter(T* arg) {
  delete static_cast<ATenDLMTensor<T>*>(arg->manager_ctx);
}

// Adds version information for DLManagedTensorVersioned.
// This is a no-op for the other types.
template <class T>
void fillVersion(T* tensor) {}

template <>
void fillVersion<DLManagedTensorVersioned>(
    DLManagedTensorVersioned* tensor) {
  tensor->flags = 0;
  tensor->version.major = DLPACK_MAJOR_VERSION;
  tensor->version.minor = DLPACK_MINOR_VERSION;
}

// This function returns a shared_ptr to memory managed DLpack tensor
// constructed out of ATen tensor
template <class T>
T* toDLPackImpl(const Tensor& src) {
  auto view = src;

  // Detect whether there is need to normalize the strides
  // Background: gh-83069
  //
  // However, normalizing strides can come at a high-cost
  // to slow down toDLPack conversion 3x, so we
  // only normalize if needed.
  //
  // The following code detects whether the src follows
  // a continuous pattern. If the src follows such pattern (common-case)
  // then we do not need to normalize the strides.
  bool need_normalize_strides = false;
  int64_t expected_stride = 1;
  for (int i = src.dim() - 1; i >= 0; i--) {
    // detect if we do not meet continuous pattern
    // and the size is 1, so there is opportunity to normalize
    if (src.stride(i) != expected_stride && src.size(i) == 1) {
      need_normalize_strides = true;
      break;
    }
    expected_stride *= src.size(i);
  }

  // less common case, try normalizing the strides
  if (need_normalize_strides) {
    // create a new tensor with possibly normalized strides
    // gh-83069
    auto shape = src.sizes();
    auto strides = src.strides().vec();
    for (int i = 0; i < src.dim(); i++) {
      if (shape[i] < 2) {
        strides[i] = 1;
      }
    }
    view = src.as_strided(shape, strides, src.storage_offset());
  }

  ATenDLMTensor<T>* atDLMTensor(new ATenDLMTensor<T>);
  atDLMTensor->handle = view;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.deleter = &deleter<T>;
  atDLMTensor->tensor.dl_tensor.data = view.data_ptr();
  atDLMTensor->tensor.dl_tensor.device = torchDeviceToDLDevice(src.device());
  atDLMTensor->tensor.dl_tensor.ndim = static_cast<int32_t>(src.dim());
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataTypeX(src);
  atDLMTensor->tensor.dl_tensor.shape = const_cast<int64_t*>(view.sizes().data());
  atDLMTensor->tensor.dl_tensor.strides = const_cast<int64_t*>(view.strides().data());
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  fillVersion(&atDLMTensor->tensor);

  return &(atDLMTensor->tensor);
}

// Explicitly instantiate the template above for both classes.
template DLManagedTensor* toDLPackImpl<DLManagedTensor>(const Tensor&);
template DLManagedTensorVersioned* toDLPackImpl<DLManagedTensorVersioned>(const Tensor&);

} // namespace
} // namespace at


void dlpack_cpp_exporter_bench(const at::Tensor& src, int repeat) {
  for (int i = 0; i < repeat; i++) {
    DLManagedTensorVersioned* dlpack = at::toDLPackImpl<DLManagedTensorVersioned>(src);
    dlpack->deleter(dlpack);
  }
}

int TorchDLPackPyCExporter(void* py_obj, DLManagedTensorVersioned** out, void** env_stream) {
  try {
    py::handle handle(static_cast<PyObject*>(py_obj));
    at::Tensor tensor = handle.cast<at::Tensor>();
    if (env_stream != nullptr && tensor.is_cuda()) {
      *env_stream = at::cuda::getCurrentCUDAStream(tensor.device().index()).stream();
    }
    *out = at::toDLPackImpl<DLManagedTensorVersioned>(tensor);
    return 0;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
}

int TorchDLPackPyCExporterCached(void* obj, DLManagedTensorVersioned** out, void** env_stream) {
  PyObject* py_obj = static_cast<PyObject*>(obj);
  DLManagedTensorVersioned* cache = nullptr;
  if (DLPackPyIntrusiveCacheFetch(py_obj, &cache) != 0) {
    return -1;
  }
  if (cache != nullptr) {
    if (env_stream != nullptr && cache->dl_tensor.device.device_type != DLDeviceType::kDLCPU) {
      *env_stream = at::cuda::getCurrentCUDAStream(cache->dl_tensor.device.device_id).stream();
    }
    *out = cache;
    return 0;
  }
  if (TorchDLPackPyCExporter(obj, &cache, env_stream) != 0) {
    return -1;
  }
  return DLPackPyIntrusiveCacheAttach(py_obj, cache, out);
}

int64_t TorchDLPackPyCExporterPtr(bool cached) {
  if (cached) {
    return reinterpret_cast<int64_t>(TorchDLPackPyCExporterCached);
  }
  return reinterpret_cast<int64_t>(TorchDLPackPyCExporter);
}

inline int64_t TorchDLPackPyCExporterPtrCached(PyObject* py_obj) {
  return reinterpret_cast<int64_t>(TorchDLPackPyCExporterCached);
}

void dlpack_py_c_exporter_bench(int64_t py_obj_ptr, int64_t dlpack_c_exporter, int repeat) {
  DLPackPyCExporter exporter = reinterpret_cast<DLPackPyCExporter>(dlpack_c_exporter);
  void* py_obj = reinterpret_cast<void*>(py_obj_ptr);
  for (int i = 0; i < repeat; i++) {
    DLManagedTensorVersioned* dlpack;
    (*exporter)(py_obj, &dlpack, nullptr);
    dlpack->deleter(dlpack);
  }
}

void refcount_update(int64_t py_obj_ptr) {
  PyObject* py_obj = reinterpret_cast<PyObject*>(py_obj_ptr);
  std::cout << "refcount=" << Py_REFCNT(py_obj) << std::endl;
  Py_INCREF(py_obj);
  Py_DECREF(py_obj);
}
    """
    module = cpp_extension.load_inline(
        name="to_dlpack",
        cpp_sources=cpp_source,
        functions=["dlpack_cpp_exporter_bench", "TorchDLPackPyCExporterPtr", "dlpack_py_c_exporter_bench", "refcount_update"],
        extra_cflags=["-O3"],
        extra_include_paths=tvm_ffi.libinfo.include_paths() + cpp_extension.include_paths("cuda"),
        verbose=True,
    )
    return module


mod = load_to_dlpack()
tvm_ffi.core._torch_dlpack_c_exporter_ptr = mod.TorchDLPackPyCExporterPtr(False)
# tvm_ffi.core._torch_dlpack_c_exporter_ptr = mod.TorchDLPackPyCExporterPtr(True)
