#include <pybind11/pybind11.h>
#include <ATen/DLConvertor.h>


void toDLPack(at::Tensor& tensor) {
  DLManagedTensor* dlpack = at::toDLPack(tensor);
  dlpack->deleter(dlpack);
}

