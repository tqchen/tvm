#include <dawn/webgpu.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {
namespace webgpu {
void Init() {
    DawnProcTable procs = dawn::native::GetProcs();
    dawnProcSetProcs(&procs);

    // 1. We create a descriptor
    WGPUInstanceDescriptor desc = {};
    desc.nextInChain = nullptr;

    // 2. We create the instance using this descriptor
    WGPUInstance instance = wgpuCreateInstance(&desc);

    // 3. We can check whether there is actually an instance created
    if (!instance) {
        std::cerr << "Could not initialize WebGPU!" << std::endl;
    }

    // 4. Display the object (WGPUInstance is a simple pointer, it may be
    // copied around without worrying about its size).
    std::cout << "WGPU instance: " << instance << std::endl;
}

TVM_REGISTER_GLOBAL("tvm.runtime.webgpu_init").set_body_typed(Init);
}
}
}
