# Target Refactor Design Draft

## Scope

Design-only draft (no implementation yet) for simplifying `Target` and introducing shared config validation infra.

## High-Level Decisions

- Keep `Target` explicit fields:
  - `kind`
  - `host` (optional)
  - `keys` (dispatch keys)
  - `attrs` (options + derived `feature.*`)
- Remove separate `features` storage; derived features go into `attrs` with `feature.` prefix.
- Keep `default_device_type` on `TargetKind`.
- Keep `add_attr_option` style in target registration.
- Rename parser alias to canonicalizer:
  - `FTargetCanonicalizer`
- Keep naming as config object:
  - `Target::FromConfig(...)`
  - `Target::ToConfig(...)`
- Place schema infra under IR:
  - `include/tvm/ir/config_schema.h`
  - `src/ir/config_schema.cc`

## ConfigSchema API (Design)

`ConfigSchema` is internal C++ infra (not exposed to Python object system).

```cpp
namespace tvm {
namespace ir {

using ConfigObject = ffi::json::Object;
using FConfigCanonicalizer = ffi::TypedFunction<ConfigObject(ConfigObject)>;

// Type validation is implicit from template type T.
// Extra constraints are optional traits.
template <typename T>
using FConfigConstraint = ffi::TypedFunction<void(T)>;

class ConfigSchema {
 public:
  // doc is optional (empty by default), matching reflection style ergonomics.
  template <typename T, typename... Traits>
  ConfigSchema& def_option(ffi::String key, ffi::String doc = "", Traits... traits);

  ConfigSchema& set_canonicalizer(FConfigCanonicalizer f);

  // canonicalize + validate/coerce + apply defaults
  ConfigObject Resolve(ConfigObject cfg) const;
};

}  // namespace ir
}  // namespace tvm
```

### Trait Style

Follow reflection-style optional varargs traits:

- `refl::DefaultValue(...)`
- optional constraints
- optional additional metadata

No separate "set validator" / "set default factory" call is required for common cases.

## TargetKind API (Design)

```cpp
namespace tvm {
namespace target {

using FTargetCanonicalizer =
    ffi::TypedFunction<ffi::json::Object(ffi::json::Object)>;

class TargetKindRegEntry {
 public:
  template <typename T, typename... Traits>
  TargetKindRegEntry& add_attr_option(const ffi::String& key,
                                      ffi::String doc = "",
                                      Traits... traits);

  TargetKindRegEntry& set_default_keys(std::vector<ffi::String> keys);
  TargetKindRegEntry& set_default_device_type(int device_type);
  TargetKindRegEntry& set_target_canonicalizer(FTargetCanonicalizer f);
};

}  // namespace target
}  // namespace tvm
```

## Example Usage

```cpp
TVM_REGISTER_TARGET_KIND("llvm", kDLCPU)
    .add_attr_option<ffi::String>("mcpu", "LLVM cpu name")
    .add_attr_option<ffi::String>("mtriple", "LLVM target triple")
    .add_attr_option<ffi::Array<ffi::String>>("mattr", "LLVM feature list")
    .add_attr_option<int64_t>("num-cores", "CPU core count", refl::DefaultValue(int64_t(1)))
    .set_default_keys({"cpu"})
    .set_target_canonicalizer(tvm::target::canonicalize::llvm::Canonicalize);

TVM_REGISTER_TARGET_KIND("cuda", kDLCUDA)
    .add_attr_option<ffi::String>("arch", "GPU arch, e.g. sm_90")
    .add_attr_option<int64_t>("max_num_threads", "Thread upper bound",
                              refl::DefaultValue(int64_t(1024)))
    .set_default_keys({"cuda", "gpu"})
    .set_target_canonicalizer(tvm::target::canonicalize::cuda::Canonicalize);
```

## Construction Flow (Conceptual)

1. Decode user input into `ConfigObject`.
2. Run target-kind canonicalizer.
3. Run schema resolve (`Resolve`) for type/default checks.
4. Materialize `Target` fields:
   - `kind`, `host`, `keys`, `attrs`.

## Printing and Equality Policy

- Default print should remain concise and hide `feature.*`.
- Structural equality/hash should not rely on `target->str()`.
- `keys` are explicit dispatch keys, separate from attrs.

## Notes

- This draft intentionally keeps current registration style and vocabulary (`add_attr_option`) while clarifying backend architecture.
- Implementation details are intentionally omitted.
