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
  - header-only implementation (no `.cc`)

## ConfigSchema API (Design)

`ConfigSchema` is internal C++ infra (not exposed to Python object system).

```cpp
namespace tvm {
namespace ir {

using ConfigObject = ffi::json::Object;
using FConfigCanonicalizer = ffi::TypedFunction<ConfigObject(ConfigObject)>;

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

## `config_schema.h` Public Header Draft (Fully Documented)

```cpp
/*!
 * \file tvm/ir/config_schema.h
 * \brief Minimal schema for dynamic config canonicalization and validation.
 *
 * This utility is intended for dynamic map-like configs (e.g. Target/PassContext options),
 * where we still want type checking, default values, and canonicalization.
 *
 * Design goals:
 * - Header-only and minimal.
 * - Reflection-like declaration style (`def_option` + optional traits).
 * - Type validation/coercion implicit from template argument `T`.
 * - Throw-based error reporting (`TypeError`/`ValueError`) with contextual messages.
 */
#ifndef TVM_IR_CONFIG_SCHEMA_H_
#define TVM_IR_CONFIG_SCHEMA_H_

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace ir {

/*!
 * \brief JSON-like config object used by schema resolution.
 * \note Currently map-backed in ffi.
 */
using ConfigObject = ffi::json::Object;

/*!
 * \brief Whole-config canonicalizer hook.
 *
 * Runs before per-field defaulting and validation.
 * Typical use cases:
 * - Fill derived fields.
 * - Rewrite aliases.
 * - Normalize values based on multiple fields.
 */
using FConfigCanonicalizer = ffi::TypedFunction<ConfigObject(ConfigObject)>;

/*!
 * \brief Dynamic config schema for map-like options.
 *
 * The schema supports:
 * - Option declaration (`def_option<T>`)
 * - Optional canonicalizer (`set_canonicalizer`)
 * - Resolution (`Resolve`) that performs canonicalization, defaulting, and validation.
 */
class ConfigSchema {
 public:
  /*!
   * \brief Declare a typed option.
   *
   * Validation/coercion is implicitly generated from `T`.
   * Additional optional traits may be supplied (e.g. `refl::DefaultValue`).
   *
   * \tparam T The canonical value type of this option.
   * \tparam Traits Optional metadata/traits.
   *
   * \param key Option key.
   * \param doc Optional doc string, can be null.
   * \param traits Optional traits.
   *
   * \return Reference to `*this` for chaining.
   */
  template <typename T, typename... Traits>
  ConfigSchema& def_option(ffi::String key, const char* doc = nullptr, Traits... traits);

  /*!
   * \brief Set whole-config canonicalizer.
   *
   * \param f Canonicalizer callback.
   *
   * \return Reference to `*this` for chaining.
   */
  ConfigSchema& set_canonicalizer(FConfigCanonicalizer f);

  /*!
   * \brief Resolve config into canonical validated form.
   *
   * Resolve performs:
   * 1) Whole-config canonicalization.
   * 2) Per-option defaulting and type validation/coercion.
   * 3) Unknown key checking (policy-defined).
   *
   * \param cfg Input config object.
   *
   * \return Canonical validated config object.
   *
   * \throws TypeError if value cannot be coerced to declared type.
   * \throws ValueError if semantic validation fails or required values are missing.
   */
  ConfigObject Resolve(ConfigObject cfg) const;

 private:
  /*!
   * \brief Internal option entry.
   */
  struct OptionEntry {
    /*! \brief Option key. */
    ffi::String name;
    /*! \brief Optional doc string. */
    ffi::Optional<ffi::String> doc;
    /*! \brief Per-option validator/coercer (`Any -> Any`). */
    ffi::TypedFunction<ffi::Any(ffi::Any)> validator;
    /*! \brief Optional lazy default factory. */
    ffi::Optional<ffi::TypedFunction<ffi::Any()>> default_factory;
    /*! \brief Whether this option must be present (or defaulted). */
    bool required{false};
  };

  /*! \brief Declared options by key. */
  std::unordered_map<std::string, OptionEntry> options_;
  /*! \brief Optional whole-config canonicalizer. */
  ffi::Optional<FConfigCanonicalizer> canonicalizer_;
  /*! \brief Whether unknown keys trigger an error. */
  bool error_on_unknown_{true};
};

}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_CONFIG_SCHEMA_H_
```

### Trait Style

Follow reflection-style optional varargs traits:

- `refl::DefaultValue(...)`
- optional additional metadata

No separate "set validator" / "set default factory" call is required for common cases.
Type validation/coercion is implicit from `T`.
Validation traits can be added later to compose into a single validator.

## ConfigSchema Impl Sketch (Header-Only)

```cpp
class ConfigSchema {
 public:
  /*!
   * \brief Declare a typed option.
   * \tparam T The canonical value type for this option.
   * \tparam Traits Optional traits (e.g. refl::DefaultValue).
   * \param key Option key.
   * \param doc Optional doc string.
   */
  template <typename T, typename... Traits>
  ConfigSchema& def_option(ffi::String key, const char* doc = nullptr, Traits... traits) {
    options_.emplace(std::string(key), MakeEntry<T>(key, doc, traits...));
    return *this;
  }

  /*! \brief Set whole-object canonicalizer. */
  ConfigSchema& set_canonicalizer(FConfigCanonicalizer f) {
    canonicalizer_ = f;
    return *this;
  }

  /*!
   * \brief Canonicalize, validate, and default a config object.
   * \throws ValueError/TypeError with option context.
   */
  ConfigObject Resolve(ConfigObject cfg) const {
    // 1) whole-object canonicalization
    if (canonicalizer_) {
      cfg = canonicalizer_.value()(cfg);
    }

    // 2) per-option default + validation
    for (const auto& kv : options_) {
      const OptionEntry& e = kv.second;
      if (!cfg.count(e.name)) {
        if (e.default_factory) {
          cfg.Set(e.name, e.default_factory.value()());
        } else if (e.required) {
          TVM_FFI_THROW(ValueError) << "Missing required option: " << e.name;
        } else {
          continue;
        }
      }
      try {
        cfg.Set(e.name, e.validator(cfg.at(e.name)));
      } catch (const Error& err) {
        // Type mismatch -> TypeError, semantic mismatch -> ValueError
        throw Error(err.kind(), std::string("Invalid option \"") + std::string(e.name) +
                                   "\": " + err.message(), err.backtrace());
      }
    }

    // 3) unknown-key policy
    if (error_on_unknown_) {
      for (const auto& kv : cfg) {
        ICHECK(options_.count(std::string(kv.first)))
            << "Unknown option: " << kv.first;
      }
    }
    return cfg;
  }

 private:
  struct OptionEntry {
    /*! \brief Option key. */
    ffi::String name;
    /*! \brief Optional doc string. */
    ffi::Optional<ffi::String> doc;
    /*! \brief Field validator/coercer (Any -> Any). */
    ffi::TypedFunction<ffi::Any(ffi::Any)> validator;
    /*! \brief Optional default value factory. */
    ffi::Optional<ffi::TypedFunction<ffi::Any()>> default_factory;
    /*! \brief Whether this option is required. */
    bool required{false};
  };

  template <typename T, typename... Traits>
  OptionEntry MakeEntry(ffi::String key, const char* doc, Traits... traits) {
    OptionEntry e;
    e.name = key;
    if (doc != nullptr && doc[0] != '\0') {
      e.doc = ffi::String(doc);
    }
    e.validator = MakeImplicitTypeValidator<T>();
    ApplyTraits<T>(&e, traits...);  // parses refl::DefaultValue and other metadata traits
    return e;
  }

 private:
  /*! \brief Declared options by key. */
  std::unordered_map<std::string, OptionEntry> options_;
  /*! \brief Optional whole-object canonicalizer. */
  ffi::Optional<FConfigCanonicalizer> canonicalizer_;
  /*! \brief Unknown-key behavior (true => error). */
  bool error_on_unknown_{true};
};
```

## TargetKind API (Design)

```cpp
namespace tvm {
namespace target {

using FTargetCanonicalizer =
    ffi::TypedFunction<ffi::json::Object(ffi::json::Object)>;

class TargetKindRegEntry {
 public:
  template <typename T, typename... Traits>
  TargetKindRegEntry& add_attr_option(const ffi::String& key, const char* doc = nullptr,
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
    .add_attr_option<int64_t>("num-cores", "CPU core count",
                              refl::DefaultValue(1))
    .set_default_keys({"cpu"})
    .set_target_canonicalizer(tvm::target::canonicalize::llvm::Canonicalize);

TVM_REGISTER_TARGET_KIND("cuda", kDLCUDA)
    .add_attr_option<ffi::String>("arch", "GPU arch, e.g. sm_90")
    .add_attr_option<int64_t>("max_num_threads", "Thread upper bound",
                              refl::DefaultValue(1024))
    .set_default_keys({"cuda", "gpu"})
    .set_target_canonicalizer(tvm::target::canonicalize::cuda::Canonicalize);
```

## PassContext Reuse (Conceptual)

`PassContext` can adopt the same schema backend while preserving current API:

- `RegisterConfigOption(key, type_str, legalization)` maps to `def_option<T>(key, type_str, ...)`
- legalization callback maps to option validator (`Any -> Any`)
- `PassConfigManager::Legalize` becomes `pass_schema.Resolve(config)`

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
- `ConfigSchema` is intentionally designed as header-only and minimal.
