# TVM FFI Documentation Outline

## Manual zone

The following text contains instruction for agents to create the docs
- Get Started
  - Installation
  - Quick start
    - Go through the examples/get_started concepts
    - Add what to do next
  - Overview
    - What is TVM FFI
    - Why use TVM FFI
    - How TVM FFI works

- Guides
  - Packaging
  - Using C++ API
  - Custom Object
  - Error Handling
  - Python Integration

- Concepts
  - ABI Specification
    - Discuss that the ABI is specified in c_api.h
    - TVMFFIAny Value storage convention
      - Discuss the encoding
      - Give examples about how POD type is stored
    - Object convention
      - Give example about how Shape is stored
    - Function calling convention
      - Show C style code in caller and callee
      - Error handling
  - Type System Fundamentals

- API Reference

