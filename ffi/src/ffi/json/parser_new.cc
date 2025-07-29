/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
 * \file src/ffi/json/parser.cc
 *
 * \brief A minimalistic JSON parser implementation.
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/json/json.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/reflection/registry.h>

#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <limits>


namespace tvm {
namespace ffi {
namespace json {

/*!
 * \brief Helper class to parse a JSON string.
 */
class JSONStringTokenizer {
 public:
  JSONStringTokenizer(const char* begin, const char* end)
      : begin_(begin), end_(end) {
  }

  void Process() {
    size_t total_size = end_ - begin_;
    // guess the possible totoal quote positions
    quote_pos_.reserve(total_size / 8);
    FindQuotePos(begin_, end_, 0);
    std::cout << "quote_pos_.size(): " << quote_pos_.size() << " total_size: " << total_size << std::endl;
  }

  void FindQuotePos(const char* begin, const char* end, size_t offset) {
    size_t size = end - begin;
    for (size_t i = 0; i < size; ++i) {
      // if (begin[i] == '\\') {
      //   i += 1;
      // }
      if (begin[i] == '\"') {
        quote_pos_.push_back(i + offset);
      }
    }
  }

  void FindQuoteByI64BitTrick(const char* begin, const char* end, size_t offset) {
    const char* curr = begin;
    size_t remaining = end - begin;

    // Process head bytes until we reach 8-byte alignment
    uintptr_t curr_addr = reinterpret_cast<uintptr_t>(curr);
    size_t head_bytes = (8 - (curr_addr % 8)) % 8;
    head_bytes = std::min(head_bytes, remaining);

    // Process unaligned head
    for (size_t i = 0; i < head_bytes; ++i) {
      if (curr[i] == '\"') {
        quote_pos_.push_back((curr - begin) + i + offset);
      }
    }

    curr += head_bytes;
    remaining -= head_bytes;

    // Process 8 bytes at a time using 64-bit operations (now aligned)
    while (remaining >= 8) {
      // Direct cast to uint64_t* since we're now aligned
      uint64_t chunk = *reinterpret_cast<const uint64_t*>(curr);

      // Create a mask where each byte is 0x22 (quote character)
      const uint64_t quote_pattern = 0x2222222222222222ULL;

      // XOR to find quote bytes (quotes become 0x00)
      uint64_t xor_result = chunk ^ quote_pattern;

      // Use bit trick to detect zero bytes
      // (x - 0x01010101) & ~x & 0x80808080 has high bit set for each zero byte
      const uint64_t sub_pattern = 0x0101010101010101ULL;
      const uint64_t high_bits = 0x8080808080808080ULL;
      uint64_t zero_mask = (xor_result - sub_pattern) & ~xor_result & high_bits;

      // Extract positions of quotes
      while (zero_mask != 0) {
        // Find the position of the lowest set bit (rightmost quote)
        int bit_pos = __builtin_ctzll(zero_mask);
        int byte_pos = bit_pos / 8;
        quote_pos_.push_back((curr - begin) + byte_pos + offset);

        // Clear this bit and continue
        zero_mask &= zero_mask - 1;
      }

      curr += 8;
      remaining -= 8;
    }

    // Process remaining tail bytes (less than 8) with simple loop
    for (size_t i = 0; i < remaining; ++i) {
      if (curr[i] == '\"') {
        quote_pos_.push_back((curr - begin) + i + offset);
      }
    }
  }

 private:
  /*! \brief The beginning of the string */
  const char* begin_;
  /*! \brief End of the string */
  const char* end_;
  // quote position
  std::vector<size_t> quote_pos_;
};

class JSONParser {
 public:
};


TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.json.NewParse", [](const String& json_str) {
    JSONStringTokenizer tokenizer(json_str.data(), json_str.data() + json_str.size());
    tokenizer.Process();
  });
});

}  // namespace json
}  // namespace ffi
}  // namespace tvm
