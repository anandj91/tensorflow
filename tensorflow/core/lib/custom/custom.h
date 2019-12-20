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

#ifndef TENSORFLOW_CORE_LIB_CUSTOM_CUSTOM_H_
#define TENSORFLOW_CORE_LIB_CUSTOM_CUSTOM_H_

#include <cmath>
#include <complex>

#include "tensorflow/core/platform/byte_order.h"

#ifdef __CUDACC__
// All functions callable from CUDA code must be qualified with __device__
#define CUSTOM_DEVICE_FUNC __host__ __device__

#else
#define CUSTOM_DEVICE_FUNC

#endif

namespace Eigen {
struct half;
}

namespace tensorflow {

// Single precision complex.
typedef std::complex<float> complex64;
// Double precision complex.
typedef std::complex<double> complex128;

// see framework/custom.h for description.
struct custom {
  // The default constructor must yield a zero value, not an uninitialized
  // value; some TF kernels use T() as a zero value.
  CUSTOM_DEVICE_FUNC custom() : value(ZERO_VALUE) {}

  CUSTOM_DEVICE_FUNC static custom truncate_to_custom(const float v) {
    custom output;
    if (float_isnan(v)) {
      output.value = NAN_VALUE;
      return output;
    }
    const uint16_t* p = reinterpret_cast<const uint16_t*>(&v);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    output.value = p[0];
#else
    output.value = p[1];
#endif
    return output;
  }

  CUSTOM_DEVICE_FUNC explicit custom(const float v) {
    value = round_to_custom(v).value;
  }

  CUSTOM_DEVICE_FUNC explicit custom(const double val)
      : custom(static_cast<float>(val)) {}
  // Following the convention of numpy, converting between complex and
  // float will lead to loss of imag value.
  CUSTOM_DEVICE_FUNC explicit custom(const complex64& val)
      : custom(val.real()) {}

  CUSTOM_DEVICE_FUNC explicit custom(const complex128& val)
      : custom(static_cast<float>(val.real())) {}

  CUSTOM_DEVICE_FUNC explicit custom(const unsigned short val)
      : custom(static_cast<float>(val)) {}

  CUSTOM_DEVICE_FUNC explicit custom(const unsigned int val)
      : custom(static_cast<float>(val)) {}

  CUSTOM_DEVICE_FUNC explicit custom(const int val)
      : custom(static_cast<float>(val)) {}

  CUSTOM_DEVICE_FUNC explicit custom(const long val)
      : custom(static_cast<float>(val)) {}

  CUSTOM_DEVICE_FUNC explicit custom(const long long val)
      : custom(static_cast<float>(val)) {}

  template <class T>
  CUSTOM_DEVICE_FUNC explicit custom(const T& val)
      : custom(static_cast<float>(val)) {}

  CUSTOM_DEVICE_FUNC explicit operator float() const {
    float result = 0;

    uint16_t* q = reinterpret_cast<uint16_t*>(&result);

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    q[0] = value;
#else
    q[1] = value;
#endif
    return result;
  }

  CUSTOM_DEVICE_FUNC explicit operator bool() const {
    return static_cast<bool>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator Eigen::half() const;

  CUSTOM_DEVICE_FUNC explicit operator short() const {
    return static_cast<short>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator int() const {
    return static_cast<int>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator long() const {
    return static_cast<long>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator char() const {
    return static_cast<char>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator signed char() const {
    return static_cast<signed char>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator unsigned char() const {
    return static_cast<unsigned char>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator unsigned short() const {
    return static_cast<unsigned short>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator unsigned int() const {
    return static_cast<unsigned int>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator unsigned long() const {
    return static_cast<unsigned long>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator unsigned long long() const {
    return static_cast<unsigned long long>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator long long() const {
    return static_cast<long long>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator double() const {
    return static_cast<double>(float(*this));
  }

  CUSTOM_DEVICE_FUNC explicit operator complex64() const {
    return complex64(float(*this), float(0.0));
  }

  CUSTOM_DEVICE_FUNC explicit operator complex128() const {
    return complex128(double(*this), double(0.0));
  }

  union FP32 {
    unsigned int u;
    float f;
  };

  // Converts a float point to custom, with round-nearest-to-even as rounding
  // method.
  // TODO: There is a slightly faster implementation (8% faster on CPU)
  // than this (documented in cl/175987786), that is exponentially harder to
  // understand and document. Switch to the faster version when converting to
  // BF16 becomes compute-bound.
  CUSTOM_DEVICE_FUNC static custom round_to_custom(float v) {
    uint32_t input;
    FP32 f;
    f.f = v;
    input = f.u;
    custom output;

    if (float_isnan(v)) {
      // If the value is a NaN, squash it to a qNaN with msb of fraction set,
      // this makes sure after truncation we don't end up with an inf.
      //
      // qNaN magic: All exponent bits set + most significant bit of fraction
      // set.
      output.value = 0x7fc0;
    } else {
      // Fast rounding algorithm that rounds a half value to nearest even. This
      // reduces expected error when we convert a large number of floats. Here
      // is how it works:
      //
      // Definitions:
      // To convert a float 32 to custom, a float 32 can be viewed as 32 bits
      // with the following tags:
      //
      // Sign |  Exp (8 bits) | Frac (23 bits)
      //  S     EEEEEEEE         FFFFFFLRTTTTTTTTTTTTTTT
      //
      //  S: Sign bit.
      //  E: Exponent bits.
      //  F: First 6 bits of fraction.
      //  L: Least significant bit of resulting custom if we truncate away the
      //  rest of the float32. This is also the 7th bit of fraction
      //  R: Rounding bit, 8th bit of fraction.
      //  T: Sticky bits, rest of fraction, 15 bits.
      //
      // To round half to nearest even, there are 3 cases where we want to round
      // down (simply truncate the result of the bits away, which consists of
      // rounding bit and sticky bits) and two cases where we want to round up
      // (truncate then add one to the result).
      //
      // The fast converting algorithm simply adds lsb (L) to 0x7fff (15 bits of
      // 1s) as the rounding bias, adds the rounding bias to the input, then
      // truncates the last 16 bits away.
      //
      // To understand how it works, we can analyze this algorithm case by case:
      //
      // 1. L = 0, R = 0:
      //   Expect: round down, this is less than half value.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input may create any carry, depending on
      //   whether there is any value set to 1 in T bits.
      //   - R may be set to 1 if there is a carry.
      //   - L remains 0.
      //   - Note that this case also handles Inf and -Inf, where all fraction
      //   bits, including L, R and Ts are all 0. The output remains Inf after
      //   this algorithm.
      //
      // 2. L = 1, R = 0:
      //   Expect: round down, this is less than half value.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 1 = 0x8000
      //   - Adding rounding bias to input doesn't change sticky bits but
      //   adds 1 to rounding bit.
      //   - L remains 1.
      //
      // 3. L = 0, R = 1, all of T are 0:
      //   Expect: round down, this is exactly at half, the result is already
      //   even (L=0).
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input sets all sticky bits to 1, but
      //   doesn't create a carry.
      //   - R remains 1.
      //   - L remains 0.
      //
      // 4. L = 1, R = 1:
      //   Expect: round up, this is exactly at half, the result needs to be
      //   round to the next even number.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 1 = 0x8000
      //   - Adding rounding bias to input doesn't change sticky bits, but
      //   creates a carry from rounding bit.
      //   - The carry sets L to 0, creates another carry bit and propagate
      //   forward to F bits.
      //   - If all the F bits are 1, a carry then propagates to the exponent
      //   bits, which then creates the minimum value with the next exponent
      //   value. Note that we won't have the case where exponents are all 1,
      //   since that's either a NaN (handled in the other if condition) or inf
      //   (handled in case 1).
      //
      // 5. L = 0, R = 1, any of T is 1:
      //   Expect: round up, this is greater than half.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input creates a carry from sticky bits,
      //   sets rounding bit to 0, then create another carry.
      //   - The second carry sets L to 1.
      //
      // Examples:
      //
      //  Exact half value that is already even:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0     1000000000000000
      //
      //     This falls into case 3. We truncate the rest of 16 bits and no
      //     carry is created into F and L:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0
      //
      //  Exact half value, round to next even number:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 0 1     1000000000000000
      //
      //     This falls into case 4. We create a carry from R and T,
      //     which then propagates into L and F:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0
      //
      //
      //  Max denormal value round to min normal value:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      1 1 1 1 1 1 1     1111111111111111
      //
      //     This falls into case 4. We create a carry from R and T,
      //     propagate into L and F, which then propagates into exponent
      //     bits:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 1      0 0 0 0 0 0 0
      //
      //  Max normal value round to Inf:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     1 1 1 1 1 1 1 0      1 1 1 1 1 1 1     1111111111111111
      //
      //     This falls into case 4. We create a carry from R and T,
      //     propagate into L and F, which then propagates into exponent
      //     bits:
      //
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     1 1 1 1 1 1 1 1      0 0 0 0 0 0 0
      //
      //
      // Least significant bit of resulting bfloat.
      uint32_t lsb = (input >> 16) & 1;
      uint32_t rounding_bias = 0x7fff + lsb;
      input += rounding_bias;
      output.value = static_cast<uint16_t>(input >> 16);
    }
    return output;
  }

  static custom epsilon() {
    custom x;
    x.value = 0x3c00;  // 0x1.0p-7
    return x;
  }

  static custom highest() {
    custom x;
    x.value = 0x7F7F;  // 0x1.FEp127
    return x;
  }

  static custom lowest() {
    custom x;
    x.value = 0xFF7F;  // -0x1.FEp127
    return x;
  }

  static custom min_positive_normal() {
    custom x;
    x.value = 0x0080;  // 0x1p-126
    return x;
  }

  bool IsZero() const { return (value & 0x7FFF) == ZERO_VALUE; }

  uint16_t value;

  // A value that represents "not a number".
  static const uint16_t NAN_VALUE = 0x7FC0;

 private:
  // A value that represents "zero".
  static const uint16_t ZERO_VALUE = 0;

  CUSTOM_DEVICE_FUNC static bool float_isnan(const float& x) {
#ifdef __CUDA_ARCH__
    return ::isnan(x);
#else
    return std::isnan(x);
#endif
  }
};

CUSTOM_DEVICE_FUNC inline std::ostream& operator<<(std::ostream& os,
                                                const custom& dt) {
  os << static_cast<float>(dt);
  return os;
}

CUSTOM_DEVICE_FUNC inline custom operator+(custom a, custom b) {
  return custom(static_cast<float>(a) + static_cast<float>(b));
}
CUSTOM_DEVICE_FUNC inline custom operator+(custom a, int b) {
  return custom(static_cast<float>(a) + static_cast<float>(b));
}
CUSTOM_DEVICE_FUNC inline custom operator+(int a, custom b) {
  return custom(static_cast<float>(a) + static_cast<float>(b));
}
CUSTOM_DEVICE_FUNC inline custom operator-(custom a, custom b) {
  return custom(static_cast<float>(a) - static_cast<float>(b));
}
CUSTOM_DEVICE_FUNC inline custom operator*(custom a, custom b) {
  return custom(static_cast<float>(a) * static_cast<float>(b));
}
CUSTOM_DEVICE_FUNC inline custom operator/(custom a, custom b) {
  return custom(static_cast<float>(a) / static_cast<float>(b));
}
CUSTOM_DEVICE_FUNC inline custom operator-(custom a) {
  a.value ^= 0x8000;
  return a;
}
CUSTOM_DEVICE_FUNC inline bool operator<(custom a, custom b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
CUSTOM_DEVICE_FUNC inline bool operator<=(custom a, custom b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
CUSTOM_DEVICE_FUNC inline bool operator==(custom a, custom b) {
  return static_cast<float>(a) == static_cast<float>(b);
}
CUSTOM_DEVICE_FUNC inline bool operator!=(custom a, custom b) {
  return static_cast<float>(a) != static_cast<float>(b);
}
CUSTOM_DEVICE_FUNC inline bool operator>(custom a, custom b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
CUSTOM_DEVICE_FUNC inline bool operator>=(custom a, custom b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}
CUSTOM_DEVICE_FUNC inline custom& operator+=(custom& a, custom b) {
  a = a + b;
  return a;
}
CUSTOM_DEVICE_FUNC inline custom& operator-=(custom& a, custom b) {
  a = a - b;
  return a;
}
CUSTOM_DEVICE_FUNC inline custom operator++(custom& a) {
  a += custom(1);
  return a;
}
CUSTOM_DEVICE_FUNC inline custom operator--(custom& a) {
  a -= custom(1);
  return a;
}
CUSTOM_DEVICE_FUNC inline custom operator++(custom& a, int) {
  custom original_value = a;
  ++a;
  return original_value;
}
CUSTOM_DEVICE_FUNC inline custom operator--(custom& a, int) {
  custom original_value = a;
  --a;
  return original_value;
}
CUSTOM_DEVICE_FUNC inline custom& operator*=(custom& a, custom b) {
  a = a * b;
  return a;
}
CUSTOM_DEVICE_FUNC inline custom& operator/=(custom& a, custom b) {
  a = a / b;
  return a;
}
}  // end namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::custom> {
  size_t operator()(const tensorflow::custom& v) const {
    return hash<float>()(static_cast<float>(v));
  }
};

using tensorflow::custom;
inline bool isinf(const custom& a) { return std::isinf(float(a)); }
inline bool isnan(const custom& a) { return std::isnan(float(a)); }
inline bool isfinite(const custom& a) { return std::isfinite(float(a)); }
inline custom abs(const custom& a) { return custom(std::abs(float(a))); }
inline custom exp(const custom& a) { return custom(std::exp(float(a))); }
inline custom log(const custom& a) { return custom(std::log(float(a))); }
inline custom log10(const custom& a) {
  return custom(std::log10(float(a)));
}
inline custom sqrt(const custom& a) {
  return custom(std::sqrt(float(a)));
}
inline custom pow(const custom& a, const custom& b) {
  return custom(std::pow(float(a), float(b)));
}
inline custom sin(const custom& a) { return custom(std::sin(float(a))); }
inline custom cos(const custom& a) { return custom(std::cos(float(a))); }
inline custom tan(const custom& a) { return custom(std::tan(float(a))); }
inline custom tanh(const custom& a) {
  return custom(std::tanh(float(a)));
}
inline custom floor(const custom& a) {
  return custom(std::floor(float(a)));
}
inline custom ceil(const custom& a) {
  return custom(std::ceil(float(a)));
}
}  // namespace std

#endif  // TENSORFLOW_CORE_LIB_CUSTOM_CUSTOM_H_
