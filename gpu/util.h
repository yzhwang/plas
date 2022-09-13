#ifndef PLAS_GPU_UTIL_H_
#define PLAS_GPU_UTIL_H_

#include <cstdarg>
#include <string>

namespace plas {
namespace gpu {

namespace detail {

inline std::string stringprintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  int len = vsnprintf(0, 0, format, args);
  va_end(args);

  // allocate space.
  std::string text;
  text.resize(len);

  va_start(args, format);
  vsnprintf(&text[0], len + 1, format, args);
  va_end(args);

  return text;
}

}  // namespace detail

}  // namespace gpu
}  // namespace plas

#endif
