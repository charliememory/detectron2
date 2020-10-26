#include <cuda_runtime_api.h>

namespace densepose {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace densepose
