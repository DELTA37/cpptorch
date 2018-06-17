#include <kaspar/algorithm/linear_model.hpp>
#include <kaspar/algorithm/random.hpp>
#include <iostream>

using namespace kaspar;

int test_linear(void) {
  int batch_size = 3;
  float x[batch_size];
  float y[batch_size];
  float g[batch_size];
  Dense<1, 1, false> d(x, y, g, NULL, batch_size, 1e-2);
  for (int i = 0; i < 1000; ++i) {
    for (size_t b = 0; b < batch_size; ++b) {
      x[b] = b * i / 100.0f;
    }
    d.forward();
    for (size_t b = 0; b < batch_size; ++b) {
      g[b] = 2 * (y[b] - 2 * x[b]);
    }
    d.backward(false);
    float s = 0;
    for (size_t b = 0; b < batch_size; ++b) {
      s += (y[b] - 2 * x[b]) * (y[b] - 2 * x[b]);
    }
    std::cout << "loss: " << s << std::endl;
  }
  d.print_weights();
  return 0;
}

int test_permutation() {
  constexpr int n = 10;
  float x[n];
  RandomPermutation<n> rp(x, x, x, x, 1);
  for (size_t i = 0; i < n; ++i) {
    x[i] = i;
  }
  for (size_t t = 0; t < 100; ++t) {
    rp.forward();
    for (size_t i = 0; i < n; ++i) {
      std::cout << x[i] << " ";
    }
    std::cout << std::endl;
    rp.backward(true);
    for (size_t i = 0; i < n; ++i) {
      std::cout << x[i] << " ";
    }
    std::cout << std::endl;
  }
}

int main(void) {
  return test_permutation();
}
