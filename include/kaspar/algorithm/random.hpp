#pragma once
#include <random>
#include <algorithm>

namespace kaspar {

template<size_t in>
class RandomPermutation {
public:
  RandomPermutation(
    float* input, float* output, float* grad_input, float* grad_output, 
    size_t batch_size=10
  ) : batch_size(batch_size), 
      input(input),
      output(output),
      grad_input(grad_input),
      grad_output(grad_output) {
    for (size_t i = 0; i < in; ++i) {
      permutation[i] = i;
    }
  }

  void forward() {
    _generate_random_numbers();
    if (input != output) {
      for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t i = 0; i < in; ++i) {
          output[batch * in + i] = input[batch * in + i];
        }
      }
    }
    for (size_t batch = 0; batch < batch_size; ++batch) {
      for (int i = in - 1; i >= 0; --i) {
        std::swap(output[batch * in + i], output[batch * in + permutation[i]]);
      }
    }
  }
  
  void backward(bool grad_output_required=true) {
    if (!grad_output_required) {
      return;
    }
    if (grad_input != grad_output) {
      for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t i = 0; i < in; ++i) {
          grad_output[batch * in + i] = grad_input[batch * in + i];
        }
      }
    }
    for (size_t batch = 0; batch < batch_size; ++batch) {
      for (int i = 0; i < in; ++i) {
        std::swap(output[batch * in + permutation[i]], output[batch * in + i]);
      }
    }
  }

  void _generate_random_numbers() {
    /*
     * Knuth algorithm
     */

    int j = 0;
    for (size_t i = 0; i < in; ++i) {
      permutation[i] = std::rand() % (i + 1);
    }
  }

private:
  float* input; // [batch_size x in]
  float* output; // [batch_size x out]

  float* grad_input; // [batch_size x out]
  float* grad_output; // [batch_size x in]

  size_t batch_size;
  size_t permutation[in];
};

} // kaspar
