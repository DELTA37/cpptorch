#pragma once
#include <algorithm>
#include <utility>
#include <vector>
#include <atomic>
#include <iostream>


namespace kaspar {

float atomic_subf(std::atomic<float> &f, float d){
  float old = f.load(std::memory_order_acquire);
  float desired = old - d;
  while (!f.compare_exchange_weak(old, desired, std::memory_order_release, std::memory_order_acquire)) {
    desired = old - d;
  }
  //std::cout << "desired: " << desired << std::endl;
  return desired;
}

template <size_t in, size_t out, bool has_bias> 
class Dense {
public:
  Dense(
    float* input, float* output, float* grad_input, float* grad_output, 
    size_t batch_size=10, float learning_rate=1e-3
  ) : batch_size(batch_size), 
      learning_rate(learning_rate),
      input(input),
      output(output),
      grad_input(grad_input),
      grad_output(grad_output) {
    for (size_t i = 0; i < (has_bias + in) * out; ++i) {
      w[i].store(0);
    }
  }

  void forward() {
    for (size_t batch = 0; batch < batch_size; ++batch) {
      for (size_t i = 0; i < out; ++i) {
        if /* constexpr */ (has_bias) {
          output[batch * out + i] = w[(has_bias + in) * i].load(std::memory_order_acquire);
        } else {
          output[batch * out + i] = 0;
        }
        for (size_t j = 0; j < in; ++j) {
          output[batch * out + i] += w[(has_bias + in) * i + j + has_bias].load(std::memory_order_acquire) * input[batch * in + j];
        }
      }
    }
  }

  void backward(bool grad_output_required=true) {
    if (grad_output_required) {
      for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t i = 0; i < in; ++i) {
          grad_output[batch * in + i] = 0;
        }
        for (size_t j = 0; j < out; ++j) {
          for (size_t i = 0; i < in; ++i) {
            grad_output[batch * in + i] += grad_input[batch * out + j] * w[(has_bias + in) * j + i + has_bias].load(std::memory_order_acquire);
          }
        }
      }
    }
    for (size_t batch = 0; batch < batch_size; ++batch) {
      for (size_t j = 0; j < out; ++j) {
        if /* constexpr */ (has_bias) {
          atomic_subf(w[(has_bias + in) * j], learning_rate * grad_input[batch * out + j]);
          // w[(has_bias + in) * j].fetch_sub(learning_rate * grad_input[batch * out + j], std::memory_order_release);
        }
        for (size_t i = 0; i < in; ++i) {
          atomic_subf(w[(has_bias + in) * j + i + has_bias], learning_rate * grad_input[batch * out + j] * input[batch * in + i]);
          // w[(has_bias + in) * j + i + has_bias].fetch_sub(learning_rate * grad_input[batch * out + j] * input[batch * in + i]);
        }
      }
    }
  }
  
  void print_weights(void) {
    std::cout << "weigth A:" << std::endl;
    for (size_t j = 0; j < out; ++j) {
      for (size_t i = 0; i < in; ++i) {
        std::cout << w[j * in + i + has_bias].load() << " ";
      }
      std::cout << std::endl;
    }
    if /* constexpr */ (has_bias) {
      std::cout << "weigth b:" << std::endl;
      for (size_t j = 0; j < out; ++j) {
        std::cout << w[j * in].load() << " ";
        std::cout << std::endl;
      }
    }
  }
private:
  float* input; // [batch_size x in]
  float* output; // [batch_size x out]

  float* grad_input; // [batch_size x out]
  float* grad_output; // [batch_size x in]

  std::atomic<float> w[(has_bias + in) * out];

  float learning_rate;
  size_t batch_size;
};

} // kaspar
