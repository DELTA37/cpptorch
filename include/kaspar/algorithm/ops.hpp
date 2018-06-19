#pragma once

namespace kaspar {

template<size_t in>
class PlusOp {
public:
  PlusOp(
    float* input1, float* input2, float* output, float* grad_input, float* grad_output1, float* grad_output2,
    size_t batch_size=10
  ) : batch_size(batch_size), 
      input1(input1),
      input2(input2),
      output(output),
      grad_input(grad_input),
      grad_output1(grad_output1),
      grad_output2(grad_output2) {}

  void forward() {
    size_t idx = 0;
    for (size_t batch = 0; batch < batch_size; ++batch) {
      for (size_t i = 0; i < in; ++i) {
        idx = batch * in + i;
        output[idx] = input1[idx] + input2[idx];
      }
    }
  }
  
  void backward(bool grad_output1_required=true, bool grad_output2_required=true) {
    if (grad_output1_required && (grad_output1 != grad_input)) {
      for (size_t i = 0; i < batch_size * in; ++i) {
        grad_output1[i] = grad_input[i];
      }
    }
    if (grad_output2_required && (grad_output2 != grad_input)) {
      for (size_t i = 0; i < batch_size * in; ++i) {
        grad_output2[i] = grad_input[i];
      }
    }
  }

private:
  float* input1; // [batch_size x in]
  float* input2; // [batch_size x in]
  float* output; // [batch_size x out]

  float* grad_input; // [batch_size x out]
  float* grad_output1; // [batch_size x in]
  float* grad_output2; // [batch_size x in]

  size_t batch_size;
};



} // kaspar
