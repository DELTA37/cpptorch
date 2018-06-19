#pragma once

namespace kaspar {

class MSE {

public:
  MSE(
    float* y_true, float* y_pred, float* output, float* grad_true, float* grad_pred,
    size_t batch_size=10
  ) : batch_size(batch_size), 
      y_true(y_true),
      y_pred(y_pred),
      output(output),
      grad_true(grad_true),
      grad_pred(grad_pred) {}

  void forward(bool output_required=true) {
    if (!output_required) {
      return;
    }
    *output = 0;
    for (size_t batch = 0; batch < batch_size; ++batch) {
      *output += (y_true[batch] - y_pred[batch]) * (y_true[batch] - y_pred[batch]);
    }
  }
  
  void backward(bool grad_true_required=false, bool grad_pred_required=true) {
    if (grad_pred_required) {
      for (size_t batch = 0; batch < batch_size; ++batch) {
        grad_pred[batch] = 2 * (y_pred[batch] - y_true[batch]);
      }
    }
    if (grad_true_required) {
      for (size_t batch = 0; batch < batch_size; ++batch) {
        grad_true[batch] = 2 * (y_true[batch] - y_pred[batch]);
      }
    }
  }

private:
  float* y_true; // [batch_size x 1]
  float* y_pred; // [batch_size x 1]
  float* output; // [1]

  float* grad_true; // [batch_size x 1]
  float* grad_pred; // [batch_size x 1]

  size_t batch_size;
};

} // kaspar
