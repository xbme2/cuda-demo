#include <algorithm>

template <typename T> void softmax_cpu(T *i_data, T *o_data, int N, int C) {
    for (int i = 0; i < N; i++) {
        T max_val = -INFINITY;
        for (int j = 0; j < C; j++) {
            max_val = std::max(max_val, i_data[i * C + j]);
        }
        T sum = 0;
        for (int j = 0; j < C; j++) {
            o_data[i * C + j] = expf(i_data[i * C + j] - max_val);
            sum += o_data[i * C + j];
        }
        T norm = 1 / sum;
        for (int j = 0; j < C; j++) {
            o_data[i * C + j] *= norm;
        }
    }
}
