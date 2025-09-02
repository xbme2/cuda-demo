#include "cuda_header.h"

float sum_cpu(float *i_data, int N) {
    float sum = 0.f;
    for (int i = 0; i < N; i++) {
        sum += i_data[i];
    }
    return sum;
}

void sum_v0(float *i_data, int N) {
    
}

int main() {}