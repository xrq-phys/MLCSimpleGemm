#include <stdbool.h>

extern void mlcsgemm_simple_(bool transA,
                             bool transB,
                             int m,
                             int n,
                             int k,
                             float alpha,
                             float *addrA,
                             float *addrB,
                             float *addrC);

void mlcsgemm_simple(bool transA,
                     bool transB,
                     int m,
                     int n,
                     int k,
                     float alpha,
                     float *addrA,
                     float *addrB,
                     float *addrC) {
    mlcsgemm_simple_(transA,
                     transB,
                     m, n, k,
                     alpha,
                     addrA,
                     addrB,
                     addrC);
}

