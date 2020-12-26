#include <stdio.h>
#include <stdbool.h>

// Rawmaj
#define A(i, j) A[ (i)*ldA + (j) ]
#define B(i, j) B[ (i)*ldB + (j) ]
#define C(i, j) C[ (i)*ldC + (j) ]

#define M 6
#define N 6
#define K 6
const int ldA = K;
const int ldB = N;
const int ldC = N;

void mlcsgemm_simple(bool transA,
                     bool transB,
                     int m,
                     int n,
                     int k,
                     float alpha,
                     float *addrA,
                     float *addrB,
                     float *addrC);

int main(const int argc, const char *argv[])
{
    float A[M * K];
    float B[K * N];
    float C[M * N];
    // FILE *inA = fopen("INA.dat", "w");
    // FILE *inB = fopen("INB.dat", "w");
    // FILE *inC = fopen("INC.dat", "w");
    FILE *ouC = fopen("OUC.dat", "w");

    for (int j = 0; j < K; ++j)
        for (int i = 0; i < M; ++i)
            A(i, j) = 0.1 * j + 0.4 * i;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < K; ++i)
            B(i, j) = 0.2 * j + 0.3 * i;
    for (int i = 0; i < M * N; ++i)
        C[i] = 1.0;

    mlcsgemm_simple(false, false, M, N, K, 1.0, A, B, C);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j)
            fprintf(ouC, "%lf ", C(i, j));
        fprintf(ouC, "\n");
    }
    fclose(ouC);

    return 0;
}

