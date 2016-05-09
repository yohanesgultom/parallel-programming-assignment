// author: yohanes.gultom@gmail.com
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// create random matrix row-major-format
float* create_flat_matrix_rand(int row, int col, int max)
{
    float* m = (float*)malloc(row*col*sizeof(float));
    int i, j = 0;
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            float val = (max > 0) ? (float)(rand() % max) : 0.0f;
            m[col * i + j] = val;
        }
    }
    return m;
}

float* create_flat_matrix(int row, int col, float val)
{
    float* m = (float*)malloc(row*col*sizeof(float));
    int i, j = 0;
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            m[col * i + j] = val;
        }
    }
    return m;
}


// print matrix row-major-format
void print_flat_matrix(float *m, int row, int col)
{
    int i, j = 0;
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            printf("%.2f ", m[col * i + j]);
        }
        printf("\n");
    }
}

void mmul(float *first, int m, int p, float *second, int q, float *multiply)
{
    int c, d, k = 0;
    float sum = .0f;
    for (c = 0; c < m; c++) {
        for (d = 0; d < q; d++) {
            for (k = 0; k < p; k++) {
                sum = sum + first[c*m+k] * second[k*q+d];
            }
            multiply[c*q+d] = sum;
            sum = 0;
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("insufficient args. for A x B = C, required args: [row num A] [col num A/row num B] [col num B] [reps] [compare]\n");
        return EXIT_FAILURE;
    }

    int m, n, p, q, reps, compare = 0;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = n;
    q = atoi(argv[3]);
    reps = (argc >= 5) ? atoi(argv[4]) : 1;
    compare = (argc >= 6) ? atoi(argv[5]) : 0;

    float *first, *second, *multiply;
    int i = 0;
    double total_time = 0.0f;
    for (i = 0; i < reps; i++) {
        double exec_time = ((double) clock()) * -1;
        first = create_flat_matrix(m, n, 1);
        second = create_flat_matrix(p, q, 2);
        multiply = create_flat_matrix(m, q, 0);
        mmul(first, m, n, second, q, multiply);

        if (compare == 1) {
            printf("first:\n");
            print_flat_matrix(first, m, n);
            printf("second:\n");
            print_flat_matrix(second, p, q);
            printf("multiply:\n");
            print_flat_matrix(multiply, m, q);
        }

        free(multiply); free(second); free(first);
        total_time = total_time + ((exec_time + ((double)clock())) / CLOCKS_PER_SEC);
    }
    printf("%d\t%d\t%d\t%d\t%.6f\n", m, n, q, reps, (total_time / reps));
    return EXIT_SUCCESS;
}
