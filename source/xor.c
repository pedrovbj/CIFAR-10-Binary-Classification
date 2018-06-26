#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Dimension of input: 48x48 images +1 bias term
#define D_INPUT (3+1)
//Bias value
#define V_BIAS 1
//Learning rate
#define L_RATE (1e-1)
//Number of training samples
#define N_TRAIN 4
//Number of test samples
#define N_TEST 4
//Number of epochs for training
#define EPOCHS 100

// Training Input Data
double **X_train; //[N_TRAIN][D_INPUT]
// Testing Input Data
double **X_test; //[N_TEST][D_INPUT]
// Training Targets
int *T_train; //[N_TRAIN]
// Testing Targets
int *T_test; //[N_TEST]

double dot_product(double* a, double* b, int len) {
    int i;
    double res = 0.0;

    for (i = 0; i < len; i++) {
        res += a[i]*b[i];
    }

    return res;
}

int quantify(double y) {
    if(y >= 0.5)
        return 1;
    else
        return 0;
}

double cut_func(double t, double a, double b) {
    if (t <= a) {
        return 0.0;
    } else if (t >= b) {
        return 1.0;
    } else {
        return (t-a)/(b-a);
    }
}

/*
Inputs
    M: Matrix to be transposed
    nrow: number of rows of M
    ncol: number of cols of M
Output
    M_T: transpose of M
*/
double** transpose(double** M, size_t nrow, size_t ncol) {
    size_t i, j;

    //Allocate matrix in memory
    double** M_T = (double**) malloc(ncol*sizeof(double*));

    for(i = 0; i < ncol; i++) {
        M_T[i] = (double*) malloc(nrow*sizeof(double));
    }

    //Transposition operation
    for (i = 0; i < nrow; i++) {
        for (j = 0; j < ncol; j++) {
            M_T[j][i] = M[i][j];
        }
    }

    return M_T;
}

void trainNet(double *w) {
	int i, j;
    double y, Error[N_TRAIN];
    double** X_T;

    printf("> Computing transpose...\n");
    X_T = transpose(X_train, N_TRAIN, D_INPUT);
    printf("> Transpose computed.\n");

	for(i = 0; i < EPOCHS; i++) {
        printf("*************** EPOCH %d ***************\n", i+1); fflush(stdout);
		//Iterate through traning data set
		for(j = 0; j < N_TRAIN; j++) {
			y = cut_func(dot_product(X_train[j], w, D_INPUT), -1.0, 1.0);
			Error[j] = T_train[j]-y;
		}
        printf("> Updating w...");fflush(stdout);
        //Update weight vector
        for(j = 0; j < D_INPUT; j++) {
            w[j] += L_RATE*dot_product(X_T[j], Error, N_TRAIN);
        }
        printf("done.\n");fflush(stdout);
	}
}

void readLine(FILE* f, char* buffer) {
    char* p = buffer;
    int c;
    while((c=fgetc(f))!= '\n') {
        *p = c;
        p++;
    }
    *p = '\0';
}

void readData(FILE* f, int len, double** X, int* T) {
    int i, j;
    char *buffer;
    const char s[2] = ",";
    char *token;
    //double* x;

    buffer = (char*) malloc(10000*sizeof(char));

    for(i = 0; i < len; i++) {
        readLine(f, buffer);

        // quebra linha em tokens, cada token eh um inteiro
        token = strtok(buffer, s);
        fflush(stdout);
        for (j = 0; j < D_INPUT-1; j++) {
            fflush(stdout);
           X[i][j] = ((double)atoi(token));///255.0; //grava cada pixel normalizado
           token = strtok(NULL, s); //le proximo token
        }
        // atribuimos o valor do bias
        X[i][D_INPUT-1] = V_BIAS;///255.0;
        //atribui a classificacao ao vetor de targets
        T[i] = atoi(token);
    }
}

int main(void) {
    int i, j;
    double *w;
    FILE* f;
    int ys[N_TEST];

    //f = fopen("../data/proc_db_teste.dat", "r");
    f = fopen("xor.dat", "r");
    if (f == NULL) {
        printf("Error opening file\n");
        return(-1);
    }

    w = (double*) calloc(N_TRAIN, sizeof(double));

    X_train = (double**) malloc(N_TRAIN*sizeof(double*));
    for(i = 0; i < N_TRAIN; i++) {
        X_train[i] = (double*) malloc(D_INPUT*sizeof(double));
    }

    X_test = (double**) malloc(N_TEST*sizeof(double*));
    for(i = 0; i < N_TEST; i++) {
        X_test[i] = (double*) malloc(D_INPUT*sizeof(double));
    }

    T_train = (int*) malloc(N_TRAIN*sizeof(int));
    T_test = (int*) malloc(N_TEST*sizeof(int));

    readData(f, N_TRAIN, X_train, T_train);
    readData(f, N_TEST, X_test, T_test);

    printf("[Training samples]\n");
    for(i = 0; i < N_TRAIN; i++) {
        printf("[ ");
        for(j = 0; j < D_INPUT; j++) {
            printf("%f ", X_train[i][j]);
        }
        printf("]");
        printf(" -> %d\n", T_train[i]);
    }

    printf("[Test samples]\n");
    for(i = 0; i < N_TRAIN; i++) {
        printf("[ ");
        for(j = 0; j < D_INPUT; j++) {
            printf("%f ", X_test[i][j]);
        }
        printf("]");
        printf(" -> %d\n", T_test[i]);
    }

    printf("[Training net...]\n");

	trainNet(w);

    printf("[Done training net.]\n");

    for(i = 0; i < D_INPUT; i++) {
        printf("%f ", w[i]);
    }
    printf("\n");

    for(i = 0; i < N_TEST; i++) {
        ys[i] = quantify(cut_func(dot_product(X_test[i], w, D_INPUT), -1.0, 1.0));
    }
    printf("predictions: ");
    for(i = 0; i < N_TEST; i++) {
        printf("%d:%d ", ys[i], T_test[i]);
    }
    printf("\n");

    free(w);
    exit(0);
}
