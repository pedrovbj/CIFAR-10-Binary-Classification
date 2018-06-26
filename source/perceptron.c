#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Dimension of input: 48x48 images +1 bias term
#define D_INPUT (2304+1)
//Bias value
#define V_BIAS 1
//Learning rate
#define L_RATE (1e-6)
//Number of training samples
#define N_TRAIN 850
//Number of test samples
#define N_TEST 150
//Number of epochs for training
#define EPOCHS 120

// Training Input Data
 double **X_train; //[N_TRAIN][D_INPUT]
// Testing Input Data
 double **X_test; //[N_TEST][D_INPUT]
// Training Targets
 int *T_train; //[N_TRAIN]
// Testing Targets
 int *T_test; //[N_TEST]

double dot_product(double* a,  double* b, int len) {
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

void trainNet(double* w) {
	int i, j, k;
    double y;
    double *Error;
    double** X_T;

    Error = (double*) malloc(N_TRAIN*sizeof(double));

    // printf("> Computing transpose...\n");
    X_T = transpose(X_train, N_TRAIN, D_INPUT);
    // printf("> Transpose computed.\n");

	for(i = 0; i < EPOCHS; i++) {
        // printf("*************** EPOCH %d ***************\n", i+1); fflush(stdout);
		//Iterate through traning data set

        for(j = 0; j < N_TRAIN; j++) {
			y = cut_func(dot_product(X_train[j], w, D_INPUT), -1.0, 1.0);
			Error[j] = T_train[j]-y;
		}
        // printf("> Updating w...");fflush(stdout);
        //Update weight vector
        for(k = 0; k < D_INPUT; k++) {
            w[k] += L_RATE*dot_product(X_T[k], Error, N_TRAIN);
        }
        // printf("done.\n");fflush(stdout);
	}

    free(Error);
    for (i = 0; i < D_INPUT; i++) {
        free(X_T[i]);
    }
    free(X_T);
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

void readData(FILE* f, int len,  double** X, int* T) {
    int i, j;
    char *buffer;
    char s[2] = ",";
    char *token;

    buffer = (char*) malloc(10000*sizeof(char));

    for(i = 0; i < len; i++) {
        readLine(f, buffer);

        // breaks line in tokens
        token = strtok(buffer, s);
        for (j = 0; j < D_INPUT-1; j++) {
           //convert and normalize pixel value
           X[i][j] = ((double)atoi(token))/255.0;
           token = strtok(NULL, s); //le proximo token
        }
        // atribuimos o valor do bias
        X[i][D_INPUT-1] = V_BIAS/255.0;
        //atribui a classificacao ao vetor de targets
        T[i] = atoi(token);
    }

    free(buffer);
}

int main(void) {
    int i, j;
    double *w;
    FILE* f;
    int *Y_test;

    // Open file
    f = fopen("../data/proc_db.dat", "r");
    if (f == NULL) {
        printf("Error opening file\n");
        return(-1);
    }

    // Allocate matrices and vectors
    X_train = (double**) malloc(N_TRAIN*sizeof(double*));
    for(i = 0; i < N_TRAIN; i++) {
        X_train[i] = ( double*) malloc(D_INPUT*sizeof(double));
    }

    X_test = (double**) malloc(N_TEST*sizeof( double*));
    for(i = 0; i < N_TEST; i++) {
        X_test[i] = ( double*) malloc(D_INPUT*sizeof(double));
    }

    T_train = (int*) malloc(N_TRAIN*sizeof(int));
    T_test = (int*) malloc(N_TEST*sizeof(int));
    Y_test = (int*) malloc(N_TEST*sizeof(int));

    w = (double*) calloc(D_INPUT, sizeof(double));

    //Same order as in python prototype
    readData(f, N_TEST, X_test, T_test);
    readData(f, N_TRAIN, X_train, T_train);

    printf("[Test samples]\n");
    for(i = 0; i < 4; i++) {
        printf("[ ");
        for(j = 0; j < 10; j++) {
            printf("%f ", X_test[i][j]);
        }
        printf("...]");
        printf(" -> %d\n", T_test[i]);
    }
    printf("...\n");

    printf("[Training samples]\n");
    for(i = 0; i < 4; i++) {
        printf("[ ");
        for(j = 0; j < 10; j++) {
            printf("%f ", X_train[i][j]);
        }
        printf("...]");
        printf(" -> %d\n", T_train[i]);
    }
    printf("...\n");

    printf("[Training net...]\n");
	trainNet(w);
    printf("[Done training net.]\n");

    // Computes score
    float score = 0.0;
    for(i = 0; i < N_TEST; i++) {
        Y_test[i] = quantify(cut_func(dot_product(X_test[i], w, D_INPUT), -1.0, 1.0));
        if (Y_test[i] == T_test[i]) {
            score += 1.0/N_TEST;
        }
    }
    printf("\nScore: %0.2f\n", 100*score);

    // Frees memory
    for(i = 0; i < N_TRAIN; i++) {
        free(X_train[i]);
    }
    free(X_train);

    for(i = 0; i < N_TEST; i++) {
        free(X_test[i]);
    }
    free(X_test);

    free(T_train);
    free(T_test);
    free(Y_test);

    free(w);

    fclose(f);

    // Return SUCCESS
    return 0;
}
