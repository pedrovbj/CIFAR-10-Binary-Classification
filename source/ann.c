#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Dimension of input: 48x48 images +1 bias term
#define D_INPUT 2304+1
//Bias value
#define V_BIAS 1
//Learning rate
#define L_RATE (1e-6)
//Number of training samples
//#define N_TRAIN 500
#define N_TRAIN 8
//Number of test samples
//#define N_TEST 500
#define N_TEST 2
//Number of epochs for training
// #define EPOCHS 120
#define EPOCHS 5

// Training Input Data
double **X_train; //[N_TRAIN][D_INPUT]
// Testing Input Data
double **X_test; //[N_TEST][D_INPUT]
// Training Targets
double *T_train; //[N_TRAIN]
// Testing Targets
double *T_test; //[N_TEST]

double dot_product(double* a, double* b, size_t len);
int quantify(double y);
double cut_func(double t, double a, double b);
void trainNet(double *w); //w: weight vector

double dot_product(double* a, double* b, size_t len) {
    int i;
    double res = 0.0;

    for (i = 0; i < len; i++) {
        printf("%d ", i);fflush(stdout);
        printf("a: %f, ", a[i]);fflush(stdout);
        printf("b: %f\n", b[i]);fflush(stdout);
        res += a[i]*b[i];
    }

    return res;
}

int quantify(double y) {
    if(y > 0.5)
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
    //printf("(%ld, %ld)->(%ld, %ld)\n", nrow, ncol, ncol, nrow);fflush(stdout);
    for (i = 0; i < nrow; i++) {
        for (j = 0; j < ncol; j++) {
            //printf("[%ld][%ld] ", i, j);fflush(stdout);
            M_T[j][i] = M[i][j];
        }
    }

    return M_T;
}

void trainNet(double *w) {
	int i, j;
    double y, Error[N_TRAIN];
    double** X_T;

    printf("Computing transpose...\n");
    X_T = transpose(X_train, N_TRAIN, D_INPUT);
    printf("Transpose computed\n");

	for(i = 0; i < EPOCHS; i++) {
        printf("*************** EPOCH %d ***************\n", i+1); fflush(stdout);
		//Iterate through traning data set
		for(j = 0; j < N_TRAIN; j++) {
            printf("Dot? ");fflush(stdout);
			//y = cut_func(dot_product(X_train[j], w, D_INPUT), -1.0, 1.0);
            printf("(X_train[j]: %p) ", X_train[j]);fflush(stdout);
            y = dot_product(X_train[j], w, D_INPUT);
            printf("%f\n", y);fflush(stdout);
			Error[j] = T_train[j]-y;
		}
        printf("updating w...");fflush(stdout);
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

void readData(FILE* f, int len, double** X, double* T) {
    int i;
    char *buffer;
    const char s[2] = ",";
    char *token;
    double* x;

    buffer = (char*) malloc(10000*sizeof(char));

    for(i = 0; i < len; i++) {
        x = X[i];
        readLine(f, buffer);

        // quebra linha em tokens, cada token eh um inteiro
        token = strtok(buffer, s);
        // printf("Linha %d: ", i);
        // fflush(stdout);
        while(token != NULL) {
        //    printf("%s ", token);
        //    fflush(stdout);
           *x = ((double)atoi(token))/255.0; //grava cada pixel normalizado
           //printf("%f ", *x);
           x++;
           token = strtok(NULL, s);
        }
        //printf("\n");
        //Gravou a classificacao no lugar do bias e andou um pra frente
        //Voltamos uma posicao e atribuimos a classificacao ao vetor de targets
        x--;
        T[i] = *x;
        // atribuimos o valor do bias
        *x = V_BIAS/255.0;
    }
}

int main(void) {
    int i;
    double *w;
    FILE* f;

    f = fopen("../data/proc_db_teste.dat", "r");
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

    T_train = (double*) malloc(N_TRAIN*sizeof(double));
    T_test = (double*) malloc(N_TEST*sizeof(double));

    // SE INVERTER A ORDEM DA BUG
    readData(f, N_TEST, X_test, T_test);
    readData(f, N_TRAIN, X_train, T_train);

    printf("Training net...\n");

	trainNet(w);

    printf("Done\n");

    for(i = 0; i < D_INPUT; i++) {
        printf("%f ", w[i]);
    }
    printf("\n");


    free(w);
    exit(0);
}

// void print_matrix(double **M, int nrow, int ncol) {
//     int i, j;
//     for (i = 0; i < nrow; i++) {
//         for (j = 0; j < ncol; j++) {
//             printf("%f ", M[i][j]);
//         }
//         printf("\n");
//     }
// }
//
// int main(void) {
//     int i, j;
//     //TESTE TRANSPOSE
//     double **M;// = {{1.0,2.0,-1.0},{3.0,4.0,-2.0}};
//
//     M = (double**) malloc(2*sizeof(double*));
//     for(i = 0; i < 2; i++) {
//         M[i] = (double*) malloc(3*sizeof(double));
//     }
//
//     for(i = 0; i < 2; i++) {
//         for(j = 0; j < 3; j++) {
//             M[i][j] = i+j;
//         }
//     }
//     print_matrix(M, 2, 3);
//
//     double** M_T = transpose(M, 2, 3);
//     print_matrix(M_T, 3, 2);
//
//     return 0;
// }
