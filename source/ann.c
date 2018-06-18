#include <stdio.h>
#include <stdlib.h>

//Dimension of input: 48x48 images +1 bias term
#define D_INPUT 2304+1
//Bias value
#define V_BIAS 1
//Learning rate
#define L_RATE 1
//Number of training samples
#define N_TRAIN 100
//Number of epochs for training
#define EPOCHS 100

// Training Input Data
const float **X_train; //[N_TRAIN][D_INPUT]
// Training Targets
const float *T_train; //[N_TRAIN]

float dot_product(float* a, float* b, size_t len);
int step_func(float s);
void trainNet(float *w); //w: weight vector

float dot_product(float* a, float* b, size_t len) {
    int i;
    float res = 0.0;

    for (i = 0; i < len; i++) {
        res += a[i]*b[i];
    }

    return res;
}

int step_func(float s) {
    if(s > 0)
        return 1;
    else
        return 0;
}

/*
Inputs
    M: Matrix to be transposed
    nrow: number of rows of M
    ncol: number of cols of M
Output
    M_T: transpose of M
*/
float** transpose(float** M, size_t nrow, size_t ncol) {
    size_t i, j;

    //Allocate matrix in memory
    float** M_T = (float**) malloc(ncol*sizeof(float*));
    for(i = 0; i < nrow; i++) {
        M_T[i] = (float*) malloc(nrow*sizeof(float));
    }

    //Transposition operation
    for (i = 0; i < nrow; i++) {
        for (j = 0; j < ncol; j++) {
            M_T[j][i] = M[i][j];
        }
    }

    return M_T;
}

void trainNet(float *w) {
	int i, j;
    float y, Error[N_TRAIN];
    float** X_T = transpose(X_train, N_TRAIN, D_INPUT);

	for(i = 0; i < EPOCHS; i++) {
		//Iterate through traning data set
		for(j = 0; j < N_TRAIN; j++) {
			y = dot_product(X_train[j], w, D_INPUT));
			Error[j] = T_train[j]-y;
		}
        //Update weight vector
        for(j = 0; j < D_INPUT; j++) {
            w[j] += L_RATE*dot_product(X_T[j], Error, N_TRAIN);
        }
	}
	printf("RESULTADO TREINAMENTO");
	printf("w[bias] = %f\n", sinapses[0]);
	printf("w[e1] = %f\n", sinapses[1]);
	printf("w[e2] = %f\n", sinapses[2]);
}

void readData(FILE* f) {

}

int main(void) {
    float *w;

    w = (float*) calloc(N_TRAIN, sizeof(float));

	trainNet(w);

    free(w);
    return 0;
}
