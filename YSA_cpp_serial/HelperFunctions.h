#pragma once

#include <cmath>
#include <iostream>


using namespace std;

static void Multiply(double* A, double* B, double* C, size_t I, size_t J, size_t K)
{                                                     //     K                   I
	// Multiplication : A[I][K] * B[K][J] = C[I][J]

	double productSum;


	for (size_t i = 0; i < I; i++)
	{
		for (size_t j = 0; j < J; j++)
		{
			productSum = 0;
			for (size_t k = 0; k < K; k++)
			{
				productSum += A[i * K + k] * B[k * J + j];
			}
			C[i * J + j] = productSum;
		}
	}

}


static void Transpose(double* A, double* A_trans, size_t I, size_t J)
{
	// A[I][J]  ---->   A_trans[J][I]

	for (size_t i = 0; i < I; i++)
	{
		for (size_t j = 0; j < J; j++)
		{
			A_trans[j * I + i] = A[i * J + j];
		}
	}

}


static void Add(double* A, double* B, double* C, size_t I, size_t J)
{
	// Addition : A[I][J] + B[I][j] = C[I][j]

	for (size_t i = 0; i < I; i++)
	{
		for (size_t j = 0; j < J; j++)
		{
			C[i * J + j] = A[i * J + j] + B[i * J + j];
		}
	}
}

// SUBTRACT CUT
static void Subtract(double* A, double* B, double* C, size_t I, size_t J)
{
	// Subtracttion : A[I][J] - B[I][j] = C[I][j]

	for (size_t i = 0; i < I; i++)
	{
		for (size_t j = 0; j < J; j++)
		{
			C[i * J + j] = A[i * J + j] - B[i * J + j];
		}
	}
}

static void dotProduct(double* A, double* B, double* C, size_t I, size_t J)
{
	// Dot Product : A[I][J] .* B[I][j] = C[I][j]

	for (size_t i = 0; i < I; i++)
	{
		for (size_t j = 0; j < J; j++)
		{
			C[i * J + j] = A[i * J + j] * B[i * J + j];
		}
	}
}

static void skalerMul(double A, double* B, size_t I, size_t J)
{
	// Skaler Multiplicaiton : A * B[I][j] = C[I][j]

	for (size_t i = 0; i < I; i++)
	{
		for (size_t j = 0; j < J; j++)
		{
			B[i * J + j] *= A;
		}
	}
}

static void SkalerDiv(double A, double* B, size_t I, size_t J)
{
	// Skaler Division : B[I][j] / A = C[I][j]

	for (size_t i = 0; i < I; i++)
	{
		for (size_t j = 0; j < J; j++)
		{
			B[i * (J + 1) + j] /= A;
		}
	}
}

static void Softmax_derivative(double* O, double* D, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		D[i] = O[i] * (1.0 - O[i]);
	}
}


static void ReLU_derivative(double* X, double* D, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		D[i] = (X[i] > 0) ? 1.0 : 0.0;
	}
}

static void stdError(double* Error, double& Erms, size_t OutputDim)
{
	double sqSum = 0;

	for (size_t i = 0; i < OutputDim; i++)
	{
		sqSum += (Error[i] * Error[i]);
	}

	Erms = sqrt(sqSum) / (double)OutputDim;
}

static void TotalError(double* Error, double& Erms, size_t OutputDim)
{
	Erms = 0;

	for (size_t i = 0; i < OutputDim; i++)
	{
		Erms += (Error[i] * Error[i]);//   abs(Error[i]);
	}

	Erms /= OutputDim;
}

static void Clean(double* A, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		A[i] = 0.0;
	}
}

static void Round(double* A, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		A[i] = round(A[i]);
	}
}


static void MaxNormalization(double* A, size_t N)
{
	double max = A[0];
	for (size_t i = 1; i < N; i++)
	{
		//max = (A[i - 1] < A[i]) ? i : i - 1;
		max = (A[i] > max) ? A[i] : max;
	}

	for (size_t i = 0; i < N; i++)
	{
		A[i] /= max;
	}
}