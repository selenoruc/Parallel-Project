#pragma once

void Multiply(double* A, double* B, double* C, size_t I, size_t J, size_t K)
{
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

void Add(double* A, double* B, double* C, size_t I, size_t J)
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

void Subtract(double* A, double* B, double* C, size_t I, size_t J)
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

void dotProduct(double* A, double* B, double* C, size_t I, size_t J)
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

void skalerMul(double A, double* B, size_t I, size_t J)
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

void SkalerDiv(double A, double* B, size_t I, size_t J)
{
	// Skaler Division : B[I][j] / A = C[I][j]

	for (size_t i = 0; i < I; i++)
	{
		for (size_t j = 0; j < J; j++)
		{
			B[i * J + j] /=  A;
		}
	}
}

void Softmax_derivative(double* O, double* D, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		D[i] = O[i] * (1 - O[i]);
	}
}