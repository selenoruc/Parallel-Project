#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <time.h>
#include "HelperFunctions.h"

using namespace std;


class NeuralNetwork
{
public:

	NeuralNetwork();
	~NeuralNetwork();

	NeuralNetwork(size_t HidLayerNeurons, size_t OutLayerNeurons);
	void ReadTrainingSamples(string SamplesFile);
	void ReadLabels(string labels);
	void RandomizeWeights();
	void NormalizeInput();
	void calcNet_1();
	void ReLU();
	void calcNet_2();
	void Softmax();
	void updateW();
	void updateV();
	void Diff_Softmax();
	void Diff_ReLU();
	
private:
	int	   BIAS;
	size_t InputVectorSize;			// input vector size
	size_t TrainingSampleCount;		// Total Number of Training Samples
	size_t NeuronSize_HiddenLayer;	// Neuron Size Hidden Layer
	size_t NeuronSize_OutLayer;		// Neuron Size Output Layer
	size_t idxSample;				// Sample idx
	double* Input;					// All Training Inputs [inputvectorsize x TrainingSampleCount]
	double* V;						// Hidden Layer Weights
	double* dV;
	double* Net1;					// Hidden Layer Nets    
	double* y;						// Hidden Layer Outputs (input to next layer)
	double* delta_y;				// Derivative ReLU
	double* W;						// Output Layer Weights
	double* dW;
	double* Net2;					// Output Layer Nets	
	double* o;						// Final Network Outputs
	double* delta_o;				// Derivative Softmax
	double* d;						// Desired Outputs
	double* Error;					// d-o Vector
};

