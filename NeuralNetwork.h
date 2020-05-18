#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

using namespace std;


class NeuralNetwork
{
public:

	NeuralNetwork();
	NeuralNetwork(size_t HidLayerNeurons, size_t OutLayerNeurons);
	~NeuralNetwork();
	
	void NormalizeInput();

	void ReLU();

	void Softmax();

	void Diff_ReLU();
	void Diff_Softmax();

	void ReadTrainingSamples(string SamplesFile);
	void ReadLabels(string labels);

private:
	int	   BIAS;
	size_t InputVectorSize;			// input vector size
	size_t TrainingSampleCount;		// Total Number of Training Samples
	size_t NeuronSize_HiddenLayer;	// Neuron Size Hidden Layer
	size_t NeuronSize_OutLayer;		// Neuron Size Output Layer 
	double* Input;					// All Training Inputs [inputvectorsize x TrainingSampleCount]
	double* V;						// Hidden Layer Weights
	double* Net1;					// Hidden Layer Nets    
	double* y;						// Hidden Layer Outputs (input to next layer)
	double* delta_y;				// Derivative ReLU
	double* W;						// Output Layer Weights
	double* Net2;					// Output Layer Nets	
	double* o;						// Final Network Outputs
	double* delta_o;				// Derivative Softmax
	double* d;						// Desired Outputs
};

