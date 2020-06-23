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
	void calcError();
	void calcStdError();
	void calcTotalError();
	void calcDelta_o();
	void updateW();
	void calcDelta_y();
	void updateV();
	void Train();
	bool isTrained();
	bool isTrainCompleted();
	void printOut();
	void ShuffleIdx();
	void Test();


private:
	int	   BIAS;
	size_t InputVectorSize;			// input vector size
	size_t TrainingSampleCount;		// Total Number of Training Samples
	size_t NeuronSize_HiddenLayer;	// Neuron Size Hidden Layer
	size_t NeuronSize_OutLayer;		// Neuron Size Output Layer
	size_t idxSample;				// Sample idx
	size_t sizeTrain;               // Dedicated Training Sample Count %67
	size_t sizeTest;				// Dedicated Test Sample Count %33
	size_t SuccessCount;			// Successful Tests

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
	double  Erms;					// RMS Error (for each CYCLE)



	size_t* idxShuffled;			// Shuffled indice
	size_t* Labels;					// Labels
	size_t* Predictions;			// Outputs
	//double* outs;
};

