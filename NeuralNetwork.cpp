#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork()
{
	this->InputVectorSize			= 0;
	this->NeuronSize_HiddenLayer	= 0;
	this->NeuronSize_OutLayer		= 0;
	this->TrainingSampleCount		= 0;
	this->BIAS						= 1;
	this->idxSample					= 0;

	/*this->Input						= NULL;
	this->V							= NULL;
	this->Net1						= NULL;
	this->y							= NULL;
	this->W							= NULL;
	this->Net2						= NULL;
	this->o							= NULL;
	this->d							= NULL;
	this->Error = NULL;
	this->delta_y = NULL;
	this->delta_o*/
}

NeuralNetwork::NeuralNetwork(size_t HidLayerNeurons, size_t OutLayerNeurons)
{
	this->InputVectorSize			= 0;
	this->TrainingSampleCount		= 0;
	this->NeuronSize_HiddenLayer	= HidLayerNeurons;
	this->NeuronSize_OutLayer		= OutLayerNeurons;
	this->BIAS						= 1;
	this->idxSample					= 0;

	/*
	this->Input						= new double;
	this->V							= new double;
	this->Net1						= new double;
	this->y							= new double;
	this->W							= new double;
	this->Net2						= new double;
	this->o							= new double;
	this->d							= new double;
	this->Error = NULL;
	this->delta_y = NULL;
	this->delta_o*/
}

void NeuralNetwork::ReadTrainingSamples(string SamplesFile)
{
	if ((NeuronSize_HiddenLayer + NeuronSize_OutLayer) == 0)
	{
		cout << "Construct Network Layers First!" << endl;
		return;
	}

	ifstream File;
	File.open(SamplesFile , ios::in);
	if (File.is_open())
	{
		File >> this->InputVectorSize;
		File >> this->TrainingSampleCount;

		this->Input			= new double[(InputVectorSize + 1) * TrainingSampleCount];
		this->V				= new double[(InputVectorSize + 1) * NeuronSize_HiddenLayer];
		this->dV			= new double[(InputVectorSize + 1) * NeuronSize_HiddenLayer];
		this->Net1			= new double[NeuronSize_HiddenLayer];
		this->y				= new double[NeuronSize_HiddenLayer + 1];
		this->delta_y		= new double[NeuronSize_HiddenLayer];
		this->W				= new double[(NeuronSize_HiddenLayer + 1) * NeuronSize_OutLayer];
		this->W				= new double[(NeuronSize_HiddenLayer + 1) * NeuronSize_OutLayer];
		this->Net2			= new double[NeuronSize_OutLayer];
		this->o				= new double[NeuronSize_OutLayer];
		this->delta_o		= new double[NeuronSize_OutLayer];
		this->d				= new double[TrainingSampleCount * NeuronSize_OutLayer];
		this->Error			= new double[NeuronSize_OutLayer];
	}

	/// TO DO:
	// Parse rest of the file and fill in    < Input >   array...


}

void NeuralNetwork::ReadLabels(string Labels)
{
	ifstream File;
	File.open(Labels, ios::in);

	/// TO DO:
	// Parse Label File :
	// Fill in     < d >    ( Desired Outputs : Labels )

}


void NeuralNetwork::ReLU()
{
	for (size_t i = 0; i < NeuronSize_HiddenLayer; i++)
	{
		y[i] = Net1[i] > 0 ? Net1[i] : 0;
	}
	y[NeuronSize_HiddenLayer] = this->BIAS;
}

void NeuralNetwork::Softmax()
{
	double expSUM = 0;

	for (size_t i = 0; i < NeuronSize_OutLayer; i++)
	{
		expSUM += exp(Net2[i]);
	}

	for (size_t i = 0; i < NeuronSize_OutLayer; i++)
	{
		o[i] = exp(Net2[i]) / expSUM;
	}
}

void NeuralNetwork::NormalizeInput()
{
	SkalerDiv(255.0, Input, this->TrainingSampleCount, this->InputVectorSize);
}

void NeuralNetwork::calcNet_1()
{
	Multiply(this->V, this->Input + idxSample, this->Net1, this->NeuronSize_HiddenLayer, 1, this->InputVectorSize + 1);
}

void NeuralNetwork::calcNet_2()
{
	Multiply(this->W, this->y, this->Net2, this->NeuronSize_OutLayer, 1, NeuronSize_HiddenLayer + 1);
}

void NeuralNetwork::updateW()
{
	Subtract(d, o, Error, this->NeuronSize_OutLayer, 1); // Error = d - o

}

void NeuralNetwork::Diff_ReLU()
{
	
	for (size_t i = 0; i < NeuronSize_HiddenLayer; i++)
	{
		delta_y[i] = Net1[i] > 0 ? 1 : 0;
	}
	
}

void NeuralNetwork::Diff_Softmax()
{
	for (size_t i = 0; i < NeuronSize_OutLayer; i++)
	{
		delta_o[i] = (o[i] * (1 - o[i])) * Error[i] ;
	}
}


NeuralNetwork::~NeuralNetwork()
{
	delete[] Input;
	delete[] V;
	delete[] dV;
	delete[] Net1;
	delete[] y;
	delete[] delta_y;
	delete[] W;
	delete[] dW;
	delete[] Net2;
	delete[] o;
	delete[] delta_o;
	delete[] d;
}