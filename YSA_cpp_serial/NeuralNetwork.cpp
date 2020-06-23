#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork()
{
	this->InputVectorSize = 0;
	this->NeuronSize_HiddenLayer = 0;
	this->NeuronSize_OutLayer = 0;
	this->TrainingSampleCount = 0;
	this->BIAS = 1;
	this->idxSample = 0;
	this->Erms = 0.0;

}

NeuralNetwork::NeuralNetwork(size_t HidLayerNeurons, size_t OutLayerNeurons)
{
	this->InputVectorSize = 0;
	this->TrainingSampleCount = 0;
	this->NeuronSize_HiddenLayer = HidLayerNeurons;
	this->NeuronSize_OutLayer = OutLayerNeurons;
	this->BIAS = -1.0;
	this->idxSample = 0;
	this->Erms = 0.0;

}

void NeuralNetwork::ReadTrainingSamples(string SamplesFile)
{
	if ((NeuronSize_HiddenLayer + NeuronSize_OutLayer) == 0)
	{
		cout << "Construct Network Layers First!" << endl;
		return;
	}

	ifstream File;
	File.open(SamplesFile, ios::in);
	if (File.is_open())
	{
		File >> this->InputVectorSize;
		File >> this->TrainingSampleCount;

		this->sizeTrain = (5.0/6.0) * this->TrainingSampleCount;
		this->sizeTest = this->TrainingSampleCount - this->sizeTrain;
		this->Labels = new size_t[this->TrainingSampleCount];
		this->Predictions = new size_t[this->TrainingSampleCount];

		this->Input = new double[(InputVectorSize + 1) * TrainingSampleCount];
		this->V = new double[(InputVectorSize + 1) * NeuronSize_HiddenLayer];
		this->dV = new double[(InputVectorSize + 1) * NeuronSize_HiddenLayer];
		this->Net1 = new double[NeuronSize_HiddenLayer];
		this->y = new double[NeuronSize_HiddenLayer + 1];
		this->delta_y = new double[NeuronSize_HiddenLayer];
		this->W = new double[(NeuronSize_HiddenLayer + 1) * NeuronSize_OutLayer];
		this->dW = new double[(NeuronSize_HiddenLayer + 1) * NeuronSize_OutLayer];
		this->Net2 = new double[NeuronSize_OutLayer];
		this->o = new double[NeuronSize_OutLayer];
		this->delta_o = new double[NeuronSize_OutLayer];
		this->d = new double[TrainingSampleCount * NeuronSize_OutLayer];
		this->Error = new double[NeuronSize_OutLayer];


		/// TO DO:
		// Parse rest of the file and fill in    < Input >   array...

		for (size_t i = 0; i < this->TrainingSampleCount; i++)
		{
			for (size_t j = 0; j < this->InputVectorSize; j++)
			{
				File >> this->Input[i * (InputVectorSize + 1) + j];
			}
			this->Input[i * (InputVectorSize + 1) + InputVectorSize] = this->BIAS;
		}


		File.close();
	}

}

void NeuralNetwork::ReadLabels(std::string Labels)
{
	ifstream File;
	File.open(Labels, ios::in);
	size_t index;

	if (File.is_open())
	{
		for (size_t i = 0; i < TrainingSampleCount * NeuronSize_OutLayer; i++) {
			d[i] = 0;
		}

		for (size_t i = 0; i < TrainingSampleCount; i++) {
			File >> index;
			this->Labels[i] = index;
			this->d[NeuronSize_OutLayer * i + index] = 1;
		}
	}

	File.close();
}

void NeuralNetwork::RandomizeWeights()
{
	size_t AugmentedInputSize = this->InputVectorSize + 1;
	size_t Augmented_ySize = this->NeuronSize_HiddenLayer + 1;

	//V --> 128x785
	srand(time(0));
	for (size_t i = 0; i < this->NeuronSize_HiddenLayer; ++i)
	{
		for (size_t j = 0; j < AugmentedInputSize; ++j)
		{
			this->V[i * AugmentedInputSize + j] = 2 * ((double)rand() / RAND_MAX) - 1;
			//cout << V[i * AugmentedInputSize + j] << " ";
		}
		//cout << endl;
	}

	//W --> 10 x 129
	srand(time(0));
	for (size_t i = 0; i < this->NeuronSize_OutLayer; ++i)
	{
		for (size_t j = 0; j < Augmented_ySize; ++j)
		{
			this->W[i * Augmented_ySize + j] = 2 * ((double)rand() / RAND_MAX) - 1;
			//cout << W[i * Augmented_ySize + j] << " ";
		}
		//cout << endl;
	}

}

void NeuralNetwork::ReLU()
{
	for (size_t i = 0; i < NeuronSize_HiddenLayer; i++)
	{
		y[i] = Net1[i] > 0 ? Net1[i] : 0;
	}
	MaxNormalization(y, NeuronSize_HiddenLayer);
	y[NeuronSize_HiddenLayer] = this->BIAS;

	/*for (size_t i = 0; i < NeuronSize_HiddenLayer+1; i++)
	{
		cout << y[i] << endl;
	}*/

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

	size_t max = 0;
	for (size_t i = 1; i < NeuronSize_OutLayer; i++)
	{
		max = (o[i] > o[max]) ? i : max;
	}
	this->Predictions[this->idxSample] = max;
}

void NeuralNetwork::NormalizeInput()
{
	SkalerDiv(255.0, Input, this->TrainingSampleCount, this->InputVectorSize);
}

void NeuralNetwork::calcNet_1()
{
	Multiply(this->V, this->Input + idxSample * (InputVectorSize + 1), this->Net1,
		this->NeuronSize_HiddenLayer,
		1,
		this->InputVectorSize + 1);
}

void NeuralNetwork::calcNet_2()
{
	Multiply(this->W, this->y, this->Net2,
		this->NeuronSize_OutLayer,
		1,
		NeuronSize_HiddenLayer + 1);
}

void NeuralNetwork::calcError()
{
	// Error = d - o
	Subtract(this->d + idxSample * NeuronSize_OutLayer, this->o, this->Error, this->NeuronSize_OutLayer, 1);
}

void NeuralNetwork::calcDelta_o()
{
	double* Soft_Diff = new double[this->NeuronSize_OutLayer];
	// f'(net2)
	Softmax_derivative(this->o, Soft_Diff, this->NeuronSize_OutLayer);
	// Error = d - o
	//// delta_o = Error .* f'(net)
	dotProduct(this->Error, Soft_Diff, this->delta_o, this->NeuronSize_OutLayer, 1);

	delete[] Soft_Diff;
}

void NeuralNetwork::updateW()
{
	double c = 0.5;

	// dw = c * delta_o * y        [I][J]
	Multiply(this->delta_o/*Error*/, this->y, this->dW,
		this->NeuronSize_OutLayer,          // I = 10
		this->NeuronSize_HiddenLayer + 1,   // J = 128 + 1 = 129
		1);                                 // K = 1

	skalerMul(c, dW, this->NeuronSize_OutLayer, this->NeuronSize_HiddenLayer + 1);

	// W += dW
	Add(this->W, this->dW, this->W, this->NeuronSize_OutLayer, this->NeuronSize_HiddenLayer + 1);
}

void NeuralNetwork::calcDelta_y()
{
	double* ReLU_Diff = new double[this->NeuronSize_HiddenLayer];
	ReLU_derivative(this->Net1, ReLU_Diff, this->NeuronSize_HiddenLayer);

	// W --> W'      10x128 ---> 128x10
	double* W_t = new double[this->NeuronSize_OutLayer * this->NeuronSize_HiddenLayer];
	Transpose(this->W, W_t, this->NeuronSize_OutLayer, this->NeuronSize_HiddenLayer);
	// M = W' * delta_o 
	double* M = new double[this->NeuronSize_HiddenLayer];
	Multiply(W_t, this->delta_o, M, this->NeuronSize_HiddenLayer, 1, this->NeuronSize_OutLayer);

	// delta_y = M .* f'(net1)
	dotProduct(M, ReLU_Diff, this->delta_y, this->NeuronSize_HiddenLayer, 1);

	delete[] ReLU_Diff;
}

void NeuralNetwork::updateV()
{
	double c = 0.1;
	// dV = c * delta_y * Input
	Multiply(this->delta_y, this->Input + idxSample * (InputVectorSize + 1), this->dV,
		this->NeuronSize_HiddenLayer,
		this->InputVectorSize + 1,
		1);

	skalerMul(c, this->dV, this->NeuronSize_HiddenLayer, this->InputVectorSize + 1);

	// V += dV
	Add(this->V, this->dV, this->V, this->NeuronSize_HiddenLayer, this->InputVectorSize + 1);
}

void NeuralNetwork::calcStdError()
{
	// Error ---> Erms
	stdError(this->Error, this->Erms, this->NeuronSize_OutLayer);
}

void NeuralNetwork::calcTotalError()
{
	TotalError(this->Error, this->Erms, this->NeuronSize_OutLayer);
}

void NeuralNetwork::Train()
{
	size_t counter;

	for (this->idxSample = 0; this->idxSample < this->sizeTrain; this->idxSample++)
	{
		counter = 0;
		while (true)
		{
			this->calcNet_1();
			this->ReLU();
			this->calcNet_2();
			this->Softmax();
			this->calcError();
			this->calcStdError();
			//this->calcTotalError();
			
			if (isTrained())
			{
				//cout << " Target: " << this->Labels[idxSample]
				//	 << " Out: " << this->Predictions[idxSample] << endl;
				//cout << "idxSample: " << idxSample << " Erms: " << this->Erms << " Cycle: " << ++counter << endl;
				/*for(size_t i = 0; i < 10; i++)
				{
					cout << "idxSample: " << idxSample << " Target: " << this->d[idxSample * 10 + i]
						<< " Out: " << this->o[i] << " Erms: " << this->Erms << endl;
				}*/
				break;
			}
			
			this->calcDelta_o();
			this->calcDelta_y();
			this->updateW();
			this->updateV();
			++counter;
			//cout << "idxSample: " << idxSample << " Erms: " << this->Erms << endl;
			/*for (size_t i = 0; i < 10; i++)
			{
				cout << "idxSample: " << idxSample << " Target: " << this->d[idxSample*10 + i]
					 << " Out: " << this->o[i] << " Erms: " << this->Erms << endl;
			}*/

		}
	}
}

void NeuralNetwork::Test()
{
	int count_20 = 0;
	this->SuccessCount = 0;
	for (this->idxSample = sizeTrain; this->idxSample < this->TrainingSampleCount; this->idxSample++)
	{
		this->calcNet_1();
		this->ReLU();
		this->calcNet_2();
		this->Softmax();
		if (this->Labels[idxSample] == this->Predictions[idxSample])
			this->SuccessCount++;
		if (++count_20 <= 20)
		{
			cout << "Data ID: " << this->idxSample 
				 << "\t Target: " << this->Labels[idxSample] 
				 << "\t Prediction: " << this->Predictions[idxSample] << endl;
		}
	}

	cout << endl << "-------------- REPORT ------------------" << endl;

	cout << "Learning Constant - Layer 1 : " << 0.1 << endl
		<< "Learning Constant - Layer 2 : " << 0.5 << endl
		<< "TotalSet : " << this->TrainingSampleCount << endl
		<< "TrainSet : " << sizeTrain << endl
		<< "TestSet  : " << sizeTest << endl
		<< "Success  : " << SuccessCount << endl
		<< "Accuracy : " << (float)SuccessCount / (float)sizeTest << endl;

}

void NeuralNetwork::printOut()
{
	for (size_t i = 0; i < this->NeuronSize_OutLayer; i++)
	{
		printf("%f\t%f\n", this->d[i], this->o[i]);
	}
}

bool NeuralNetwork::isTrained()
{
	return Erms < 0.01;//0.00001;
}

bool NeuralNetwork::isTrainCompleted()
{
	return this->idxSample == this->TrainingSampleCount;
}

void NeuralNetwork::ShuffleIdx()
{
	this->idxShuffled = new size_t[TrainingSampleCount];

	for (size_t i = 0; i < TrainingSampleCount; i++)
	{
		idxShuffled[i] = i;
	}

	srand(time(NULL));

	for (size_t i = 0; i < TrainingSampleCount; i++)
	{
		size_t j, t;
		j = rand() % (TrainingSampleCount - i) + i;
		t = idxShuffled[j];
		idxShuffled[j] = idxShuffled[i];
		idxShuffled[i] = t;
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
	delete[] Error;
}