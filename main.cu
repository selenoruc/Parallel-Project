#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "NeuralNetwork.h"

int main()
{
    // 1 - Initiate NeuralNetwork Object
    NeuralNetwork YSA(128, 10);
    // 2 - Load Trainig Samples and Labels
    YSA.ReadTrainingSamples("Train60000_1D_Array.txt");
    YSA.ReadLabels("Labels60000_1D_Array.txt");
    YSA.RandomizeWeights();
    YSA.NormalizeInput();
    // 3 - Start Training
    YSA.Iterate();
    YSA.printOut();
    printf("OK\n");
    // 4 - Loop While Not Trained

    return 0;
}