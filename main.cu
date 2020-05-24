#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "NeuralNetwork.h"
#include <chrono>




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
    
    // counterCYCLE
   /* while(!YSA.isTrained())*/

    auto start = std::chrono::high_resolution_clock::now();

    YSA.Train();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double > fp_ms = end - start;
    std::cout << " Gecen Sure :" << fp_ms.count() << "\n";

    //YSA.printOut();
    printf("OK\n");
    // 4 - Loop While Not Trained

    return 0;
}