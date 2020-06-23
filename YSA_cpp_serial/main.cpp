#include <stdio.h>
#include "NeuralNetwork.h"
#include <chrono>

int main()
{
    // 1 - Initiate NeuralNetwork Object
    NeuralNetwork YSA(128, 10);
    // 2 - Load Trainig Samples and Labels
    YSA.ReadTrainingSamples("../Train60000_1D_Array.txt");
    YSA.ReadLabels("../Labels60000_1D_Array.txt");
    YSA.RandomizeWeights();
    YSA.NormalizeInput();

    // 3 - Start Training
    auto start = std::chrono::high_resolution_clock::now();

    YSA.Train();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double > fp_ms = end - start;

    // 4 - Calculate Training Time and Test
    YSA.Test();
    std::cout << "\nTRAINING TIME : " << fp_ms.count() << "  seconds.." << "\n";


    return 0;
}