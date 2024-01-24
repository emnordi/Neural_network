#pragma once
#include <vector>
#include "Neuron.h"
class Filereader
{
public:
	void readFile(int w, std::vector<double>* xVal1, std::vector<double>* xVal2, std::vector<double>* yVal1, std::vector<double>* yVal2, double* min1, double* max1, double* min2, double* max2, double* min3, double* max3, double* min4, double* max4);
	void storeWeights(vector<Neuron> in, vector<Neuron> hiddens, double lambda, double eta, double alpha);
	void readWeights(vector<Neuron>* ins, vector<Neuron>* hiddens);
};

