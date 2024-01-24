#pragma once
#include "Neuron.h"
#include <vector>
using namespace std;
class Neuron
{
public:
	double inputVal;   // Value of input
	double activationVal;  // Value after activation function
	double lambda;
	double eta;
	int index;
	double error;
	double alpha;
	std::vector<double> prevDelta;
	int rounds;
	std::vector<double> weights;

	void initiate(double lam, double et, int ind, std::vector<double> w, double alph);

	void calculateActivationValue();

	void calculateInputValue(std::vector<Neuron> neurons);

	double calculateGradient();

	double calculateGradientHidden(std::vector<double> gradients);

	void updateWeight(std::vector<double> gradients);

	double calculateError(double y1);
};

