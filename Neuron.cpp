#include "Neuron.h"
#include <vector>

/*
 * The neuron class is used to represent a neuron, the functions represent the functionality of a neuron
 * including calculating all the values needed for feedforward and backpropagation
 */


/*
 * Initialise all the parameters
 */
void Neuron::initiate(double lam, double et, int ind, std::vector<double> w, double alph)
{
	lambda = lam;
	eta = et;
	index = ind;
	weights = w;
	rounds = 0;
	alpha = alph;
}
/*
 * Calculate activation value based on input value and parameters using sigmoid function
 */
void Neuron::calculateActivationValue()
{
	activationVal = 1 / (1 + exp(-lambda * inputVal));
}

/*
 * Calculate input value by iterating over activation values and weights from previous layer
 */
void Neuron::calculateInputValue(std::vector<Neuron> neurons)
{
	inputVal = 0;
	for (int i = 0; i < neurons.size(); i++)
	{
		inputVal += neurons[i].weights[index - 1] * neurons[i].activationVal;
	}
	calculateActivationValue();
}

/*
 * Calculate gradient if neuron is in output layer
 */
double Neuron::calculateGradient()
{
	return lambda * activationVal * (1 - activationVal) * error;
}
/*
 * Calculate gradient if neuron is in hidden layer
 */
double Neuron::calculateGradientHidden(std::vector<double> gradients)
{
	double temp = 0;

	for (int i = 0; i < weights.size(); i++) {
		temp += gradients[i] * weights[i];
	}
	;			return lambda * activationVal * (1 - activationVal) * temp;
}
/*
 * Update weights by calculating delta weights and adding them onto the current weights
 * Iterates over 
 */
void Neuron::updateWeight(std::vector<double> gradients)
{
	for (int i = 0; i < weights.size(); i++) {
		if (rounds == 0) {
			double temp = eta * activationVal * gradients[i];
			prevDelta.push_back(temp);
			weights[i] += temp;

		}
		else {
			double temp = eta * activationVal * gradients[i];
			temp += prevDelta[i] * alpha;
			prevDelta[i] = temp;
			weights[i] += temp;
		}
	}
	rounds++;

}
/*
 * Calculate error as difference between the input y value and the predicted y value
 */
double Neuron::calculateError(double y1)
{
	error = y1 - activationVal;
	return error;
}
