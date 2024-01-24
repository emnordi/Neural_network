#include "Filereader.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <Windows.h>

/**
 * readFile function read a file of either training data or validation data
 * loops through the file to retrieve values for the two input and two output neurons
 * also stores the maximum value of each column
 * 
 */
void Filereader::readFile(int w, std::vector<double>* xVal1, std::vector<double>* xVal2, std::vector<double>* yVal1, std::vector<double>* yVal2, double* min1, double* max1, double* min2, double* max2, double* min3, double* max3, double* min4, double* max4) {
	std::fstream file;
	if (w == 1)
	{
		file.open("training1.csv");
	}
	else
	{
		file.open("validate.csv");
	}

	std::string line;
	double temp = 0;
	while (getline(file, line, '\n'))
	{
		std::istringstream templine(line);
		std::string data;
		int i = 0;
		while (getline(templine, data, ','))
		{
			switch (i) {
			case 0:
				temp = std::stod(data);
				xVal1->push_back(temp);
				if (temp > * max3) {
					*max3 = temp;
				}
				else if (temp < *min3) {
					*min3 = temp;
				}
				break;
			case 1:
				temp = stod(data);
				xVal2->push_back(temp);
				if (temp > * max4) {
					*max4 = temp;
				}
				else if (temp < *min4) {
					*min4 = temp;
				}
				break;
			case 2:
				temp = stod(data);
				yVal1->push_back(temp);
				if (temp > * max1) {
					*max1 = temp;
				}
				else if (temp < *min1) {
					*min1 = temp;
				}
				break;
			case 3:
				temp = stod(data);
				yVal2->push_back(temp);
				if (temp > * max2) {
					*max2 = temp;
				}
				else if (temp < *min2) {
					*min2 = temp;
				}
				break;
			}
			i++;
		}

	}
	file.close();
}
/**
 * Stores the trained weights in a csv file so that they can be retrieved without training again
 * iterates over the neurons and storing the weights in a vector which is the iterated over whilst writing to a csv file
 */

void Filereader::storeWeights(vector<Neuron> in, vector<Neuron> hiddens, double lambda, double eta, double alpha) {
	std::fstream file;
	vector<double> ws;
	/*
	 * Iterates over input and hidden neurons and stores their weight in a vector
	 */
	for (int i = 0; i < in.size(); i++) {
		for (int j = 0; j < in[i].weights.size(); j++) {
			ws.push_back(in[i].weights[j]);
		}
	}

	for (int i = 0; i < hiddens.size(); i++) {
		for (int j = 0; j < hiddens[i].weights.size(); j++) {
			ws.push_back(hiddens[i].weights[j]);
		}
	}


	string fn = "lambda=" + std::to_string(lambda);
	fn += "_eta=" + std::to_string(eta);
	fn += "_alpha=" + std::to_string(alpha);
	fn += "_HiddenNeurons=" + std::to_string(hiddens.size());
	fn += ".csv";
	/*
	 * Stores all the weights from input and hidden neurons in csv file
	 */
	file.open(fn, fstream::out);
	for (int i = 0; i < ws.size(); i++)
	{
		file << ws[i] << "\n";
	}
	file.close();
}
/**
 * retrieves the weights from a csv file and inputs them into the neurons
 */
void Filereader::readWeights(vector<Neuron> * ins, vector<Neuron> * hiddens) {
	//lambda=0.700000_eta=0.800000_alpha=0.050000_HiddenNeurons=5.csv works as well
	string fname = "lambda=0.800000_eta=0.800000_alpha=0.050000_HiddenNeurons=5.csv";
	std::fstream file;
	file.open(fname);
	std::string line;
	vector<double> ws;
	/*
	 * Iterate over file
	 */
	while (getline(file, line, '\n'))
	{
		ws.push_back(std::stod(line));
	}
	int wscount = 0;
	/*
	Add weights from input-> hidden
	*/
	for (int i = 0; i < ins->size(); i++) {
		for (int j = 0; j < hiddens->size() - 1; j++) {
			ins->at(i).weights[j]= ws[wscount++];
		}
	}
	/*
	Add weights from hidden->output
	*/
	for (int i = 0; i < hiddens->size(); i++) {
		for (int j = 0; j < 2; j++) {
			hiddens->at(i).weights[j] = ws[wscount++];
		}
	}



	file.close();
}