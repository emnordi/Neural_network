#include "Aria.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <Windows.h>
#include <algorithm>
#include <thread>
#include "Neuron.h"
#include "Filereader.h"
#include <string>
using namespace std;

/*
 * The main file of the system.
 * The neurons are created here and the network trained.
 * It also contains the code for controlling the robot based on predictions
 */

double lambda = 0.8;
double eta = 0.7;
const int hiddenNeurons = 4; //Amount of hidden layer neurons
const int inputs = 2; //Amount of input neurons
const int outputs = 2; //Amount of output neurons
double alpha = 0.05; //Momentum
std::vector<double> gradients; //Stores gradients for the output layer
std::vector<double> gradientsHidden; //Stores gradients for the hidden layer
std::vector<Neuron> ins; //Stores all input neurons
std::vector<Neuron> hiddens; //Stores all hidden layer neurons
std::vector<Neuron> outs; // Stores all output neurons
vector<double> errors; //During training this vector stores the output error used for gradient calculation
vector<double> epokerrors; //Holds the error for each epoch based on training labels and prediction
vector<double> validationerror; //Holds the error based on prediction and each row of the test data for one epoch
vector<double> validationtotal; //Holds the MSE error based on the testing data of each epoch to evaluate model
vector<int> iterate; //Vector holds all values from 0 up to the length of the training data so that it can be shuffled in order to not train data in same order
int rounds = 0; //Amount of training rounds of each epoch
int epoks = 500; //Amount of epoch to train before stopping
boolean cont = true; //User can set a specific error value they want to stop at 
double stoppingerror = 0;
std::vector<double> xValues1; //stores left input values from training data
std::vector<double> xValues2; //stores forward input values from training data
std::vector<double> yValues1; //stores left output values from training data
std::vector<double> yValues2; //stores forward output values from training data

int train = 0; //Set 0 to train data and 1 to use robot
int main(int argc, char** argv)
{
	Filereader fr;
	fstream file;
	vector<double> iWeights[inputs + 1]; //used to hold initial random weights for input layer
	vector<double> hWeights[hiddenNeurons + 1]; //used to hold initial random weights for hidden layer
	vector<double> w1;
	vector<double> w2;
	/*
	 * Used to set maximum and minimum values of each column to normalise and denormalise data
	 */
	double max1 = 0;
	double min1 = 500;
	double max2 = 0;
	double min2 = 500;
	double max3 = 0;
	double min3 = 5000;
	double max4 = 0;
	double min4 = 5000;

	/*
	 * Reads the training data files and stores all the values while also finding maximum and minimum values for each column
	 */
	fr.readFile(1, &xValues1, &xValues2, &yValues1, &yValues2, &min1, &max1, &min2, &max2, &min3, &max3, &min4, &max4);
	rounds = xValues1.size();
	for (int i = 0; i < rounds; i++) {
		iterate.push_back(i);
	}


	/*
	 *Get random values between -1 and 1 for the initial weights
	 */
	for (int i = 0; i <= inputs; i++) {
		for (int j = 0; j < hiddenNeurons; j++) {
			w1.push_back(((rand() % (10000)) / 5000.0) - 1);
		}
		iWeights[i] = w1;
		w1.clear();
	}

	for (int i = 0; i <= hiddenNeurons; i++) {
		for (int j = 0; j < outputs; j++) {
			w2.push_back(((rand() % (10000)) / 5000.0) - 1);
		}
		hWeights[i] = w2;
		w2.clear();
	}

	/*
	*Create input neurons and initialise them with input parameters and random weights
	*Also create bias neuron with activation value of 1
	*/
	Neuron bias;
	bias.initiate(lambda, eta, 0, iWeights[0], alpha);
	bias.activationVal = 1;
	ins.push_back(bias);
	for (int i = 0; i < inputs; i++) {
		Neuron temp;
		temp.initiate(lambda, eta, i + 1, iWeights[i + 1], alpha);
		ins.push_back(temp);
	}

	/*
	*Create hidden neurons and initialise them with input parameters and random weights
	*Also create bias neuron with activation value of 1
	*/
	Neuron bias2;
	bias2.initiate(lambda, eta, 0, hWeights[0], alpha);
	bias2.activationVal = 1;
	hiddens.push_back(bias2);
	for (int i = 0; i < hiddenNeurons; i++) {
		Neuron temp;
		temp.initiate(lambda, eta, i + 1, hWeights[i + 1], alpha);
		hiddens.push_back(temp);
	}

	/*
	* Create output neurons and initialise them with input parameters
	*/
	for (int i = 0; i < outputs; i++) {
		Neuron temp;
		temp.initiate(lambda, eta, i + 1, iWeights[i + 1], alpha);
		outs.push_back(temp);

	}

	/*
	 * If train variable is 0 the neural network will use the parameters and training data to train a neural network
	 */
	if (train == 0) {
		/*
		 * xvalcount and yvalcount is an iterator for the x and y values
		 */
		int xvalcount = 0;
		int yvalcount = 0;
		//variable used to keep track of current epoch
		int epok = 0;
		/*
		 * Training starts here
		 * The main loop while loop until either the set number or epoch is reached or a requested error value reached
		 */
		while (epok < epoks && cont) {

			/*
			* Second loop will iterate over each row of training data to do both feedforward and backpropagation
			*/
			for (int r = 0; r < rounds; r++) {

				//Enter input values for input neurons which are minimised
				double minimised = (xValues1[iterate[xvalcount]] - min3) / (max3 - min3);
				ins[1].activationVal = minimised;
				minimised = (xValues2[iterate[xvalcount++]] - min4) / (max4 - min4);
				ins[2].activationVal = minimised;

				//Calculate vi^h and hi
				for (int i = 1; i <= hiddenNeurons; i++) {
					hiddens[i].calculateInputValue(ins);
				}

				//Calculate vi, yi, error and gradient 
				outs[0].calculateInputValue(hiddens);
				minimised = (yValues1[iterate[yvalcount]] - min1) / (max1 - min1);
				errors.push_back(outs[0].calculateError(minimised));
				gradients.push_back(outs[0].calculateGradient());

				outs[1].calculateInputValue(hiddens);
				minimised = (yValues2[iterate[yvalcount++]] - min2) / (max2 - min2);
				errors.push_back(outs[1].calculateError(minimised));
				gradients.push_back(outs[1].calculateGradient());

				/*
				Calculate gradient for hidden neurons
				*/
				for (int i = 1; i < hiddens.size(); i++) {
					gradientsHidden.push_back(hiddens[i].calculateGradientHidden(gradients));
				}

				/*
				Update hidden -> output weights
				*/
				for (int i = 0; i < hiddens.size(); i++) {
					hiddens[i].updateWeight(gradients);
				}

				/*
				Update input -> hidden weights
				*/

				for (int i = 0; i < ins.size(); i++) {
					ins[i].updateWeight(gradientsHidden);
				}
				gradients.clear();
				gradientsHidden.clear();

			}
			/*
			 * Calculate an error value representing the epoch based on the error of each prediction
			 */
			double temp = 0;
			for (int i = 0; i < errors.size(); i += 2)
			{
				temp += pow((errors[i] + errors[i + 1]), 2) / 2;
			}
			errors.clear();
			epokerrors.push_back(temp / rounds);
			xvalcount = 0;
			yvalcount = 0;

			//Reset neuron variables after epoch
			for (int i = 0; i < ins.size(); i++) {
				ins[i].rounds = 0;
				ins[i].prevDelta.clear();
			}
			for (int i = 0; i < hiddens.size(); i++) {
				hiddens[i].rounds = 0;
				hiddens[i].prevDelta.clear();
			}
			//Shuffle order of training data
			random_shuffle(iterate.begin(), iterate.end());

			//Stop training if preferred error value is reached
			if (temp / rounds < stoppingerror) {
				cout << "Finished training in " << epok << " epochs\n";
				cont = false;
			}

			/*
			* After each epoch the validation dataset is used to calculate how well the neural net is performing
			*/
			vector<double> valx1;
			vector<double> valx2;
			vector<double> valy1;
			vector<double> valy2;
			double mm = 0, mmm = 10, ma = 10, mam = 0;
			fr.readFile(2, &valx1, &valx2, &valy1, &valy2, &mm, &mmm, &ma, &mam, &mm, &mmm, &ma, &mam);
			int inc = 0;
			/*
			 * Iterate over all the validation data and predict values for each point
			 */
			while (inc < valx1.size())
			{
				/*
				 * Each value is checked to not be smaller or larger than the minimum or maximum values found from the training data so normalisation
				 * will be correct
				 */
				double temp1 = valx1[inc];
				double temp2 = valx2[inc];
				if (temp1 > max3)
				{
					temp1 = max3;
				}
				if (temp1 < min3)
				{
					temp1 = min3;
				}
				if (temp2 > max4)
				{
					temp2 = max4;
				}

				if (temp2 < min4)
				{
					temp2 = min4;
				}
				//Enter input values for input neurons

				ins[1].activationVal = (temp1 - min3) / (max3 - min3);
				ins[2].activationVal = (temp2 - min4) / (max4 - min4);


				//Calculate vi^h and hi
				for (int i = 1; i <= hiddenNeurons; i++) {
					hiddens[i].calculateInputValue(ins);
				}

				//Calculate vi, yi, error and gradient
				for (int i = 0; i < outputs; i++) {
					outs[i].calculateInputValue(hiddens);
				}
				/*
				 * Predict two y values and denormalise them to compare the result to the actual values of the validation data
				 */
				double pred1 = ((outs[0].activationVal * (max1 - min1)) + min1);
				double pred2 = ((outs[1].activationVal * (max2 - min2)) + min2);
				double er = sqrt(pow(valy1[inc] - pred1, 2));
				double er2 = sqrt(pow(valy2[inc] - pred2, 2));
				validationerror.push_back((er + er2) / 2);
				inc++;
			}
			/*
			 * Calculate error values based on the performance of the net on the validation data
			 */
			double totalerror = 0;
			for (int i = 0; i < validationerror.size(); i++) {
				totalerror += validationerror[i];
			}
			validationtotal.push_back(totalerror / validationerror.size());
			validationerror.clear();
			epok++;
			if (epok == epoks) {
				cout << "Could not reach an error of " << stoppingerror << " in " << epok << " epochs" << "\n";
			}

		}
		/*
		 * A csv file containing the error values based on the validation data for each epoch is written to.
		 * These values are used to evaluate how good the model is performing.
		 */

		string fname = "l=" + to_string(lambda);
		fname += "_e=" + to_string(eta);
		fname += "_validation";
		fname += ".csv";
		file.open(fname, fstream::out);
		for (int i = 0; i < validationtotal.size(); i++)
		{
			file << validationtotal[i] << "\n";
		}
		file.close();

		fr.storeWeights(ins, hiddens, lambda, eta, alpha);
		/*
		* A csv file containing the error values based on the error of each round for each epoch is written to.
		* These values are used to evaluate how good the model is performing.
		*/
		string fnam = "l=" + to_string(lambda);
		fnam += "_e=" + to_string(eta);
		fnam += ".csv";
		file.open(fnam, fstream::out);

		for (int i = 0; i < epokerrors.size(); i++)
		{
			file << epokerrors[i] << "\n";
		}
		file.close();
	}
	else {
		//Retrieves stored weights from a csv file
		fr.readWeights(&ins, &hiddens);
	}


	/*
	* The code below is for activating and controlling the robot.
	* The neural net predicts output values based on the input values
	*/

	if (train == 1)
	{
		/*
		 * Initialise and connect robot
		 */

		Aria::init();
		ArRobot robot;
		ArArgumentParser argParser(&argc, argv);
		argParser.loadDefaultArguments();

		//Connect
		ArRobotConnector robotConnector(&argParser, &robot);
		if (robotConnector.connectRobot())
			std::cout << "Robot connected!" << std::endl;

		robot.runAsync(false);
		robot.lock();
		robot.enableMotors();
		robot.unlock();

		double sonarRange[8];
		/*
		 * Reads sensor values, predicts output values based on the input sensor values
		 * and then set wheel speed based on the output predictions
		 */
		while (true)
		{
			/*
			 * Retrieve sonar readings and store them in array
			 */
			for (int i = 0; i < 8; i++) {
				sonarRange[i] = robot.getSonarReading(i)->getRange();
			}
			/*
			 * The input values are chosen as the smallest values of the sensors 0, 1 and 2, 3
			 */
			if (sonarRange[1] < sonarRange[0])
			{
				sonarRange[0] = sonarRange[1];
			}
			if (sonarRange[3] < sonarRange[2])
			{
				sonarRange[1] = sonarRange[3];
			}
			else
			{
				sonarRange[1] = sonarRange[2];
			}
			/*
			 * The values are made sure to be in the range of the minimum and maximum values retrieved from training data
			 */
			if (sonarRange[0] > max3)
			{
				sonarRange[0] = max3;
			}
			if (sonarRange[0] < min3)
			{
				sonarRange[0] = min3;
			}
			if (sonarRange[1] > max4)
			{
				sonarRange[1] = max4;
			}

			if (sonarRange[1] < min4)
			{
				sonarRange[1] = min4;
			}

			//Enter input values for input neurons also normalise them
			ins[1].activationVal = (sonarRange[0] - min3) / (max3 - min3);
			ins[2].activationVal = (sonarRange[1] - min4) / (max4 - min4);


			//Calculate vi^h and hi
			for (int i = 1; i <= hiddenNeurons; i++) {
				hiddens[i].calculateInputValue(ins);
			}

			//Calculate vi, yi, error and gradient
			for (int i = 0; i < outputs; i++) {
				outs[i].calculateInputValue(hiddens);
			}
			/*
			 * Denormalise the prediction value and set the speed of the robot wheels based on the denormalised prediction
			 */
			robot.setVel2(outs[0].activationVal * (max1 - min1) + min1, outs[1].activationVal * (max2 - min2) + min2);
			//Small pause between readings
			ArUtil::sleep(100);
		}

	}


	return 0;
}