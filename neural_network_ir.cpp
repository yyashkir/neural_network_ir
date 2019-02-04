//	Copyright © 2018 Yashkir Consulting
/*
26/12/2018	-	hjm model fast calibration project started
				random driver generation moved out of Monte Carlo loop (cube ji6)
28/12/2018	-	end
*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <sstream>
#include <cstdio>
#include <ctime>
#include "functions_uni.h"
#include <armadillo>
#include <windows.h>
using namespace std;
//
// global variables and constants
string s;
int kmax;	// number of tenors
arma::Row <int> tenors;		//tenors are expressed using  time point array  
arma::Mat <double> historical_dataset;		// input hist data rates for all tenor over period of time
string historical_datafile;	// file with historical yield rates 
arma::Mat <double> train_dataset;		// input hist data rates for all tenor over period of time
arma::Mat <double> validation_dataset;		// input hist data rates for all tenor over period of time

double pi = 3.14159265359;
int i_start_train;		// start day from historical yield data
int train_size;
int i_start_validation;			// length (days) of historical rates used for calibration
int validation_size;
int i_tenor_max;	// index (on axis t for largest tenor
int i_hist;			// total number of lines in file with historical yields
int	back_range;		// number of time steps back for training with historical data
int	fwd_range;		// number of time steps forward for prediction
int  iteration_max_number;
double learning_rate;

clock_t start;

// reading data from input files:
void  read_input_files(char *argv)
{
	int i, k;	
	//
	ifstream infile(argv); //reading control parameters from file argv
	if (!infile.is_open())
	{
		string err = "\nfile ";
		err.append(argv);
		err.append(" cannot be opened");
		cout << err;
		exit(0);
	}
	infile
		>> s >> i_start_train  >> train_size
		>> s >>	i_start_validation	>> validation_size
		>> s >>	historical_datafile
		>> s >> back_range >>	fwd_range 
		>> s >> iteration_max_number
		>> s >> learning_rate
		;
	infile.close();
//
	//reading historical rates 
	ifstream data_stream(historical_datafile.c_str());
	data_stream >> kmax;
	tenors.set_size(kmax);
	data_stream >> i_hist;	// number of lines in file with historical yields
	for (k = 0; k < kmax; k++)
		data_stream >> tenors(k);
//
	historical_dataset.set_size(i_hist, kmax);
	for (i = 0; i < i_hist; i++)
		for (k = 0; k < kmax; k++)
			data_stream >> historical_dataset(i, k);
	data_stream.close();
	cout << "\nHistorical_dataset number of rows: "<< historical_dataset.n_rows <<endl;
//
	train_size = min(train_size, i_hist - i_start_train);
	train_dataset.set_size(train_size, kmax);
	for (i = i_start_train; i < i_start_train + train_size; i++)
		for (k = 0; k < kmax; k++)
			train_dataset(i - i_start_train, k) = historical_dataset(i, k);
//
	validation_size = min(validation_size, i_hist - i_start_validation);
	validation_dataset.set_size(validation_size, kmax);
	for (i = i_start_validation; i < i_start_validation + validation_size; i++)
		for (k = 0; k < kmax; k++)
			validation_dataset(i - i_start_validation, k) = historical_dataset(i, k);
//
	stringstream st, sv;
	st << "Training set: " << train_dataset.n_rows << " rows";
	sv << "Validation set: " << validation_dataset.n_rows << " rows";
	/*train_dataset.print(st.str());
	validation_dataset.print(sv.str());*/
}
//
void iteration_loop()
{
	int it;
	int i, j;
	int k, p;
	double sum;
	ofstream out_obj_fn;
	out_obj_fn.open("err.csv");
	arma::Mat <double> ERR;
	ERR.set_size(iteration_max_number,3);
	ERR.fill(0);
	arma::Row <double> x;	//input (hist)
	arma::Row <double> y;	//output (hist)
	arma::Mat <double> weight_01;	//weights before hidden layer
	arma::Row <double> bias_01;	//bias before hidden layer
	arma::Mat <double> weight_12;	//weights before output layer
	arma::Row <double> bias_12;	//bias before output layer
//
	arma::Row <double> z_01;	// input to hidden layer
	arma::Row <double> z_12;	// input to output layer
	arma::Row <double> a_1;	// output of hidden layer
	arma::Row <double> a_2;	// output of output layer
	arma::Mat <double> a_1all;	// output of hidden layer
	arma::Mat <double> a_2all;	// output of output layer
//
	weight_01.set_size(kmax,kmax);
	bias_01.set_size(kmax);
	weight_12.set_size(kmax, kmax);
	bias_12.set_size(kmax);
//
	z_12.set_size(kmax);
	z_01.set_size(kmax);
	a_1all.set_size(train_size,kmax);
	a_2all.set_size(train_size,kmax);

	a_1.set_size(kmax);
	a_2.set_size(kmax);
//
	arma::Mat <double> delta_12;	// derivatives by z at output layer 2, for all samples for each node
	arma::Mat <double> delta_01;	// derivatives by z at hidden layer 1, for all samples for each node
	arma::Row <double> beta_12;	// derivatives by bias at output layer 2, for each node
	arma::Row <double> beta_01;	// derivatives by bias at hidden layer 1, for each node
	delta_12.set_size(train_size, kmax);
	delta_01.set_size(train_size, kmax);
	beta_12.set_size(kmax);
	beta_01.set_size(kmax);
//
	arma::Mat <double> w_d12;	// derivatives by weight components at output layer 2
	arma::Mat <double> w_d01;	// derivatives by  weight components at hidden layer 1
	w_d01.set_size(kmax, kmax);
	w_d12.set_size(kmax, kmax);
//
	for (k = 0; k < kmax; k++)
	{
		bias_01(k) = arma::randn();
		bias_12(k) = arma::randn();
		for (p = 0; p < kmax; p++)
		{
			weight_01(k, p) = arma::randn();
			weight_12(k, p) = arma::randn();
		}
	}
	weight_01.print("weight_01");
	bias_01.print("bias_01");
	x.set_size(kmax);
	y.set_size(kmax);
//main iteration loop starts here
	for (it = 0; it < iteration_max_number; it++)
	{
		beta_01.fill(0);
		beta_12.fill(0);
		w_d01.fill(0);
		w_d12.fill(0);
		for (i = back_range; i < train_size - fwd_range; i++)
		{
			ERR(it, 0) = it;
		// forward propagation through neural network
			for (k = 0; k < kmax; k++)
			{
				x(k) = train_dataset(i, k);
				//y(k) = train_dataset(i+1, k);
				y(k) = train_dataset(i + fwd_range, k);
			}
			z_01 = x * weight_01 + bias_01;
			for (k = 0; k < kmax; k++)
			{
				a_1(k) = sigmoid(z_01(k));
				a_1all(i, k) = a_1(k);
			}
			z_12 = a_1 * weight_12 + bias_12;
			for (k = 0; k < kmax; k++)
			{
				a_2(k) = sigmoid(z_12(k));
				a_2all(i, k) = a_2(k);
				ERR(it,1) = ERR(it,1) + pow(a_2(k) - y(k), 2) / train_size;
			}
		// backward propagation:
			//at output layer:
			for (k = 0; k < kmax; k++)
				delta_12(i, k) = (a_2(k) - y(k)) * sigmoid_d(z_12(k));
			//at hidden layer:
			for (k = 0; k < kmax; k++)
			{
				sum = 0;
				for(j=0; j<kmax;j++)
					sum = sum + delta_12(i,j) * weight_12(k,j) * sigmoid_d(z_12(k));
				delta_01(i, k) = sum;
			}
			for (k = 0; k < kmax; k++)
			{
				beta_12(k) = beta_12(k) + delta_12(i,k) / train_size;
				beta_01(k) = beta_01(k) + delta_01(i,k) / train_size;
			}
		}

		//validation start
		for (i = 0; i < validation_size - fwd_range; i++)
		{
			// forward propagation through neural network
			for (k = 0; k < kmax; k++)
			{
				x(k) = validation_dataset(i, k);
				//y(k) = validation_dataset(i + 1, k);
				y(k) = validation_dataset(i + fwd_range, k);
			}
			z_01 = x * weight_01 + bias_01;
			for (k = 0; k < kmax; k++)
			{
				a_1(k) = sigmoid(z_01(k));
			}
			z_12 = a_1 * weight_12 + bias_12;
			for (k = 0; k < kmax; k++)
			{
				a_2(k) = sigmoid(z_12(k));
				ERR(it, 2) = ERR(it, 2) + pow(a_2(k) - y(k), 2) / validation_size;
			}
			
		}

		//validation end
		cout << endl << it << " " << ERR(it, 1) << " " << ERR(it, 2);

		
		for (i = back_range; i < train_size - fwd_range; i++)
		{
			for (k = 0; k < kmax; k++)
			{
				for (j = 0; j < kmax; j++)
				{
				//	cout << endl <<i<<" "<< w_d01(k, j) << " " << delta_01(i, j) << " " << a_1all(i, k);
					w_d01(k, j) = w_d01(k, j) + delta_01(i, j) * a_1all(i, k) / train_size;
					w_d12(k, j) = w_d12(k, j) + delta_12(i, j) * a_2all(i, k) / train_size;
				}
			}
		}
		//correction:
		for (k = 0; k < kmax; k++)
		{
			for (j = 0; j < kmax; j++)
			{
				//cout << endl << k << " " << j << " " << weight_01(k, j) << " " << w_d01(k, j);
				weight_01(k, j) = weight_01(k, j) - learning_rate * w_d01(k, j);
				weight_12(k, j) = weight_12(k, j) - learning_rate * w_d12(k, j);

			}
			bias_01(k) = bias_01(k) - learning_rate * beta_01(k);
			bias_12(k) = bias_12(k) - learning_rate * beta_12(k);
		}
	}
	cout << endl << it;
	weight_01.print("w01");
	bias_01.print("b01");
	weight_12.print("w12");
	bias_12.print("b12");
	//ERR.save(out_obj_fn, arma::raw_ascii);
	out_obj_fn << "i,err,val_err";
	for (it = 0; it < iteration_max_number; it++)
		out_obj_fn << endl << ERR(it,0) << "," << ERR(it,1) << "," << ERR(it,2);
	out_obj_fn.close();
}
//	
int main(int argc, char **argv)
{	
	cout << "Reading input files: ";
	read_input_files(argv[1]);	
	iteration_loop();
	
//
	return 0;
}
//	Copyright © 2019 Yashkir Consulting
