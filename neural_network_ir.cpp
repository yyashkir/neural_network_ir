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
int ny;
clock_t start;
int flag = 0;
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
		>> s >> ny
		;
	infile.close();
//
	//reading historical rates 
	ifstream data_stream(historical_datafile.c_str());
	data_stream >> kmax;		// number of tenors (=columns)
	tenors.set_size(kmax);
	data_stream >> i_hist;		// number of lines in file with historical yields
	for (k = 0; k < kmax; k++)
		data_stream >> tenors(k);
//
	historical_dataset.set_size(i_hist, kmax);
	double max_v = 0;
	for (i = 0; i < i_hist; i++)
		for (k = 0; k < kmax; k++)
		{
			data_stream >> historical_dataset(i, k);
			if (max_v < historical_dataset(i, k)) max_v = historical_dataset(i, k);
		}
	data_stream.close();
	/*for (i = 0; i < i_hist; i++)
		for (k = 0; k < kmax; k++)
			historical_dataset(i, k) = historical_dataset(i, k) / max_v;*/

	cout << "\nHistorical_dataset number of rows: "<< historical_dataset.n_rows <<"  max element is= "<<max_v<<endl;
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
	cout<<endl<<st.str() <<endl<<sv.str()<<endl;
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
	int it_min = -10;
	ofstream out_obj_fn;
	out_obj_fn.open("err.csv");
	arma::Mat <double> ERR;
	ERR.set_size(iteration_max_number,3);
	ERR.fill(0);
	ofstream a2y;
	a2y.open("a2y.csv");
	a2y << "y,a";
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
	arma::Mat <double> a_1all;	// output of hidden layer for ech sample
	arma::Mat <double> a_2all;	// output of output layer for each sample
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
	// random fill of all weights and biases:
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
	/*weight_01.print("weight_01");
	bias_01.print("bias_01");*/
	x.set_size(kmax);
	y.set_size(kmax);
	int flag = 0;
//main iteration loop starts here:
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
				x(k) = train_dataset(i, k);				//rates at date 'i' for tenor number 'k' (input)
				y(k) = train_dataset(i + fwd_range, k);	//rates at date 'i+fwd_range' for tenor number 'k' (output)
			}
			z_01 = x * weight_01 + bias_01;	//vector of weighted averages entering hidden layer '1'
			for (k = 0; k < kmax; k++)
			{
				a_1(k) = sigmoid(z_01(k));	//output of neuron 'k' of hidden layer
				a_1all(i, k) = a_1(k);		//memorizing it for i-th sample
			}
			z_12 = a_1 * weight_12 + bias_12;//vector of weighted averages entering output layer '2'
			for (k = 0; k < kmax; k++)
			{
				a_2(k) = sigmoid(z_12(k));	//output of neuron 'k' of output layer
				a_2all(i, k) = a_2(k);		//memorizing it for i-th sample
				ERR(it,1) = ERR(it,1) + pow(a_2(k) - y(k), 2) / train_size;	//accumulation of errors
			}

		// backward propagation:
			//at output layer:
			for (k = 0; k < kmax; k++)
				delta_12(i, k) = (a_2(k) - y(k)) * sigmoid_d(z_12(k));// derivative by z(k) at output layer 2, for all sample 'i' for node 'k'
			//at hidden layer:
			for (k = 0; k < kmax; k++)
			{
				sum = 0;
				for(j=0; j<kmax;j++)
					sum = sum + delta_12(i,j) * weight_12(k,j) * sigmoid_d(z_12(k));
				delta_01(i, k) = sum;	//derivative by z(k) at hidden layer for sample 'i'
			}
			// accumulation of derivatives by biases (getting in the end average by all samples)
			for (k = 0; k < kmax; k++)
			{
				beta_12(k) = beta_12(k) + delta_12(i,k) / train_size;
				beta_01(k) = beta_01(k) + delta_01(i,k) / train_size;
			}
		}	//single iteration forward/back propagation & derivatives calculation ends here

		//validation starts here:
		for (i = 0; i < validation_size - fwd_range; i++)
		{
			// forward propagation through neural network
			for (k = 0; k < kmax; k++)
			{
				x(k) = validation_dataset(i, k);				//input
				y(k) = validation_dataset(i + fwd_range, k);	//output
			}
			z_01 = x * weight_01 + bias_01;		// weighted average for entry to hidden layer '1'
			for (k = 0; k < kmax; k++)
			{
				a_1(k) = sigmoid(z_01(k));		//out of hidden layer '1'
			}
			z_12 = a_1 * weight_12 + bias_12;	// weighted average for entry to output layer '2'
			for (k = 0; k < kmax; k++)
			{
				a_2(k) = sigmoid(z_12(k));		// out of output layer '2'
				ERR(it, 2) = ERR(it, 2) + pow(a_2(k) - y(k), 2) / validation_size;	//error accumulation
			}
			if (it == it_min + 2 || it == iteration_max_number - 1  && flag == 0)
			{
				a2y << endl << y(ny) << "," << a_2(ny);
				if(i==0)a2y << endl << y(ny) << "," << 0;
			}
		}	//validation end
		
		if (it > 1 && ERR(it - 2, 2) >= ERR(it - 1, 2) && ERR(it - 1, 2) <= ERR(it, 2) || it == iteration_max_number - 1)
		{
			it_min = it - 1;
			flag = 1;
		}
//
		if ((it / 50) * 50 == it) 	cout <<"							"<< it << "\r";
//
		// derivatives by weights calculation:
		for (i = back_range; i < train_size - fwd_range; i++)
		{
			for (k = 0; k < kmax; k++)
			{
				for (j = 0; j < kmax; j++)
				{
					w_d01(k, j) = w_d01(k, j) + delta_01(i, j) * a_1all(i, k) / train_size;
					w_d12(k, j) = w_d12(k, j) + delta_12(i, j) * a_2all(i, k) / train_size;
				}
			}
		}
		//gradient downhill step:
		for (k = 0; k < kmax; k++)
		{
			for (j = 0; j < kmax; j++)
			{
				weight_01(k, j) = weight_01(k, j) - learning_rate * w_d01(k, j);
				weight_12(k, j) = weight_12(k, j) - learning_rate * w_d12(k, j);

			}
			bias_01(k) = bias_01(k) - learning_rate * beta_01(k);
			bias_12(k) = bias_12(k) - learning_rate * beta_12(k);
		}
		
	}	//iteration loop end
	
	
	double min_val_err = 1e12;
	for (it = 1; it < iteration_max_number; it++)
		if (ERR(it, 2) < min_val_err)
		{
			it_min = it;
			min_val_err = ERR(it, 2);
		}
	cout << endl << "val err ("<<min_val_err<<") is min for it=" << it_min;
	cout << endl << it <<" iterations completed"<<endl;
	
	//ERR.save(out_obj_fn, arma::raw_ascii);
//	error vs iterations saving to a file:
	out_obj_fn << "iteration,err,validation_err";
	for (it = 0; it < iteration_max_number; it++)
		out_obj_fn << endl << ERR(it,0) << "," << ERR(it,1) << "," << ERR(it,2);
	out_obj_fn.close();
//
	a2y.close();
}
//	
int main(int argc, char **argv)
{	
	cout << "Reading input files: ";
	read_input_files(argv[1]);	
	iteration_loop();
	system("play.bat");
//
	// errors vs iteration: python code call
	system("python view_loglog.py");
	//system("python view_logy.py");
	system("python a2y.py");
//
	return 0;
}
//	Copyright © 2019 Yashkir Consulting
