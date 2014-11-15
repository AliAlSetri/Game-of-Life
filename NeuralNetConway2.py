# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 08:28:3hidden_unit_number + bias 2014

@author: alialsetri
"""

import numpy as np

import random

#Exponential function
def exp(x):
    return np.exp(x)

#Sigmoid non-linearity function
def sig(k,x):
    return 1/(1+exp(-k*x))

#Derivative of Sigmoid function
def sig_prime(k,x):
    return k*exp(-k*x)/(1+exp(-k*x))**2

#This makes a 2D Game of Life array with dimensions ROWSxCOLS
def universe(rows,cols):
	initial = np.zeros((rows,cols))
	for j in range(cols):
		for i in range(rows):
			initial[i,j]=round(random.random())
	return initial

#This function takes in a 2D Game of Life array and returns the next time-step grid.
def evolve(initial):
	(rows,cols)=np.shape(initial)
	neighbor_count_array = np.zeros((rows+2,cols+2))
	framed_universe = np.zeros((rows+2,cols+2))
	framed_universe[1:rows+1,1:cols+1]=initial
	new_universe = np.zeros((rows+2,cols+2))
	for i in range(1,rows+1):
		for j in range(1,cols+1):
			neighbor_count_array[i,j]=sum([framed_universe[i-1,j-1],framed_universe[i-1,j],framed_universe[i-1,j+1],framed_universe[i,j+1],framed_universe[i,j-1],framed_universe[i+1,j],framed_universe[i+1,j-1],framed_universe[i+1,j+1]])
	for i in range(1,rows+1):
		for j in range(1,cols+1):
			if framed_universe[i,j]==1:
				if neighbor_count_array[i,j]<2:
					new_universe[i,j]=0
				elif neighbor_count_array[i,j]==2 or neighbor_count_array[i,j]==3:
					new_universe[i,j]=1
				else:
					new_universe[i,j]=0
			else:
				if neighbor_count_array[i,j]==3:
					new_universe[i,j]=1
				else:
					new_universe[i,j]=0
	return new_universe[1:rows+1,1:cols+1]
			

#This	function takes in a 2D array INITIAL, and returns the array after TIME_STEPS number of time-steps.
def life(initial,time_steps):
        t=0
        while t<time_steps:
                initial = evolve(initial)
                t=t+1
        return initial
        
#Finds the 'distance' between two numpy arrays.
def ssd(a,b):
    squares = (a - b)**2
    return np.sum(squares)
 
#Returns the square of the argument.   
def square(number):
    return number*number




#Below is the code for the neural network.

#This variable determines the size of the grid.   
dimension = 4     

#Bias term of each unit in the network.
bias = 1

#Number of units in the hidden layer. 
hidden_unit_number = 5

#Size of the training set.
training_example_size = 10
    
#Number of times the weights are updated during training. 
num_of_iterations = 300

#Array of arrays. Each array in this array holds the weights of a unit in the hidden layer.
Hidden_Unit_Weights = [np.random.uniform(-1,1,square(dimension) + bias), np.random.uniform(-1,1,square(dimension) + bias), np.random.uniform(-1,1,square(dimension) + bias), np.random.uniform(-1,1,square(dimension) + bias), np.random.uniform(-1,1,square(dimension) + bias)]

#Array of arrays. Each array in this array holds the weights of a unit in the output layer.
Output_Unit_Weights = [np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias), np.random.uniform(-1,1,hidden_unit_number + bias)]


#Input generated based upon the "training_example_size" and "dimension" variables above.
training_example_input = [universe(dimension, dimension) for i in range(training_example_size)]

#Inputs made into 1D arrays with a 1 for bias term appended at the end of each array.
training_example_input_modified = [np.append(i.flatten(), bias) for i in training_example_input]

#Desired output computed for each input.
training_example_output = [evolve(i) for i in training_example_input]

#Desired output made into 1D array form.
training_example_output_modified = [i.flatten() for i in training_example_output]


#This is where the magic happens.
def LearnGameofLife(learning_rate, initial_conditions, time_steps):
    if time_steps == 0:
        return initial_conditions
    t=0
    while t<num_of_iterations:       
        for i in range(len(training_example_input_modified)):
            HiddenLayerOutputs = np.array([sig(1,np.dot(Hidden_Unit_Weights[0], training_example_input_modified[i])), sig(1, np.dot(Hidden_Unit_Weights[1], training_example_input_modified[i])), sig(1,np.dot(Hidden_Unit_Weights[2], training_example_input_modified[i])), sig(1,np.dot(Hidden_Unit_Weights[3], training_example_input_modified[i])), sig(1,np.dot(Hidden_Unit_Weights[4], training_example_input_modified[i])), 1])
            Final_Outputs = np.array([sig(1, np.dot(Output_Unit_Weights[0], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[1], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[2], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[3], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[4], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[5], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[6], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[7], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[8], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[9], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[10], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[11], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[12], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[13], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[14], HiddenLayerOutputs)), sig(1, np.dot(Output_Unit_Weights[15], HiddenLayerOutputs))])
            for neuron in range(len(Output_Unit_Weights)):    
                for weight in range(Output_Unit_Weights[0].size - 1): 
                    Output_Unit_Weights[neuron][weight] = Output_Unit_Weights[neuron][weight] + learning_rate*(training_example_output_modified[i][neuron] - Final_Outputs[neuron])*sig_prime(1, np.dot(HiddenLayerOutputs, Output_Unit_Weights[neuron]))*HiddenLayerOutputs[weight]
            for neuron in range(len(Output_Unit_Weights)):
                Output_Unit_Weights[neuron][hidden_unit_number] = Output_Unit_Weights[neuron][hidden_unit_number] + learning_rate*(training_example_output_modified[i][neuron] - Final_Outputs[neuron])*sig_prime(1, np.dot(HiddenLayerOutputs, Output_Unit_Weights[neuron]))
            for neuron in range(len(Hidden_Unit_Weights)):
                for weight in range(Hidden_Unit_Weights[0].size):
                    for k in range(len(Output_Unit_Weights)):
                        Hidden_Unit_Weights[neuron][weight] = Hidden_Unit_Weights[neuron][weight] + learning_rate*sig_prime(1, np.dot(Hidden_Unit_Weights[neuron], training_example_input_modified[i]))*training_example_input_modified[i][weight]*(training_example_output_modified[i][k] - Final_Outputs[k])*Output_Unit_Weights[k][neuron]
            for neuron in range(len(Hidden_Unit_Weights)):
                for k in range(len(Output_Unit_Weights)):
                    Hidden_Unit_Weights[neuron][square(dimension)] = Hidden_Unit_Weights[neuron][square(dimension)] + learning_rate*sig_prime(1, np.dot(Hidden_Unit_Weights[neuron], training_example_input_modified[i]))*(training_example_output_modified[i][k] - Final_Outputs[k])*Output_Unit_Weights[k][neuron]
        t = t + 1
    time = 0
    temp = np.append(initial_conditions.flatten(), 1)
    while time < time_steps:
        input1 = temp
        Hidden1 = sig(1,np.dot(input1, Hidden_Unit_Weights[0]))
        Hidden2 = sig(1,np.dot(input1, Hidden_Unit_Weights[1]))
        Hidden3 = sig(1,np.dot(input1, Hidden_Unit_Weights[2]))
        Hidden4 = sig(1,np.dot(input1, Hidden_Unit_Weights[3]))
        Hidden5 = sig(1,np.dot(input1, Hidden_Unit_Weights[4]))
        hidden_layer_output = np.array([Hidden1,Hidden2,Hidden3,Hidden4,Hidden5,1])
        output_layer = []
        count = 0
        while count < len(Output_Unit_Weights):
            output_layer.append(sig(1, np.dot(hidden_layer_output, Output_Unit_Weights[count])))
            count = count + 1
        output_layer = np.array(output_layer)
        output_layer = np.rint(output_layer)
        next_grid = np.reshape(output_layer, (dimension,dimension))
        temp = np.append(next_grid.flatten(), 1)
        time = time + 1    
    return next_grid
