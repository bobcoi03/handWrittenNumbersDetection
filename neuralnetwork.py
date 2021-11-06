import numpy
import scipy.special
from matplotlib import pyplot as plt

class NeuralNetwork:
	def __init__(self, inputnodes, hiddennodes, outputnodes, learning_rate):
		self.activation_function = lambda x: scipy.special.expit(x)
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learning_rate

		# random starting weights
		self.wih = numpy.random.normal(0.0,pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0,pow(self.onodes, -0.5),(self.onodes,self.hnodes))
	
	def train(self, inputs_list, targets_list):
		# convert inputs list, targets list into 2d numpy array
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T

		#calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.wih, inputs)
		#calculate signals emergin from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)
		#calculate signals into final output layer
		final_inputs = numpy.dot(self.who, hidden_outputs)
		#calculate signals emergin from final output layer
		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs
		# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors = numpy.dot(self.who.T, output_errors)
		# update the weights for the links between the hidden and output layers       
		self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))      


	def query(self, inputs_list):
		inputs = numpy.array(inputs_list, ndmin=2).T

		hidden_inputs = numpy.dot(self.wih, inputs)

		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who, hidden_outputs)

		final_outputs = self.activation_function(final_inputs)

		return final_outputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.25

bob = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

data_file = open("mnist_dataset/mnist_test_60000.csv","r")
data_list = data_file.readlines()

#plt.imshow(image_array, cmap='Greys',interpolation='None')
#plt.show()
data_file.close()

#train neuralnet by going through data_list
score_list = []
correct_percentage = 0
epochs = 1
for e in range(epochs):
	for record in data_list:
		all_values = record.split(',')
		correct_value = int(all_values[0])
		#convert rgb values of indivdual pixels to range 0.01-0.99
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		targets = numpy.zeros(output_nodes) + 0.01
		targets[int(all_values[0])] = 0.99
		bob.train(inputs, targets)
		outputs = bob.query(inputs)
		index_of_highest_value = numpy.argmax(outputs)

		if index_of_highest_value == correct_value:
			score_list.append(1)
		else:
			score_list.append(0)
print(score_list)
for i in score_list:
	if i == 1:
		correct_percentage += 1
print(f"Correct Percentage: {correct_percentage/len(score_list)*100}")

