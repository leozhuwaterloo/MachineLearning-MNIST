import random
import math
import copy
import sys

class Perceptron():
	def __init__(self, input_count):
		self.weights = []
		self.size = input_count
		for i in range(0, input_count):
			self.weights.append(random.random() * 4 - 2)
	
	def __sigmoid(self, x):
		return (1 / (1 + pow(math.e, -x)))
		
	def getOutput(self, inputs):
		return self.__sigmoid(sum((value * weight) for value, weight in zip(inputs, self.weights)))
		
	def mutate(self):
		index = random.randint(0, self.size-1)
		self.weights[index] += (random.random() * 0.2 - 0.1)
	

class People():
	def __init__(self):
		self.perceptrons =  [Perceptron(2), Perceptron(2)]
		self.size = len(self.perceptrons)
		self.output = Perceptron(2)
	
	def think(self, inputs):
		perceptrons_output = [perceptron.getOutput(inputs) for perceptron in self.perceptrons]
		return self.output.getOutput(perceptrons_output)
		
	def getError(self, input_set, output_set):
		temp_sum = 0
		for inputs, output in zip(input_set, output_set):
			temp_result = self.think(inputs)
			temp_sum += pow(temp_result - output, 2)
		self.current_error = temp_sum
		return self.current_error
		
	def mutate(self):
		index = random.randint(0, self.size + 1)
		if index >= self.size:
			self.output.mutate()
		else:
			self.perceptrons[index].mutate()
		
class Population():
	def __init__(self, size):
		self.size = size
		self.populations = []
		for i in range(0, size):
			self.populations.append(People())
		self.generation = 0
		
	def bestThink(self, inputs):
		return self.populations[0].think(inputs);
		
	def __repr__(self):
		final_string = str(self.populations[0].current_error) + ": "
		final_string += str(self.populations[0].perceptrons[0].weights)
		final_string += str(self.populations[0].output.weights)
		return final_string

	def train(self, input_set, output_set, number_of_training_iterations):
		for i in range(0, number_of_training_iterations):
			self.populations = sorted(self.populations, key=lambda people: people.getError(input_set, output_set))
			if (self.populations[0].current_error < 0.01):
				break;
			my_print(self)
			for i in range(self.size-4, self.size-2):
				self.populations[i] = self.breed()
				self.populations[i].mutate()
			for i in range(self.size-2, self.size):
				self.populations[i] = People()
				
			self.generation += 1
	
	def breed(self):
		father, mother = random.sample(self.populations, 2)
		baby = People()
		baby.perceptrons = copy.deepcopy([random.choice([father_p, mother_p]) for father_p, mother_p in zip(father.perceptrons, mother.perceptrons)])
		baby.output = copy.deepcopy(random.choice([father.output, mother.output]))
		return baby
		
def my_print(text):
    sys.stdout.write(str(text) + '\n')
    sys.stdout.flush()
	
if __name__ == "__main__":
	random.seed(3)
	p = Population(6)
	p.train([(0, 0),(0, 1),(1, 0),(1, 1)], [0,1,1,0], 100000)
	my_print("Generation: " + str(p.generation))
	my_print("(0, 0) -> " + str(p.bestThink((0,0))))
	my_print("(0, 1) -> " + str(p.bestThink((0,1))))
	my_print("(1, 0) -> " + str(p.bestThink((1,0))))
	my_print("(1, 1) -> " + str(p.bestThink((1,1))))
	