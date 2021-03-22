import numpy as np
import random
from random import randrange

class NeuralNetwork():
    INPUT_NEURONS = 3
    OUTPUT_NEURONS = 1

    def __init__(self,first_layer,second_layer,training_inputs,expected_output):
        #inicializamos los parametros del codigo
        self.total_fitness = 0
        self.params_output = {}
        self.sum_error_iteration = []
        self.num_neurons = [self.INPUT_NEURONS, first_layer, second_layer, self.OUTPUT_NEURONS]
        self.layers= []
        # lo pasamos a float para que se pueda operar con las generaciones mutadas
        self.training_inputs = training_inputs.astype(float)
        self.expected_output = expected_output.astype(float)
        self.init_layers()
    
    def init_layers(self):
        # inicializamos las Layers con el numero de neuronas seleccionadas
        for previous_neurons, current_neurons in zip(self.num_neurons, self.num_neurons[1:]):
            self.layers.append(NeuronLayer(previous_neurons, current_neurons))

    def reset_total_fitness(self):
       self.total_fitness = 0

    def forward_propagation(self):
        # cada vez que realizamos un forward propagation se han de resetera los antiguos fitness ya que sino se irian sumando
        self.reset_total_fitness()
        # calculamos la salida de cada una de los layers
        for idx, train_input  in enumerate(self.training_inputs):
            first = self.layers[0].get_output_neuron(train_input)
            second = self.layers[1].get_output_neuron(np.array(first).T)
            final = self.layers[2].get_output_neuron(np.array(second).T)
            # miramos el resultado y calculamos el fitness total (lo predicho - lo que deberia salir)
            self.total_fitness += pow(final-self.expected_output[idx][0], 2)
        return self.total_fitness.sum()

    def check_trained_algorithm(self,final_input):
        # calculamos el resultado final 
        first = self.layers[0].get_output_neuron(final_input)
        second = self.layers[1].get_output_neuron(np.array(first).T)
        final = self.layers[2].get_output_neuron(np.array(second).T)
        
        return final

    def create_generation(self,best_output, second_best_output):
        # Hacemos una iteracion de las diferentes layers 
       for idx_layer, layer in enumerate(self.layers):
            # hacemos una iteracion de las diferentes neuronas dentro de cada layer
            for neurons in range(0,self.num_neurons[(idx_layer+1)]):
                # mutamos cada neurona a traves de la misma neurona de la mejor o segunda mejor generacion anterior
                if (second_best_output != 0):
                    layer.neurons[neurons].mutate(best_output.layers[idx_layer].neurons[neurons],second_best_output.layers[idx_layer].neurons[neurons])
                else:
                    layer.neurons[neurons].mutate(best_output.layers[idx_layer].neurons[neurons],0)


class Neuron:
    
    # Al inicializar esta classe se le asignaran cuantas connexiones de entrada tiene
    def __init__ (self, input_connections):
        #son random.randn es una funcion randn para obtener floats de una distribucion normal
        # pesos de -1 a 1 
        self.weights = np.random.uniform(low=-1.0, high=1, size=input_connections)
        # entre -n y n
        self.bias = randrange(-input_connections, input_connections)
        # establecemos la mutacion que tendra puede que se tenga que ajustar segun si se quiere que mute más o menos rapido
        self.mutation = 0.5
        self.input_connections = input_connections

    def sigmoid(self, x, derivative = False):
        if (derivative):
            #Calculamos el resultado de la derivada de la funcion sigmoid
            #Sirve para realizar los ajustes de los weights 
            return x * (1 - x)
        else:
            # calculamos el resultado de la funcion sigmoid
            # Esta funcion marca que el resultado este entre 0 y 1 con una media de 0
            return 1 / (1 + np.exp(-x))

    def calculate_output(self, input):
        # multiplicamos las entradas por sus pesos escogidos al azar con su bias 
        # y aplicando la funcion resultante sigmoid
        return self.sigmoid(np.dot(input, self.weights ) + self.bias)
    
    def mutate(self,old_best_neuron, second_old_best_neuron):
        """ 
         hacemos una mutacion de la mejor weight y bias de la mejor generacion de la anterior iteracion
         en el caso de los weights al depender de los inputs y tener una diferente segun cada uno de ellos 
         se pasa por la funcion, se puede optimizar mas si al declarar ya se guarda el num inputs, así no se pasa por parametro
         """
        for inputs in range(self.input_connections):
            old_neuron_selected = old_best_neuron
            if (second_old_best_neuron !=0):
                # first or second layer
                if (randrange(0,2)==0):
                    old_neuron_selected = second_old_best_neuron
            # ahora en la mutacion
            if (randrange(0,2)==0):
                self.weights[inputs] = old_neuron_selected.weights[inputs] + self.mutation*random.uniform(-1, 1)
            else:
                self.weights[inputs] = old_neuron_selected.weights[inputs]
        old_neuron_selected = old_best_neuron
        if (second_old_best_neuron !=0):
            # first or second layer
            if (randrange(0,2)==0):
                old_neuron_selected = second_old_best_neuron
        # ahora en la mutacion
        if (randrange(0,2)==0):
            self.bias = old_neuron_selected.bias + self.mutation*random.uniform(-1, 1)
        else:
            self.bias = old_neuron_selected.bias


class NeuronLayer():
    def __init__(self, inputs_last_layer, neurons_in_layer):
        self.inputs_last_layer = inputs_last_layer
        self.neurons_in_layer = neurons_in_layer
        self.neurons = []
        # creamos x neuronas dentro la layer
        # cada neurona se crea con las x connexiones de la layer anterior
        for neuron in range(neurons_in_layer):
            self.neurons.append(Neuron(inputs_last_layer))

    def get_output_neuron(self, input):
        output = []
        for neuron in self.neurons:
            output.append(neuron.calculate_output(input))
        return output
