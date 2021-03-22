# -*- coding: utf-8 -*-
import numpy as np
from random import randrange

from .ui import style
from .neuralnetwork import neuralnetwork as nn

####### Variables de entreno ####
# Los trainigs seran 7 ejemplos de 8 disponibles con 3 possibles entradas
TRAINING_INPUTS = np.array([[0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0]])
# Consisten en las 7 possibles salidas controladas para entrenar el algoritmo
# .T es para transposar la matriiz es decir para que sea un 7x1
EXPECTED_OUTPUTS = np.array([[1,0,1,0,1,0,1]]).T
# Ultimo input una vez entrenado el codigo
INPUT_ONES_TRAINED = np.array([1,1,1])
# ultimo output que tendria que salir una vez entrenado el codigo
# para comprovar si ha ido correctamente
EXPECTED_FINAL_OUTPUT = 0

different_neural_networks = []

def init_generations(generations, first_layer, second_layer):
    for generation in range(generations):
        different_neural_networks.append(nn.NeuralNetwork(first_layer, second_layer, TRAINING_INPUTS, EXPECTED_OUTPUTS))

def calculate_generation_fitness(generations):
    fitness = []
    for generation in range(generations):
        fitness.append(different_neural_networks[generation].forward_propagation())

    return fitness

def get_index_best_network(generations, fitness):
    if (generations > 1 ):
        for generation in range(generations):
            if (different_neural_networks[generation].total_fitness == fitness[0]):
                best_fitness = generation
            elif (different_neural_networks[generation].total_fitness == fitness[1] ):
                    second_best_fitness = generation
    else:
        return 0 , -1
    return best_fitness, second_best_fitness

def main():
    # Style instance 
    s = style.StyleGrupo08()

    # Best fitness list
    best_fitness_record = []
    # pedimos al usuario que introduzca los datos de la red neuronal
    first_layer_neuronal_network = s.ask_user(s.FIRSTNEURONLAYER) 
    second_layer_neuronal_network = s.ask_user(s.SECONDNEURONLAYER) 
    
    #cuantas redes neuronales se crearan en paralelo cada ejecucion para escoger las dos mejores de cada generacion
    individuals_neural_network = s.ask_user(s.INDIVIDUALS)
    generations_neural_network = s.ask_user(s.GENERATIONS)
    
    # Inicializamos por primera vez las x generaciones 
    init_generations(
        individuals_neural_network,
        first_layer_neuronal_network,
        second_layer_neuronal_network,
        )

    for iteration in range(generations_neural_network):
        # realizamos el forward propagation y obtenemos el fitness de cada generacion
        all_fitness = calculate_generation_fitness(individuals_neural_network)
        """ordenamos los valores con menos error descendente
        se tiene que buscar los dos que tienen el menor fit con otra funcion
        hacemos un random para saber cual coger si el primero o el segundo mejor de generacion"""
        all_fitness.sort()
        best_fitness_record.append(all_fitness[0])
        best_fitness, second_best_fitness = get_index_best_network(individuals_neural_network, all_fitness)
        
        # a partir del mejor o el segundo mejor fitnes creamos las nuevas generaciones mutando el codigo
        best_old = different_neural_networks[best_fitness]
        second_old = different_neural_networks[second_best_fitness]
        for generations in range(individuals_neural_network):
            different_neural_networks[generations].create_generation(best_old, second_old)
        
    
    # una vez finalizadas las generaciones, miramos la salida de el input no entrenado y comprovamos que sea igual al que queriamos
    print("New situation: input data = ", INPUT_ONES_TRAINED)
    output = int(np.around(different_neural_networks[best_fitness].check_trained_algorithm(INPUT_ONES_TRAINED)).astype(int))
    if (output == EXPECTED_FINAL_OUTPUT ):
        print("Output data: ", output, "OK")
    else: 
        print("Output data: ", output, "KO")
    # Hacemos un plot de todos los mejores fitness de cada generacion
    s.plot_error(best_fitness_record)
    
    
   