import matplotlib.pyplot as plt


class StyleGrupo08():
    # constantes de printar para el usuario
    FIRSTNEURONLAYER = "¿Cuantas neuronas quieres que tenga la primera capa de la red neuronal?"
    SECONDNEURONLAYER = "¿Cuantas neuronas quieres que tenga la segunda capa de la red neuronal?"
    INDIVIDUALS = "¿Cuantos individuos quieres poner en cada una de las generaciones?"
    GENERATIONS = "¿Cuantas generaciones quizeres iterar para la solución actual?"
    # mensaje de bienvenida
    def __init__ (self):
        print (f"""Bienvenido al algoritmo neuronal del grupo 08 !!\n\nNuestro algoritmo es el siguiente:\n
            A  B  C  | Out
            ---------|---
            0  0  0  |  1 
            0  0  1  |  0
            0  1  0  |  1
            0  1  1  |  0
            1  0  0  |  1
            1  0  1  |  0
            1  1  0  |  1
            1  1  1  |  0
            """)
    # comprovamos que el imput introducido sea entero y positivo 
    def ask_user(self, text):
        while True:
            try:
                number = int(input(text))
            except ValueError:
                print("No se ha introducido un numero ... vuelve a intentar: ")
            else: 
                if number > 1:
                   return number
                else:
                    print("Se ha de introducir un numero mayor a 1 ... vuelve a intentar:")


    def plot_error(self, sum_error_fitness):
        # se muestran los errores en las diferentes iteraciones de la mejor salida
        # en un plot
        plt.plot(sum_error_fitness)
        plt.xlabel("Iteration")
        plt.ylabel("Suma de Error en la tabla")
        plt.title("Curva de error durante el entrenamiento")
        plt.show()
