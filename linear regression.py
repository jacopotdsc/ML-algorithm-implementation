from array import array
from math import hypot
from operator import truediv
from turtle import hideturtle, update
import pandas as pd
from sklearn.datasets import load_iris
from time import sleep
                            ###########################
                            # Tedeschi Jacopo 1882789 #
                            ###########################


data = load_iris()  # dizionario con chiave { 'data', 'target'}

DEBUG = False
TEST_DEBUG = False
LOSS_DEBUG = False
INITIAL_LEARNING_RATE = 0.01
INITIAL_SEPAL_LENGTH = 0.5 
INITIAL_SEPAL_WIDTH  = 0.6 
INITIAL_PETAL_LENGTH = 0.1 
INITIAL_PETAL_WIDTH  = 0.8 
ACCEPTABLE_VALUE_LOSS_CONVEGENCE = 0.08  # valore per cui si può considerare la convergenza dell'update rule
SLEEP_TIME = 0.0    # sleep per vedere come variano alcuni parametri durante il ciclo, per debug

actual_learning_rate = INITIAL_LEARNING_RATE

class learn_rate:   # classe per far cambiare il learning rate
    def __init__(self):
        self.learn_rate = INITIAL_LEARNING_RATE
    
    def update_learning_rate(self):
        if self.learn_rate > 0.005: self.learn_rate = self.learn_rate / 2
        if self.learn_rate < 0.005: self.learn_rate = 0.005
    
    def get_learn_rate(self):
        return self.learn_rate

def initialize_hypotesis(): # ritorna il dizionario con i valore dei pesi
    features  = data.feature_names    # colonne del dataset
    hypotesis = {}

    for x in features:
        if x == "sepal length (cm)": hypotesis[x] = INITIAL_SEPAL_LENGTH
        elif x == "sepal width (cm)": hypotesis[x] = INITIAL_SEPAL_WIDTH
        elif x == "petal length (cm)": hypotesis[x] = INITIAL_PETAL_LENGTH
        elif x == "petal width (cm)": hypotesis[x] = INITIAL_PETAL_WIDTH

    hypotesis['bias'] = 1
    return hypotesis

def learning_rate():    # TO - DO
    return INITIAL_LEARNING_RATE

def my_predict( hypotesis, input ): # calcola il valore predetto dal modello
    sum = 0

    keys = list( hypotesis.keys() ) # lista di valori delle colonne

    for index in range( len ( keys ) ):
        if keys[index] == "bias": sum += hypotesis['bias']
        else: sum += hypotesis[keys[index]]*input[index]

    return sum

def get_index_of_column( column ): # mi restituisce l'indice di cui si trova la stringa "column"
    
    keys = list( data.feature_names )
    index = 0

    for k in keys:
        if k == column:
            return index
        index += 1
    return index

def loss_derivate(hypotesis, column):   # funzione MSE di loss
    loss_value = 0

    index = 0
    while index < 150 : 
        correct_outuput = data['target'][index]  # valore di output corretto
        input = data['data'][index]  # valori di input per il mio modello

        calculated_value = correct_outuput - my_predict(hypotesis, input)

        x_ji = get_index_of_column(column)

        if column == 'bias': loss_value += ( calculated_value ) / 150
        else: loss_value += ( calculated_value * input[x_ji] ) / 150

        index += 1  

    # valore della derivata della loss
    value_loss_derivate = loss_value
    return value_loss_derivate

def is_converged(loss_value): # controlla la convergenza quando utilizzo le update rules
    if DEBUG == True:   print("is_converged -> loss: " + str(loss_value))
    if abs( loss_value ) < ACCEPTABLE_VALUE_LOSS_CONVEGENCE: return True
    else: return False

def loss(hypotesis):
    loss_value = 0
    index = 0
    while index < 150: 
        input = data['data'][index]
        correct_value = data['target'][index]

        term = ( correct_value - my_predict(hypotesis, input ) )

        if LOSS_DEBUG == True: print("term: {}, correct_value: {}, predict: {}".format(term, correct_value, my_predict(hypotesis, input ) ))
        loss_value +=  (term*term) / 150

        index += 1
    return loss_value

def linear_regression():    # ritorna il modello della regressione lineare

    hypotesis = initialize_hypotesis()

    print("inizial hypotesis:" )
    print(hypotesis)
    print()

    while is_converged( loss( hypotesis ) ) == False:

        if DEBUG == True:
            print("\n-----------------CHIAMATA-----------------")
            print("actual hypotesis")
            print(hypotesis)

        for wi in hypotesis.keys():
            #update rule
            derivate_loss_value = loss_derivate(hypotesis,wi)
            if DEBUG == True:
                print("*******\n")
                print("features: {},\thypotesis[wi]: {},\t LR: {},\t loss: {}, \t derivate_loss: {}\n".format(wi,hypotesis[wi], learning_rate(), loss( hypotesis ), derivate_loss_value))
            
            # update rule
            hypotesis[wi] = hypotesis[wi] + learning_rate()*loss_derivate(hypotesis, wi)
            
            if DEBUG == True:   print("features: {}, new value: {}\n".format(wi, hypotesis[wi]))

        sleep(SLEEP_TIME)
    return hypotesis

def main():
    if DEBUG == True: 
        print( " stampo il mio dataset\n")
        #my_print()
        print( "\nfinito ")

    hypotesis = linear_regression()

    print("\nFINAL HYPOTESIS:")
    print(hypotesis)
    print()


def my_print():
    print("dataset: " + str(len(data['data'])))
    print("features_names: " + str(data.feature_names))
    print("target_names:" + str(data.target_names))
    #print(data['data'])

    print("\ntarget: " + str(len(data['target'])))
    print(data['target'])

main()


                            ###########################
                            # Tedeschi Jacopo 1882789 #
                            ###########################