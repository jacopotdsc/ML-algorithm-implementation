from array import array
from lib2to3.pygram import pattern_grammar
from math import hypot
from operator import truediv
from turtle import hideturtle, update
import math as m
from wsgiref.validate import PartialIteratorWrapper
import pandas as pd
from sklearn.datasets import load_iris
from time import sleep
                            ###########################
                            # Tedeschi Jacopo 1882789 #
                            ###########################


data = load_iris()  # dizionario con chiave { 'data', 'target'}

####    MODIFICARE QUESTO PARAMETRO PER PROVARE A CLASSIFICARE LE ALTRE SPECIE
TO_PREDICT = 2 # sostituire con { 0, 1, 2 }, mi dice quale specie il mio predittore deve sapere identificare
###################

DEBUG = False
LOSS_DERIVATE_DEBUG = False
LOSS_DEBUG = False
TEST_DEBUG = False
PRINT_PREDICT = False
INITIAL_LEARNING_RATE = 0.005
INITIAL_SEPAL_LENGTH = 0.05 
INITIAL_SEPAL_WIDTH  = 0.06 
INITIAL_PETAL_LENGTH = 0.01 
INITIAL_PETAL_WIDTH  = 0.08 
ACCEPTABLE_VALUE_LOSS_CONVERGENCE = 10  # valore per cui si può considerare la convergenza dell'update rule
THRESHOLD_VALUE = 0.85 # valore di threshold per il predittore
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
    return 0.005


def my_predict( hypotesis, input ): # calcola il valore predetto dal modello
    sum = 0

    keys = list( hypotesis.keys() ) # lista di valori delle colonne

    # calcolo il termine w * x
    for index in range( len ( keys ) ):
        if keys[index] == "bias": sum += hypotesis['bias']
        else: sum += hypotesis[keys[index]]*input[index]
        
        if PRINT_PREDICT == True: print("keys: {}, hypotesis[wi]: {}, input: {}".format(keys[index], hypotesis[keys[index]], input))

    denominatore =  1 + m.exp(-sum) 
    #print("input: {}, sum_value: {}, denominatore: {}".format(input, sum, denominatore))
    if denominatore == 0: predict_value = 1
    else: predict_value = 1 / ( 1 + m.exp(-sum) )
    
    if PRINT_PREDICT == True: print("threshold: {}, predict_value: {}, sum: {}, 1+e^-sum: {}\n".format( THRESHOLD_VALUE, predict_value, sum, 1 + m.exp(-sum)  ))

    return predict_value
    #if predict_value >= THRESHOLD_VALUE: return 1
    #else: return 0

def calculate_correct_output( output ): # converte per avere outupt "corretto = 1", " non corretto = -1"
    if output == TO_PREDICT : return 1
    else: return -1

def get_index_of_column( column ): # mi restituisce l'indice di cui si trova la stringa "column"
    
    keys = list( data.feature_names )
    index = 0

    for k in keys:
        if k == column:
            return index
        index += 1
    return index

def loss_derivate(hypotesis, column):
    loss_value = 0
    index = 0
    while index < 150: 
        input = data['data'][index]
        correct_value = data['target'][index]
        modified_correct_value = calculate_correct_output(correct_value)

        predict = my_predict(hypotesis, input )

        if LOSS_DERIVATE_DEBUG == True: print("correct_value: {}, predict: {}".format(modified_correct_value, predict ))
        
        x_ji = get_index_of_column(column)

        if column == 'bias': loss_value +=  ( predict - modified_correct_value )
        else: loss_value +=  ( predict - modified_correct_value )*input[x_ji]

        index += 1
    return loss_value

def is_converged(loss_value): # controlla la convergenza quando utilizzo le update rules
    if DEBUG == True:   print("is_converged -> loss: " + str(loss_value))
    if abs( loss_value ) < ACCEPTABLE_VALUE_LOSS_CONVERGENCE: return True
    else: return False

def loss(hypotesis):   # funzione Binary Cross Entropy di loss
    loss_value = 0

    index = 0
    while index < 150 : 
        correct_outuput = data['target'][index]  # valore di output corretto
        input = data['data'][index]  # valori di input per il mio modello

        modified_correct_output = calculate_correct_output(correct_outuput)
        predict = my_predict(hypotesis, input)

        if predict == 1: calculated_value = - modified_correct_output*m.log(predict)
        elif predict == 0: calculated_value = - (1 - modified_correct_output)*m.log(1- predict) 
        else: calculated_value =  - ( modified_correct_output*m.log(predict) + (1 - modified_correct_output)*m.log(1- predict)  )
        
        if LOSS_DEBUG == True: print("my_predict: {}, targtet_output: {}, modified_output: {}, calculated_Value: {}".format(my_predict(hypotesis, input),correct_outuput, modified_correct_output, calculated_value))

        loss_value += calculated_value

        index += 1  

    if LOSS_DEBUG == True: print("loss_value: {}\n".format(loss_value))
    return loss_value

def logistic_regression():    # ritorna il modello della regressione logistica

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

    hypotesis = logistic_regression()

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