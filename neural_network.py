from array import array
from imp import source_from_cache
from multiprocessing.spawn import prepare
import pandas as pd
from sklearn.datasets import load_iris
import math as m
from time import sleep
#import matplotlib as plt
                            ###########################
                            # Tedeschi Jacopo 1882789 #
                            ###########################

DEBUG = False
PRINT_NN = False
PRINT_WITH_ID = False  
DEBUG_WEIGTH_CHANGE = False
LEARNING_RATIO = 0.01
INITAL_WEIGTH_VALUE = 0.2
BATCH_SIZE= 10
MAX_EPOCH = 100
INPUT_PERCEPTRON_STR  = 'input perceptron'
HIDDEN_PERCEPTRON_STR = 'hidden perceptron'
OUTPUT_PERCEPTRON_STR = 'output perceptron'
   

dataset = load_iris()       # dizionario con chiave { 'data', 'target' }
data = dataset['data']      # dataset con i dati di training
target = dataset['target']  # dataset con i target obbiettivo

n_neurons = [10, 20, 10]    # numero neuroni negli hidden layer
n_hidden_layer = len( n_neurons )   # numero degli hidden  lyaer
n_input_neurons = len( data[0] )    # numero di neuroni per l'input, 1 per ogni feature
n_output_neurons = 1                # numeri di neuroni per l'output: per regressione è 1


#### fai la classe vettore per gli elementi del dizionario che si trasformerò in un array di vettori

def reLU(x):    # funzione di attivazione per il percettrone
    if x < 0: return 0
    #elif x > 1: return 1
    else: return x

def reLU_derivate(x):
    return 1

def sigmoid(x):
    denominatore =  1 + m.exp(x) 

    if denominatore == 0: predict_value = 1
    else: predict_value = 1 / ( 1 + m.exp(x) ) 

    return predict_value

def sigmoid_derivate(x):
    return sigmoid(x)*(1-sigmoid(x))

def activation_function(x): # funzione wrapper 
    return reLU(x)

def activation_fucntion_derivate(x):   # funzione wrapper 
    return reLU_derivate(x)

# vettore che collega i neuroni: lettura source-dest da sinistra a destra
class Vector:
    def __init__(self, lower_perceptron = 0, upper_perceptron = 0, weigth = INITAL_WEIGTH_VALUE):
        self.upper_perceptron = upper_perceptron    # perceptron del livello più alto
        self.lower_perceptron = lower_perceptron    # perceptron del livello più basso
        self.weigth = weigth
 
    def get_upper_perceptron(self):
        return self.upper_perceptron
    
    def get_lower_perceptron(self):
        return self.lower_perceptron
    
    def get_weigth(self):
        return self.weigth

    def set_weigth(self, weigth):
        self.weigth = weigth

    def __str__(self):
        return "{} -> {}, weigth: {}".format(self.lower_perceptron.id_perceptron, self.upper_perceptron.id_perceptron, self.weigth)
    
    def __repr__(self):
        return "{} -> {}: weigth: {}".format(self.lower_perceptron.id_perceptron, self.upper_perceptron.id_perceptron, self.weigth)

class Perceptron:
    def __init__(self, id_perceptron, layer_number, back_perceptron = None , forward_perceptron = None, perceptron_value = 0):

        # calcolo del tipo del perceptron
        if layer_number == 0:                        self.type = INPUT_PERCEPTRON_STR
        elif layer_number <= ( n_hidden_layer + 1 ): self.type = HIDDEN_PERCEPTRON_STR
        else:                                        self.type = OUTPUT_PERCEPTRON_STR

        # crea un id univoco
        self.id_perceptron = id_perceptron 

        # mi dice a che livello della rete sta
        self.layer_number = layer_number       

        # si riferisce ai neuroni dello strato precedente
        # array di arraty di vettori del tipo [v11, ..., v1n]
        self.back_perceptron = []

        # neuroni nello strato successivo
        # array di arraty di vettori del tipo [v11, ..., v1n]
        self.forward_perceptron = [] 

        # è il valore del percettrone aj = f( sum( w*ai ) ) ), con i < j
        self.perceptron_value = perceptron_value    

        self.delta = 0  # variabile per back-propagation

    def set_delta(self, d):
        self.delta = d
    
    def get_delta(self):
        return self.delta

    def add_back_perceptron(self,vector):   
        #print("in add_back_perceptron: {}".format(type(vector)))
        self.back_perceptron.append(vector)
    
    def add_list_back_perceptron(self, list):
        self.back_perceptron = list

    def get_list_back_perceptron(self): 
        return self.back_perceptron

    def add_forward_perceptron(self, vector):
        self.forward_perceptron.append(vector)
    
    def add_list_forward_perceptron(self, list):
        self.forward_perceptron = list

    def get_list_forward_perceptron(self): 
        return self.forward_perceptron

    def set_perceptron_value(self, perceptron_value):
        self.perceptron_value = perceptron_value
    
    def get_perceptron_value(self): 
        return self.perceptron_value

    def set_weigth(self, new_weigth, id_to_change):
        for v in self.forward_perceptron:
            if v.get_source().id_perceptron == id_to_change:
                v.set_weigth(new_weigth)
    
    # restituisce il vettore che ha come upper_perceptron il neurone con goal_id dai forward vector
    def get_forward_vector(self, source_p, goal_p):
        #print("source: {}, goal: {}".format(source_p.id_perceptron,goal_p.id_perceptron))
        #print(self.get_list_forward_perceptron())
        for v in self.get_list_forward_perceptron():
            if v.get_upper_perceptron().id_perceptron == goal_p.id_perceptron and v.get_lower_perceptron().id_perceptron == source_p.id_perceptron:
                #print(v)
                return v
                

    def weigth_str( self, list_p ):
        SLEEP = 0
        SLEEP_DEBUG = False
        DEEP_DEBUG = False

        if list_p == []:
            return "id: {}, layer: {}, value: {} ".format(self.id_perceptron, self.layer_number, self.perceptron_value)

        weigth_str = ''

        if SLEEP_DEBUG == True:
            #print("lista: {}\n".format(list_p))
            sleep(SLEEP)

        counter = 0
        for index in range(len(list_p)):
            v = list_p[index]
            if SLEEP_DEBUG == True:
                print("i: {} -> vector in list: {}".format(index, v))
            
            if DEEP_DEBUG == True: print("*** inizializzo la stringa: {} ***".format(counter))
            #my_string = "{}".format( v.get_upper_perceptron().id_perceptron )
            my_string = "{}".format(v.get_weigth())


            if DEEP_DEBUG == True: print("*** stampo counter *** ")
            if counter == 0:
                if DEEP_DEBUG == True: print("{} percettrone".format(counter))
                weigth_str +=  '[ ' + my_string
            elif counter == len(list_p) - 1:
                if DEEP_DEBUG == True: print("{} percettrone".format(counter))
                weigth_str += ', ' + my_string 
            else: 
                if DEEP_DEBUG == True: print("{} percettrone".format(counter))
                weigth_str += ', ' + my_string

            counter += 1
        weigth_str += ' ]'

        if SLEEP_DEBUG == True: print("weigth: {}".format(weigth_str))
        return weigth_str
    
    def __str__(self):
        STR_DEBUG = False

        string_to_print = "id: {}, layer: {}, value: {}".format(self.id_perceptron, self.layer_number, self.perceptron_value)

        if self.back_perceptron != []:
            if STR_DEBUG == True:
                print(string_to_print)
                print("id: {}, b_list: {}".format(self.id_perceptron, self.back_perceptron))

            string_to_print += "\n\tback_weigth:    {}".format( self.weigth_str(self.back_perceptron) )

        if self.forward_perceptron != []:
            if STR_DEBUG == True:
                print(string_to_print)
                print("id: {}, f_list: {}".format(self.id_perceptron, self.forward_perceptron))
            string_to_print += "\n\tforward_weigth: {}".format(self.weigth_str(self.forward_perceptron))

        if STR_DEBUG == True:
            print("return string: {}".format(string_to_print))

        return string_to_print
    
    def __repr__(self):
        return self.id_perceptron

class NeuralNetwork:
    def __init__(self, total_layer, neural_netowrk = []):
        self.total_layer = total_layer

        # mi restituisce un array devo l'indice è l'i-esimo livell
        #self.array_with_levels = self.prepare_array_with_layer()  
                                 
        # mi unisce i livelli ed i percettroni correttamente 
        self.neural_network = neural_netowrk

        self.create_neural_network() #( self.array_with_levels )


    '''  funzioni per il costruttore '''
    # mi crea un id da assegnare ai percettroni: sarà usare come chiave nel dizionario per la rete neurale
    def create_id(self, level, n_of_layer):
        return "{}-{}".format(level,n_of_layer)

    # mi restituisce un array devo l'indice è l'i-esimo livello
    def prepare_array_with_layer( self ):
    
        array_of_level = [] # array in cui ogni indice contiene un array di vettori <source, dest, weigth >

        for level in range( self.total_layer ):
        
            my_array = []

            # calcolo il layer di input
            if level == 0:
                perceptron_counter = 0  # conta il numero di percerptron nel layer
                for p in range(n_input_neurons):

                    # creo un id con il numero del livello e il numero del percettrone con formato l-n
                    id = self.create_id(level, perceptron_counter)

                    # self, id_perceptron, layer_number, back_perceptron, forward_perceptron, perceptron_value = 0
                    my_perceptron = Perceptron( id, level)

                    my_array.append( Perceptron( id, level) )
                    
                    perceptron_counter += 1

            # calcolo gli hidden layer
            elif level != 0 and level < self.total_layer - 1:
            
                perceptron_counter = 0
                for p in range(n_neurons[level - 1]):   # il -1 serve per aggiustare l'indice nell'array che contine le info sugli hidden layer

                    id = self.create_id(level,perceptron_counter)

                    # self, id_perceptron, layer_number, back_perceptron, forward_perceptron, perceptron_value = 0
                    # self, id_perceptron, layer_number, back_perceptron, forward_perceptron, perceptron_value = 0
                    list_f = []
                    list_b = []
                    my_perceptron = Perceptron( id, level, list_f.copy(), list_b.copy())

                    my_array.append( my_perceptron )

                    perceptron_counter += 1

            # calcolo output layer
            else:
                perceptron_counter = 0
                for p in range(n_output_neurons):  

                    id = self.create_id(level,perceptron_counter)

                    # self, id_perceptron, layer_number, back_perceptron, forward_perceptron, perceptron_value = 0
                    # self, id_perceptron, layer_number, back_perceptron, forward_perceptron, perceptron_value = 0
                    my_perceptron = Perceptron( id, level)

                    my_array.append( my_perceptron )

                    perceptron_counter += 1

            array_of_level.append( my_array )

        #return array_of_level
        self.neural_network = array_of_level

    # mi unisce i vari dizionari per legare i vari layer e percettroni
    def unify_level( self, array ):
        UNIFY_DEBUG = False
        UNIFY_DEBUG_VECTOR_B = False
        UNIFY_DEBUG_VECTOR_F = False
        UNIFY_DEBUG_BACK_PERCEPTRON = False
        PRINT_F_VECTOR = False
        BREAK = False

        level = 0
        if UNIFY_DEBUG == True: print(array)
        for level in range( len(array) ):
            
            layer = array[level]

            if UNIFY_DEBUG == True:
                print()
                print("LIVELLO: {}, id array: {}".format(level,id(array[level])))
                print()
                print(array[level])

            if level == 0:  # layer di input
                upper_layer = array[level + 1]

                if UNIFY_DEBUG == True: print(upper_layer)

                for p in layer:                     # per ogni neurone
                    for upper_p in upper_layer:     # collego tutti i neuroni a lv superiore
                        # creo il vettore, gli settore il lower_perceptron e dopo lo attacco al perceptron
                        my_vector = Vector(p, upper_p)

                        if UNIFY_DEBUG_VECTOR_F == True: print("vector: {}".format(my_vector))

                        p.add_forward_perceptron( my_vector )

                    if PRINT_F_VECTOR == True: 
                        print("\nid: {}-{}, forward_list_id: {}".format(p.id_perceptron,id(p), id(p.get_list_forward_perceptron())))
                        print( p.get_list_forward_perceptron())
                        # problema: la lista dei forward vector è unica

            elif level != 0 and level < len(array) - 1: # hidden layer
                upper_layer = array[level + 1]
                lower_layer = array[level - 1]
                
                if UNIFY_DEBUG == True:
                    print(lower_layer)
                    print(upper_layer)

                for p in layer:                     # per ogni neurone
                    for upper_p in upper_layer:     # collego tutti i neuroni a lv superiore
                        # creo il vettore, gli settore il lower_perceptron e dopo lo attacco al perceptron
                        my_vector = Vector(p, upper_p)
                        if UNIFY_DEBUG_VECTOR_F == True: print("vector: {}".format(my_vector))
                        p.add_forward_perceptron( my_vector )

                    if UNIFY_DEBUG == True: 
                        print("collego i back per: {}".format(p.id_perceptron))

                    # colllego i vettori back
                    for lower_p in lower_layer:
                        add_vector = lower_p.get_forward_vector(lower_p, p)
                        if UNIFY_DEBUG_VECTOR_B == True: print("l_p: {}, p: {}, vector: {}".format(lower_p.id_perceptron, p.id_perceptron, add_vector))
                        p.add_back_perceptron( add_vector )
                    
                    if UNIFY_DEBUG_BACK_PERCEPTRON == True:
                        print("\nlista dei back perceptron, list_id: {}".format(p.id_perceptron, id(p.get_list_back_perceptron())))
                        print(p.get_list_back_perceptron())
                    
                    if BREAK == True: break
        
            else:
                for p in layer: # layer di output
                    lower_layer = array[level - 1]
                    if UNIFY_DEBUG == True: print(lower_layer)

                    # colllego i vettori back
                    for lower_p in lower_layer:
                        add_vector = lower_p.get_forward_vector(lower_p, p)
                        if UNIFY_DEBUG_VECTOR_B == True: print("l_p: {}, p: {}, vector: {}".format(lower_p.id_perceptron, p.id_perceptron, add_vector))
                        p.add_back_perceptron( add_vector )

                    if UNIFY_DEBUG_BACK_PERCEPTRON == True:
                        print("\nlista dei back perceptron, list_id: {}".format(p.id_perceptron, id(p.get_list_back_perceptron())))
                        print(p.get_list_back_perceptron())

    
    # crea la rete neurale
    def create_neural_network(self):
        #my_array = self.prepare_array_with_layer()  # mi prepara l'array che ha solo gli strati senza collegamenti
        #self.neural_network = self.unify_level( my_array ) # collega tutti i nodi
        self.prepare_array_with_layer()
        self.unify_level( self.neural_network )

    '''______________________________'''

    # cambia il peso di un neurone nella rete neurale
    def set_weigth(self, lower_id, upper_id, new_weigth):

        # scansiono tutti i livelli per trovare quello giusto
        for level in range( len(self.neural_network) - 1):  # non considero quello di output

            layer = self.neural_network[level]

            for p in layer:
                
                 for vector in p.get_list_forward_perceptron():
                    
                    if vector.get_lower_perceptron().id_perceptron == lower_id and vector.get_upper_perceptron().id_perceptron == upper_id:
                        vector.set_weigth(new_weigth)
                        return

    # mi calcola il valore predetto
    def my_predict(self, input):
        PREDICT_DEBUG = False
        if PREDICT_DEBUG == True: print("\n predicting: {}".format(input))

        predict_value = 0

        for level in range( len(self.neural_network)):

            if level == 0:      # metto i valori di input nel layer di input
                input_layer = self.neural_network[0]
                input_index = 0

                for p in input_layer:
                    p.set_perceptron_value(input[input_index])
                    input_index += 1
                    
                    if PREDICT_DEBUG == True: print("id: {} -> value: {} ".format(p.id_perceptron, p.perceptron_value))

            else:        
                for p in self.neural_network[level]:    # per tutti i neuroni nel layer
                    
                    sum = 0.0

                    for back_p_vector in p.get_list_back_perceptron(): # prendo i neuroni che sono di input per p

                        weigth = back_p_vector.get_weigth()     
                        sum += float(weigth) * float(back_p_vector.get_lower_perceptron().perceptron_value)    # prendo il valore del neurone di input
                    
                    value_ij = activation_function(sum)
                    p.set_perceptron_value(value_ij)    # imposto il valore del neurone

                    if PREDICT_DEBUG == True: print("id: {}, sum: {}, value_ij: {} -> value: {} ".format(p.id_perceptron, sum , value_ij, p.perceptron_value))


        # restituisco l'output della rete neurale: prendo il massimo valore tra i perceptron di output
        output_layer = self.neural_network[self.total_layer - 1] # layer di output

        if PREDICT_DEBUG == True: print(output_layer)

        max = 0
        for output_perceptron in output_layer:  
            if PREDICT_DEBUG == True: print("output_perceptron: {}".format(output_perceptron))
            if output_perceptron.perceptron_value > max:
                max = output_perceptron.perceptron_value

        return max

    def print_perceptron(self, id_to_print):

        for level in range( len(self.neural_network) ): 

            layer = self.neural_network[level]

            for p in layer:
                
                 if p.id_perceptron == id_to_print:
                     print(p)

    def print_neural_network(self):
        PRINT_NN_DEBUG = False

        level = 0
        for layer in range( self.total_layer ):
            
            #perceptron_number = len( self.neural_network[layer] )
        
            if level == 0: print( "\nlevel: {} ->input layer, n_neurons: {}".format(level, len(self.neural_network[layer])))
            elif level > 0 and level < self.total_layer - 1: print( "\nlevel: {} ->hidden layer, n_neurons: {}".format(level, len(self.neural_network[layer-1])))
            else: print( "\nlevel: {} ->output layer, n_neurons: {}".format(level, len(self.neural_network[layer])))

            #print(self.neural_network)
            for p in self.neural_network[layer]:
                if PRINT_NN_DEBUG == True:
                    print("---PERCEPTRON---")
                    print("layer: {}\np_id: {}\n".format(self.neural_network[layer],p.id_perceptron))
                print(p)

            level += 1

    def calculate_ground_truth(self, input):    # TO - DO
        return 1

    def loss(self, input_data): # funzione MSE di loss
        loss_value = 0
        for output_perceptron in self.neural_network[self.total_layer - 1]: # per tutti i neuroni di output
            for i in input_data:
                value = self.calculate_ground_truth(i) - self.my_predict(i)
                loss_value += value*value

        return 0.5*loss_value   
    
    def loss_derivate_ij(self, input_data):
        delta_value = self.delta_i(input_data) 

        # la derivata dell'errore_p rispetto al peso w_ij e data da delta_i*rigth_value
        return delta_value* self.calculate_ground_truth(input_data)
    

    def get_subset(set, iteration): # mi ritorna un subset del del set in input
        low_limit = iteration * BATCH_SIZE
        upper_limit = iteration * BATCH_SIZE

        return set.loc[ set['Rownumber'] >= low_limit and set['Rownumber'] < upper_limit]

    def my_fit(self, training_set):    # input: set di training
        FIT_DEBUG = True
        SLEEP = 2

        # da fare con le epoche ( quindi SGD )
        epoch = 0
        my_training_set = training_set.copy()   # faccio la copia per non alteraro l'originale


        while self.loss(my_training_set) > 0.5 and epoch < MAX_EPOCH:
            
            if FIT_DEBUG == True: 
                print("epoch: {}, loss: {}".format(epoch, self.loss(my_training_set) ))
                #sleep(SLEEP)


            epoch += 1
            if epoch >= MAX_EPOCH: break

            for input in my_training_set:
                predict_value = self.my_predict(input)
                rigth_value = self.calculate_ground_truth(input)

                #if FIT_DEBUG == True: print("   predict_value: {}, right_value: {}".format(predict_value, rigth_value))

                if predict_value - rigth_value == 0:    # se è stato predetto correttamente, continua il for
                    continue
                else:
                    self.back_propagation(rigth_value, predict_value)

        '''
        while epoch < MAX_EPOCH:
            
            for i in range( (len(my_training_set)/BATCH_SIZE) - 1 ) :

                sub_set = self.get_subset(my_training_set, i)

                for input in sub_set:
                    predict_value = self.my_predict(input)
                    rigth_value = self.calculate_ground_truth(input)

                    if FIT_DEBUG == True:
                        print("   predict_value: {}, right_value: {}".format(predict_value, rigth_value))

                    if predict_value - rigth_value == 0:    # se è stato predetto correttamente, continua il for
                        continue
                    else:
                        self.back_propagation(rigth_value, predict_value)

            epoch += 1
            my_training_set = my_training_set.sample(frac=1)    # mischio il training set
            '''
            
            
    def back_propagation(self, right_value, predict_value):
        DEBUG_BACK_PROPAGATION = True
        BACK_SLEEP = 2
        level = self.total_layer - 1    # livello di output

        while level > 0:
        
            for p in self.neural_network[level]:    # su tutti i percettroni, calcolo il delta
                back_vector = p.get_list_back_perceptron()

                if DEBUG_BACK_PROPAGATION == True: print("prima: {}".format(p))

                for v in back_vector:
                    delta_value = self.delta_i(p, right_value, predict_value)
                    new_weigth = v.get_weigth() - delta_value
                    v.set_weigth(new_weigth)
                
                if DEBUG_BACK_PROPAGATION == True:  print("  new_weigth: {}, delta: {}\ndopo: {}\n".format(new_weigth, delta_value, p))
                if DEBUG_BACK_PROPAGATION == True:  sleep(BACK_SLEEP)

            level -= 1

    def delta_i(self, perceptron, right_value, predict_value):
        DELTA_DEBUG = False
        delta_value = 0

        if perceptron.layer_number == self.total_layer - 1: # livello di output
            delta_value = - ( right_value - predict_value )*activation_fucntion_derivate(predict_value)
            if DELTA_DEBUG == True: print("perceptron: {}, delta: {}".format(perceptron.id_perceptron, delta_value))
        else:
            sum = 0.0

            for v in perceptron.get_list_forward_perceptron():
                weigth = v.get_weigth()
                delta_k = v.get_upper_perceptron().get_delta()
                sum += float(weigth) * float(delta_k)

                if DELTA_DEBUG == True: print("upper_p: {}, delta: {}".format(v.get_upper_perceptron().id_perceptron, delta_k))

            delta_value = sum * activation_fucntion_derivate(predict_value)

        perceptron.set_delta(delta_value)
        if DELTA_DEBUG == True: print("sum: {}, act_func_val: {}, p_id: {}, p_delta_Val: {}".format(delta_value, activation_fucntion_derivate(predict_value), perceptron.id_perceptron, perceptron.get_delta()))
        return delta_value

def main():
    
    print("Inizializzazione della rete neurale")

    nn = NeuralNetwork(1 + n_hidden_layer + 1)

    print("finito")

    if PRINT_NN == True: nn.print_neural_network()

    #print( nn.my_predict([4.3, 1.7, 6.3, 2.4]) )

    if DEBUG_WEIGTH_CHANGE == True:
        perceptron_id = "2-13"
        nn.set_weigth("1-9", perceptron_id,1000)
        nn.print_perceptron(perceptron_id)
        nn.print_perceptron("1-9")
        nn.print_perceptron("1-8")
        nn.print_perceptron("1-7")
        nn.print_perceptron("1-6")
        nn.print_perceptron("1-5")
        nn.print_perceptron("1-4")
        nn.print_perceptron("1-3")
        nn.print_perceptron("1-2")
        nn.print_perceptron("1-1")
        nn.print_perceptron("1-0")

    print("predict senza addestramento: {}".format(nn.my_predict([5.7, 2.6, 3.5, 1.0])) )

    nn.my_fit(data)

    print("predict con addestramento: {}".format(nn.my_predict([5.7, 2.6, 3.5, 1.0])) )

main()

                            ###########################
                            # Tedeschi Jacopo 1882789 #
                            ###########################
