from cProfile import label
from inspect import Attribute
from re import sub
from sre_constants import CATEGORY_UNI_NOT_LINEBREAK
import pandas as pd
import math 

                            ###########################
                            # Tedeschi Jacopo 1882789 #
                            ###########################


data = pd.read_csv("restaurant_waiting.csv",delimiter=',')

# numero di Yes e No  nel dataset
query_dataset_yes   = len( data.loc[data['Wait'] == 'Yes']['Wait'] )
query_dataset_no    = len( data.loc[data['Wait'] == 'No']['Wait'] )
const_den = query_dataset_yes + query_dataset_no

debug = False

class decision_tree_node:
    def __init__(self, examples, father, soons, list_attributes, label = ''):
        self.examples        = examples         # dataset del nodo
        self.father          = father           # nodo padre
        self.soons           = soons            # lista dei figli -> dizionario
        self.list_attributes = list_attributes  # lista degli attributi rimanenti
        self.label           = label            # etichetta del sotto-dataset

    def add_soon(self, label, info):   # aggiunge un figlio nel dizionario soons
        self.soons.update({label:info}) 


def importance(attributes, examples): # trova l'attributo più importante

    importance = 0
    att_max = 0
    for a in attributes:    # ciclo su tutti gli attributi

        if debug == True: print("attributo: " + str(a))

        actual_gain = gain(a,examples)  # calcolo il guadagno dell'attributo

        if debug == True: print("\t-- gain: " + str(actual_gain))

        if actual_gain > importance :    # trovo quello con guadagno d'entropia maggiore
            importance = actual_gain
            att_max = a
        
        if debug == True: print("\t-- gain maggiore: " + str(att_max) + " -> " + str(importance))
    return att_max

def B(p, n):   # calcola il valore dell'entropia

    q = p / ( p + n)

    if q == 1:
        value = 0
    elif q == 0:
        value = 0
    else:
        value = - ( q*math.log2(q)+(1-q)*math.log2(1-q) )
    
    return value

def gain(attribute, examples):  # calcola il gain, guadagno di entropia

    # numeri si Yes e No per calcolare B della partizione
    p = len( examples.loc[examples['Wait'] == 'Yes']['Wait'] )
    n  = len( examples.loc[examples['Wait'] == 'No']['Wait'] )

    entropia = B( p, n)

    if debug == True: print("\t-- entropia di " + str(attribute) + ": " + str(entropia))

    # gain(A) = B(q) - remainder(A)
    return entropia - remainder(attribute,examples)

def remainder(attribute,examples): # funzione remainder

    value_list = set(examples[attribute].tolist()) # insieme dei valori dell'attributo
    
    actual_remainder= 0

    # numeri si Yes e No per calcolare B della partizione 
    p = len( examples.loc[examples['Wait'] == 'Yes']['Wait'] )
    n  = len( examples.loc[examples['Wait'] == 'No']['Wait'] )
    
    for v in value_list:
        # dataset prendendo la partizione con l'attributo "attribute"
        ex = examples.loc[examples[attribute] == v]

        # numeri si Yes e No per calcolare B della partizione
        pk = len( ex.loc[ex['Wait'] == 'Yes']['Wait'] )
        nk  = len( ex.loc[ex['Wait'] == 'No']['Wait'] )

        # B( pk / ( pk + nk) ) -> se query_no = 1 / 0 -> il log fa 0 -> b_value = 0
        b_value = B( pk, nk)

        # ( pk + nk ) / ( p + n )
        coef    = ( pk + nk) / ( p + n )

        # valore di remainder calcolato
        value = coef*b_value

        actual_remainder += value

        if debug == True: 
            print("\tp: " + str(p) + ", n: " + str(n) + ", pk: " + str(pk) + ", nk: " + str(nk) + " -> " + "coef: " + str(coef) + ", b_value: " + str(b_value))
            print("\tcalcolato: " + str(v) + " -> valore : " + str(value) + "\n")

    if debug == True: 
        print("\tremainder: " + str(attribute) + ", value: " + str(actual_remainder))
    return actual_remainder

def plurality_value(examples):  # ritorna il massimo tra yes o no in ex
    query_yes   = len( examples.loc['Wait == Yes']['Wait'] )
    query_no    = len( examples.loc['Wait == No']['Wait'] )

    if query_yes > query_no:
        return "Yes"
    else:
        return "No"
    
def is_same_classification(examples):   # verifica se sono tutti yes o no
    query_yes = len( examples.loc[examples['Wait'] == 'Yes']['Wait'] )
    query_no  = len( examples.loc[examples['Wait'] == 'No']['Wait'] )
    
    if  (query_yes == 0 and query_no > 0):
        if debug == True: print("hanno stessa classificazione -> clas: No")
        return True
    elif query_yes > 0 and query_no == 0:
        if debug == True: print("hanno stessa classificazione -> clas: Yes")
        return True
    else:
        if debug == True: print("non hanno stesso classificazione")
        return False

def get_classification(examples):   # ritorna Yes o No
    return examples['Wait'].tolist()[0]

# costruisce l'albero di decisione
def learn_decision_tree(
                        examples,           # input / data-set
                        attributes,         # lista di attributi da analizzare
                        parent_examples):   # dataset del nodo padre
    
    if debug == True: print("\n\n-----------CHIAMATA-------------")

    if examples.empty == True:
        if debug == True: print("example è vuoto")
        return plurality_value(parent_examples)

    elif is_same_classification(examples) == True:
        if debug == True: print("stessa classificazione")
        return get_classification(examples)

    elif attributes == []:
        if debug == True: print("attributes è vuoto")
        return plurality_value(examples)

    else:
        att_max = importance(attributes,examples) # restituisce una stringa con il nome dell'attributo

        if debug == True: print("attributo più importante: " + str(att_max))

        value_list = list( set(examples[att_max].tolist()) )    # lista dell'insieme dei valori dell'attributo

        # lista degli attributi per i sotto-alberi ( quindi levando l'attributo su cui partiziono )
        list_att_subtree = remouve_elem(attributes,att_max)

        if debug == True: 
            print(attributes)
            print(list_att_subtree)

        #    examples, father, soons, list_attributes, height = 0, label = ''
        tree = decision_tree_node(
                                    examples.copy(),                # examples
                                    parent_examples.copy(),         # father
                                    {},                             # soons
                                    list_att_subtree,               # list_attributes
                                    att_max                         # label 
                                )
        if debug == True:  
            print("\nstampo i valori")
            print(value_list)

        for v in value_list:
            if debug == True: print("\nvalore: " + str(v))
            # dati che ha il nodo
            query = examples.loc[examples[att_max] == v]

            # examples, attribute, parent_examples
            subtree = learn_decision_tree(  query.copy(),                      # dataset "ridotto",, quello che soddisfa la query
                                            list_att_subtree,    # lista degli attributo senza att_max
                                            examples.copy(),            # dataset del nodo padre                                               
                                            )

            # add a brach to tree with label ( A = V ) and subtree subtree
            tree.add_soon(v,subtree)
        
        return tree

def remouve_elem(list, elem):   # rimuove elem dalla lista 
    if list == []:
        return []
    else:
        mylist = []
        for i in list:
            if i != elem:
                mylist.append(i)
        return mylist

def main():

    print("...creazione DT ... ")

    if debug == True: print(data.columns.tolist().copy())

    my_tree = learn_decision_tree(  data.copy(),   #  examples
                                    remouve_elem(data.columns.tolist().copy(),'Wait'),   # attributi senza la colonna del risultato
                                    []   # parent_examples
                                     )

    print("\nstampo il mio DT\n")

    stampa_albero(my_tree,0)

def stampa_albero(tree, heigth):    # stampa l'albero decision-tree con tabulazioni

    num_tab = ''
    for i in range(heigth): # aggiunta tabulazioni per formattazione
        num_tab += '\t'

    print( num_tab + "|-> " + str(tree.label) + ", numero figli: " + str(len(tree.soons)))
    
    for i in tree.soons.keys():     # stampo tutte le coppie < chiave, valore >

        my_string = num_tab + "\t|-> " + str(i) + ": "
        if type(tree.soons[i]) == str:
            my_string += tree.soons[i] 
        else:
            my_string += tree.soons[i].label + " -> tree"

        print(my_string)

    print()
    for i in tree.soons.keys(): # chiamo la stampa sui sottoalberi
        elem = tree.soons[i]
        if type(elem) != str:
            stampa_albero(elem, heigth + 1)

main()

                            ###########################
                            # Tedeschi Jacopo 1882789 #
                            ###########################