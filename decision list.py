from contextlib import nullcontext
import pandas as pd
                            ###########################
                            # Tedeschi Jacopo 1882789 #
                            ###########################


data = pd.read_csv("restaurant_waiting.csv",delimiter=',')
list_attributes = data.columns.tolist()

# variabili utili
DEBUG       = False
FAILURE     = "Failed"
EMPTY_TEST  = "Empty test"

class list_node:    #class list_node:
    def __init__(self, test, result, remaning_test):
        self.test           = test    # test del nodo
        self.result         = result  # outcome, classificazione
        self.remaning_test  = remaning_test # resto della lista

class test: # definito con < attributo, valore >
    def __init__(self, attribute, value):
        self.attribute  = attribute # attributo del test
        self.value      = value     # valore del test

def is_same_classification(examples):   # verifica se sono tutti yes o no
    query_yes = len( examples.loc[examples['Wait'] == 'Yes']['Wait'] )
    query_no  = len( examples.loc[examples['Wait'] == 'No']['Wait'] )
    
    if  (query_yes == 0 and query_no > 0):
        if DEBUG == True: print("hanno stessa classificazione -> clas: No")
        return True
    elif query_yes > 0 and query_no == 0:
        if DEBUG == True: print("hanno stessa classificazione -> clas: Yes")
        return True
    else:
        if DEBUG == True: print("non hanno stesso classificazione")
        return False

def all_positive_classification(examples):   # verifica se sono tutti "Yes"
    query_yes = len( examples.loc[examples['Wait'] == 'Yes']['Wait'] )
    query_no  = len( examples.loc[examples['Wait'] == 'No']['Wait'] )
    
    # sono tutti positivi
    if query_yes > 0 and query_no == 0:
        return True
    else:
        return False

def remouve_examples(examples, test): # rimuove da examples le tuple che rispettano il test
    
    # metto gli indici nei dataset
    df      = examples.copy()

    attributes = df.columns.tolist()
    
    query = df.loc[ examples[test.attribute] != test.value ]

    if DEBUG == True:   print(query)

    return query.copy()      

def is_test_empty(test):    # controlla se c'è un test
    if test == []:
        return True
    else:
        return False
    
def find_test(examples):

    max_attribute = ''    # attributo del test
    max_value     = 0     # valore che rende max la cardinalità del subset
    max_n_tuple   = 0     # numero di tuple del subset
    max_subset    = 0     # subset

    for a in list_attributes:
        value_list = list( set( examples[a].tolist() ) )    # lista valori assunti dall'attributo

        for v in value_list:
            query = examples.loc[ examples[a] == v ] # subset con le righe che hanno, in corrispondenza di a, il valore v
            n_tuple = len(query)    # numero di tuple del subset

            # se sono tutti "Yes"/"No" e questo subset è più grande, switcho
            if is_same_classification(query) and n_tuple > max_n_tuple:
                max_attribute = a
                max_value = v
                max_n_tuple = n_tuple
                max_subset = query.copy()
    
    # rimuovo l'attributo dalla lista per non analizzarlo più
    list_attributes.remove(max_attribute)

    return test(max_attribute, max_value ), max_subset.copy()

def decision_list_learning( examples ):

    #print(examples)
    # se è vuoto, ritorna la lista banale "No"
    if examples.empty == True:
        return "No"
        #return list_node(null, "no", null)

    # crea un test
    test, subset  = find_test(examples.copy())

    # se non ci sono testo -> fallimento
    if is_test_empty(test) == True:
        return FAILURE
    
    if all_positive_classification(subset.copy()) == True:
        o = "Yes"
    else:
        o = "No"

    # nuovo dataset
    new_dataset = remouve_examples( examples.copy(), test )

    # resto della lista
    remaning_test = decision_list_learning( new_dataset.copy() )

    # nodo della lista
    my_dl = list_node(test, o, remaning_test)

    # se gli è stato attaccato un fallimento, non ritorno il nodo ma il fallimento
    if remaning_test == FAILURE:
        return FAILURE

    return my_dl


def main():

    print("...inizia il DL, calcolo...")
    list_attributes.remove('Wait')
    print(list_attributes)

    my_dl = decision_list_learning(data)

    print("\nstampo la mia dl")
    print_dl(my_dl)

def print_dl(dl):

    if type(dl) != str:
        print("test_att: {},\t\ttest_value: {},\t\toutcome: {} ".format(dl.test.attribute, dl.test.value, dl.result))

        print_dl(dl.remaning_test)
    else:
        print(dl)


main()
                            ###########################
                            # Tedeschi Jacopo 1882789 #
                            ###########################