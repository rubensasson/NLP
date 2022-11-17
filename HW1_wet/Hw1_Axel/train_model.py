import numpy as np
import scipy.optimize as opt

from model_instantation import opt_number_of_features, X_1, train_data_dict_1, number_of_tags, \
    X_2, train_data_dict_2, model_feat_func, model    #importation des modèles

tag_mats = []   #vecteur binaire pour toutes les histories
Y = []  #ensemble des tags pour chaque history
crucial_count = []  #les indices des 1 dans chaque vecteur binzaire pour toute sles histories i
REG_PARAM = 0.5
iteration = 1
X = []


"""
fonction qui recois le vecteur de weight V et une history i et qui renvoi le score 
"""
def exp_scores(v, i):
    correct_ind = tag_mats[i, :]  #les indexs des entrées 1 pour l'history i
    norm_scores = [v[ind].sum() for ind in correct_ind]  #la somme des poids correspondant aux indexs des tags dans le vecteur de l'ensemble des tags
    return np.exp(norm_scores)  #on le passe en expontiel pour récuperer une probabilité apres


# Computes likelihood given vector v
def likelihood(v):
    global tag_mats, Y, crucial_count, REG_PARAM

    res = 0.0
    res += np.dot(v, crucial_count)  #= nombre maximum de la valeur de v pour chaque entrée du vecteur v
    to_substract = 0.0
    for i, history in enumerate(X):
        for j, tag in enumerate(Y):
            to_substract += np.exp(v[tag_mats[i, j]].sum()) #on fait la somme de tous les produits scalaire entre v et les vecteurs binaires pour toutes les histories et tout les tags
        res -= np.log(to_substract)  #on l'enleve a res
        to_substract = 0.0 #et on recomence ainsi de suite
    res -= REG_PARAM * np.linalg.norm(v) ** 2 / 2
    return -res  # returning -res because we can only minimize with scipy


# Computes likelihood gradient given vector v
#on fait une optimizatioon par gradient ascent pour toutes les entrée du vecteur weight
def likelihood_grad(v):
    global tag_mats, Y, crucial_count, REG_PARAM, iteration
    res = crucial_count.copy()

    print("iteration number", iteration)
    iteration += 1
    for i, history in enumerate(X):
        probas = exp_scores(v, i)
        probas /= probas.sum()
        for j, tag in enumerate(Y):
            res[tag_mats[i, j]] -= probas[j]
    res -= REG_PARAM * v
    return -res

"""
on train le model a l'aide du model lui meme et les features fonctions qu'on utilise 
"""
def train(model, feat_func):
    global Y, tag_mats, crucial_count, X

    compute_H = True
    compute_tag_mat = True

    v = np.random.randn(opt_number_of_features)  #on créer un vecteur de v de la taille du nombre d'entré du vecteur et on met des poids au hazard
    if model == 1:
        Y = train_data_dict_1.tags_dict_count  #Y est la liste de tous les tags pour chaque history x
        X = X_1    #X est la liste de toutes les histories
    else:
        Y = train_data_dict_2.tags_dict_count
        X = X_2
    if compute_H is True:
        H = np.zeros(opt_number_of_features) #H est un vecteur de zero de la taille du nombre d'entrée du vecteur binaire
        print("number of features at first", opt_number_of_features)
        for i, history in enumerate(X):
            if i % 10000 == 0:  # To display evolution
                print(i)
            real_tag = history[-2] #c'est le tag du mot Wi
            np.add.at(H, feat_func(history, real_tag), 1)  #H est un vecteur qui indique combien de fois chaque features a été utilisé, pour chaque entrée on a un nombre qui nous dit combien de fois on a mis a un a cette position dans l'ensemble des vecteurs binaire provenant du trainSet
        np.save('H_m' + str(model) + '.npy', H)
    else:
        H = np.load('H_m' + str(model) + '.npy')

    crucial_threshold = 1
    crucial_indices = np.where(H >= crucial_threshold)[0]  #C'est une liste des indices qui renvoi au features qui ont été utilisés plus d'une fois, c'est l'indices des features importantes dans le data set
    print(crucial_indices)
    crucial_count = H[crucial_indices]  #c'est la liste du nombre de fois ou on a utiliser les features IMPORTANTES (celle qui ont été utilisé plus d'unf ois, c'est H sans les Zeros)
    print("number of features retained", len(crucial_indices))  #c'est le nombre de features qu'on retiendra

    hash_table = -np.ones(opt_number_of_features)  #vecteur de -1 de la taille du nombre de features

    for idx, cruc_ind in enumerate(crucial_indices):   #on rentre l'index de chaque features Importante ans la hash table, les features jugé non importnte auront la cvaleur -1, les importantes auront leur index qui est leur place dans le vecteur binaire
        hash_table[cruc_ind] = idx

    if compute_tag_mat is True:

        tag_mats = np.zeros((len(X), number_of_tags), dtype=object)  #on crée une matrice de zero avec n = le nombre d'history de notre data set, m = nb de tag de notre dataset
        for i, history in enumerate(X):
            if i % 1000 == 0:  # To display evolution
                print(i)
            for j, tag in enumerate(Y):
                feat_func_res = feat_func(history, tag)  #renvoi le vecteur contenant les indeces des endroits du vecteur contenant des 1 et la taille du vecteur
                correct_ind = hash_table[feat_func_res]
                correct_ind = correct_ind[correct_ind > 0]    #on prend les indexe des endroit ou il y a des 1 dans le vecteur binaire
                correct_ind = np.array(correct_ind, dtype='int')
                tag_mats[i, j] = correct_ind    #pour chaque history et pour chaque tag possible on obtient un vecteur contenant les index de la ou il y a les 1 dans le vecteur binaire
        np.save('tag_mats_m' + str(model) + '.npy', tag_mats)
        np.save('hash_table_m' + str(model) + '.npy', hash_table)
    else:
        tag_mats = np.load('tag_mats_m' + str(model) + '.npy')

    # Running Optimization function.
    #on entraine le model pour trouver le bon vecteur de poid grace a un gradient ascent
    optim_res = opt.minimize(likelihood, v[crucial_indices], method='L-BFGS-B', jac=likelihood_grad,
                             options={'disp': False, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05,
                                      'eps': 1e-05, 'maxfun': 200, 'maxiter': 200, 'iprint': -1, 'maxls': 20})

    np.save('optimal_v_m' + str(model), optim_res.x)
    np.save('crucial_indices_m' + str(model) + '.npy', crucial_indices)


if __name__ == '__main__':
    train(model, model_feat_func)
