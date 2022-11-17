import numpy as np

from model_instantation import train_data_dict_1, test1_sentences, model1_feat_func, test1_comp, model2_feat_func, \
    test2_comp, model_feat_func, model, train2_sentences
from predict import viterbi


def test(model, feat_func):
    # Loading Params
    Y = list(train_data_dict_1.tags_dict_count.keys()) #tous les tags possibles dans le dataset

    if model == 1:
        test_sentences = test1_sentences
    else:
        test_sentences = train2_sentences

    optim_v = np.load('optimal_v_m' + str(model) + '.npy')   #on load les poids des features qu'on a deja trouver via l'optimisation
    hash_table = np.load('HASH_TABLE_m' + str(model) + '.npy')  #on load la table Hash qui correspond a tous les indexs des features les plus importantes dans le dataset
    num_worst_chosen = 10
    dict_tags = {i: Y[i] for i in range(len(Y))}  #on donne a chaque  numero un tag
    dict_tags_inv = {v: k for k, v in dict_tags.items()} #on donne a chaque
    confusion_matrix = np.zeros((len(Y), len(Y)))  #a chaque fois qu'on a une prediction on la rentre, a la fin pour caluler l'accuracy on a plus qu'a calculer la trace de la matrice
    for i, s in enumerate(test_sentences):
        if i % 100 == 0:
            print(i)
        # Run Viterbi
        words, predicted, actual = viterbi(s, optim_v, feat_func, hash_table, beam_size=2, tagged=True)
        print("predicted = ", predicted)
        print("actual = ", actual)
        # Update confusion Matrix
        for j in range(len(predicted)):
            pred = dict_tags_inv[predicted[j]]
            act = dict_tags_inv[actual[j]]
            confusion_matrix[pred, act] += 1
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        print("accuracy = ", 100 * accuracy, "%")
    # Removing Trace from confusion matrix to select worst ten tags
    for_worst = confusion_matrix - np.diag(np.diag(confusion_matrix))
    # Selecting ten worst tags
    ten_worst = np.argsort(np.sum(for_worst, axis=0))[-num_worst_chosen:]
    ten_worst_conf_mat = confusion_matrix[np.ix_(ten_worst, ten_worst)]
    # Computing Accuracy

    print("Model " + str(model) + " Accuracy.: " + str(100 * accuracy) + " %.")
    print("Ten Worst Elements: " + str([dict_tags[i] for i in ten_worst]))
    print("Confusion Matrix for the ten worst elements: ")
    print(ten_worst_conf_mat)


def comp(model, feat_func):
    # Loading Params

    optim_v = np.load('optimal_v_m' + str(model) + '.npy')
    hash_table = np.load('HASH_TABLE_m' + str(model) + '.npy')

    if model == 1:
        test_comp = test1_comp
    else:
        test_comp = test2_comp
    final_sentences = []
    for i, s in enumerate(test_comp):
        if i < 100:
            continue
        if i % 1 == 0:
            print(i)
        # Run Viterbi
        words, predicted = viterbi(s, optim_v, feat_func, hash_table, beam_size=2, tagged=False)
        tagged_sentence = ' '.join([words[i] + '_' + predicted[i] for i in range(len(words))])
        tagged_sentence += '\n'
        final_sentences.append(tagged_sentence)

    with open('comp' + str(model) + '.wtag', 'w') as comp:
        comp.writelines(final_sentences)


if __name__ == '__main__':
    test(model, model_feat_func)
