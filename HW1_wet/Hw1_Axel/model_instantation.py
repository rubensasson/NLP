# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:31:25 2020

@author: galye
"""

from collections import OrderedDict


train1_filename = "../HW1_wet/data/train1.wtag"
train2_filename = "../HW1_wet/data/train2.wtag"

test1_filename = "../HW1_wet/data/test1.wtag"

comp1_filename = "../HW1_wet/data/comp1.words"
comp2_filename = "../HW1_wet/data/comp2.words"

model = 2

'''
la fonction crée des history depuis les phrase du trainSet, et elle tiens une liste des mots déjà vu
dans l'ensemble du train set 
Input : une phrase du Train set 
Output : une liste contenant l'ensemble des histoires que contient une phrase 
ex : The dog is big 
on aura donc une liste tel que : 
1 - [*,*, The, *, *, DT, 1]
2 - [*, The, Dog, *, DT, NN, 2]
3 - [The, Dog, is, DT, NN, VB, 3]
4 - [Dog, Is, Big, NN, VB, ADJ, 3]
'''
def create_histories_from_sentence(s, single):
    global seen_words
    histories = []
    words_tags = s.split()
    words = [w_t.split('_')[0] for w_t in words_tags]
    tags = [w_t.split('_')[1] for w_t in words_tags]
    for i, w in enumerate(words):
        if single and w in seen_words:
            continue
        if i == 0:
            histories.append(('*', '*', words[i], '*', '*', tags[i], i))
        elif i == 1:
            histories.append(('*', words[i - 1], words[i], '*', tags[i - 1], tags[i], i))
        else:
            histories.append((words[i - 2], words[i - 1], words[i], tags[i - 2], tags[i - 1], tags[i], i))
        if single:
            seen_words.add(w)
    return histories

'''
La fonction crée des Histories a partir de toutes les phrases du texte en réutilisant la fonction d'avant 
Input : L'ensemble des phrases du Train set 
Output : Une liste contenant toutes les histoiries 
'''
def creating_histories(sentences, single=False):
    histories = []
    for s in sentences:
        histories += create_histories_from_sentence(s, single)
    return histories


class data_dicts:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_dict_count = OrderedDict()  #compte le nombre d'apparition de tous les couples (Tags/Word)
        self.words_dict_count = OrderedDict() #compte le nombre d'apparition de tout les mots
        self.tags_dict_count = OrderedDict() #compte le nombre d'apparition de tout les tags
        self.tags_dict_count['*'] = 1  #On rajoute le tag '*'

        self.suffixes = OrderedDict()  #Compte le nombre d'apparition de chaque suffixes
        self.prefixes = OrderedDict()  #compte le nombre d'apparition de chaque prefixes

    def init_dicts_and_statistics(self, file_sentences):
        """
            Cette fonction crée des dictionaires qui donnes le nombre d'apparitons pour chaques mots et Tags
            input : Les phrases du Train set
            Output : pluieurs dictionaire contant le nombre d'apparition de :
            - (Word, tag)
            - tag
            - word
            - prefixe
            - suffixe
        """
        for line in file_sentences:
            splited_words = line.split()
            for word_idx in range(len(splited_words)):
                cur_word, cur_tag = splited_words[word_idx].split('_')
                if cur_word not in self.words_dict_count:
                    self.words_dict_count[cur_word] = 1
                else:
                    self.words_dict_count[cur_word] += 1

                if cur_tag not in self.tags_dict_count:
                    self.tags_dict_count[cur_tag] = 1
                else:
                    self.tags_dict_count[cur_tag] += 1

                if (cur_word, cur_tag) not in self.words_tags_dict_count:
                    self.words_tags_dict_count[(cur_word, cur_tag)] = 1
                else:
                    self.words_tags_dict_count[(cur_word, cur_tag)] += 1

        for word in list(self.words_dict_count.keys()):  #on passe sur toute la liste des mot et on crée un emplacement dans le dictionaire des suffices et des prefixes pour chacun
            if len(word) >= 4:
                self.suffixes[word[-4:]] = 1
                self.prefixes[word[0:4]] = 1
            if len(word) >= 3:
                self.suffixes[word[-3:]] = 1
                self.prefixes[word[0:3]] = 1
            if len(word) >= 2:
                self.suffixes[word[-2:]] = 1
                self.prefixes[word[0:2]] = 1
            if len(word) >= 1:
                self.suffixes[word[-1:]] = 1
                self.prefixes[word[0:1]] = 1

    def init_dicts_and_statistics_competition(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.split()
                for cur_word in range(len(splited_words)):
                    if cur_word not in self.words_dict_count:
                        self.words_dict_count[cur_word] = 1
                    else:
                        self.words_dict_count[cur_word] += 1


with open(train1_filename, 'r') as train:
    train1_sentences = train.readlines()
with open(test1_filename, 'r') as test:
    test1_sentences = test.readlines()
with open(comp1_filename, 'r') as test:
    test1_comp = test.readlines()

with open(train2_filename, 'r') as train:
    train2_sentences = train.readlines()
with open(comp2_filename, 'r') as test:
    test2_comp = test.readlines()

train_data_dict_1 = data_dicts()   #on crée l'objet data dict pour le train1
train_data_dict_1.init_dicts_and_statistics(train1_sentences)  #on appel la fonction qui compte le nombre d'apparition de chaque mot du train et qui le met dans un dictionaire

train_data_dict_2 = data_dicts()  #pareil que pour le train 1 mais pour le train 2
train_data_dict_2.init_dicts_and_statistics(train2_sentences)

X_1 = creating_histories(train1_sentences, single=False)  #on crée les histories pour le train 1
X_2 = creating_histories(train2_sentences, single=False)  #on crée les histoiries pour le train 2

if model == 1:
    words_as_list = list(train_data_dict_1.words_dict_count.keys())   #List des mots présent dans le train
    tags_as_list = list(train_data_dict_1.tags_dict_count.keys())   #list des tags présents dans le test
    train_data_dict = train_data_dict_1

    number_of_words = len(train_data_dict_1.words_dict_count.keys())   #nombre de mot présent dans le train
    number_of_tags = len(train_data_dict_1.tags_dict_count.keys())     #nombre de tag présnt dans le train
    len_of_suffixes = len(train_data_dict_1.suffixes.keys())   #nombre de suffix présents dans le train
    len_of_prefixes = len(train_data_dict_1.prefixes.keys())   #nombre de préfix présent dans le train

    opt_number_of_features = 2134770
else:
    words_as_list = list(train_data_dict_2.words_dict_count.keys())
    tags_as_list = list(train_data_dict_2.tags_dict_count.keys())
    train_data_dict = train_data_dict_2

    number_of_words = len(train_data_dict_2.words_dict_count.keys())
    number_of_tags = len(train_data_dict_2.tags_dict_count.keys())
    len_of_suffixes = len(train_data_dict_2.suffixes.keys())
    len_of_prefixes = len(train_data_dict_2.prefixes.keys())

    opt_number_of_features = 289726

words_as_list.append("*")   #on rajoute dans la liste '*' et '.'
words_as_list.append(".")

Y_viterbi = list(train_data_dict.tags_dict_count.keys())   #le Y du Viterbi est l'ensemble des tag présent dans le train


"""
l'implementation des features sont 
f100 (word/tag features)
f1, f2, f3 qui sont les fonctions unigram, bigram et Trigram 
f4, f5, f6, f7, f8, f9 qui sont les fonction de languages modeling 
f101 f102 qi sont les fonctions préfixes et suffixes 
f103, f104, f105, f106, f107 qui sont les fonctions de contextual features 

"""


"""
l'encodage des fonction f100, il s'agit de faire une feature tel que pour chaque couple de mot (word, tag) du trainSet 
nous ayant une sous fonction de f100 tel que si il y a ca on met 1 et sinon 0
input x = 'history qui est en vérité l'ensembe des mots 
      y = les tags (qu'il faudra prédire)
      start 
"""

def word_tag_features(x, y, start):
    if start != 0:
        print("word_tag_features should be the first function !")
        exit()
    res = []
    num_word_tag_features = number_of_words * number_of_tags  #nombr d'entrée du sous vecteur de la fonction f100
    try:
        word = x[2] #le mot est le deuxieme de la liste car x c'est l'history
        word_index = words_as_list.index(word) #index de l'emplacement du word sur la liste de l'ensemble des mots
        tag_index = tags_as_list.index(y) #index de l'emplacement du tag sur la liste de l'ensemble des mots
        res.append(word_index * number_of_tags + tag_index)  # is not 0/1   #on ajoute a la liste res l'emplacement du 1 dans le sous vecteur f100
    except ValueError:
        pass
    return res, num_word_tag_features #on retourne la liste des emplacements des 1 dans le vecteur, et l nombre d'entré du vecteur



"""
construit les fonctions de features de type prefixe et suffixes
"""
# Prefix/Suffix with Tag
def pre_suf_tag_features(x, y, start):
    res = set()
    num_pre_suf_tag_features = len_of_prefixes * number_of_tags + len_of_suffixes * number_of_tags  #il s'agit de la reunion de 2 vecteur qui seront celui de la taille nbsuffix*nbTag et pareil pour l'autre
    start_suf = len_of_prefixes * number_of_tags #le moment ou on comence les suffixes
    tag_index = list(train_data_dict.tags_dict_count.keys()).index(y)  #retourne la liste des index des tag de la phrase x du train
    word = x[2]  #on prend le dernier mot
    prefixes_as_list = list(train_data_dict.prefixes.keys())  #liste des prefix
    suffixes_as_list = list(train_data_dict.suffixes.keys())  #lisste des suffixes
    try:
        res.add(start + prefixes_as_list.index(word[0:1]) * number_of_tags + tag_index) #on met un 1 pour la premiere lettre du prefixe et son tag (si le prefix a une longeur >= 4
        res.add(start + prefixes_as_list.index(word[0:2]) * number_of_tags + tag_index) #on met un 1 pour la premiere lettre du prefixe et son tag (si le prefix a une longeur >= 3
        res.add(start + prefixes_as_list.index(word[0:3]) * number_of_tags + tag_index) #on met un 1 pour la premiere lettre du prefixe et son tag (si le prefix a une longeur >= 2
        res.add(start + prefixes_as_list.index(word[0:4]) * number_of_tags + tag_index)  #on met un 1 pour la premiere lettre du prefixe et son tag (si le prefix a une longeur >= 1
    except ValueError:
        pass

    try:
        res.add(start + start_suf + suffixes_as_list.index(word[-1:]) * number_of_tags + tag_index)
        res.add(start + start_suf + suffixes_as_list.index(word[-2:]) * number_of_tags + tag_index)
        res.add(start + start_suf + suffixes_as_list.index(word[-3:]) * number_of_tags + tag_index)
        res.add(start + start_suf + suffixes_as_list.index(word[-4:]) * number_of_tags + tag_index)
    except ValueError:
        pass

    return list(res), num_pre_suf_tag_features #on return la liste des 2 sous vecteurs f101 et f102


"""
renvoi la partie du vecteur qui concerne les annotation unigram, bigram et trigram 
input, history, et les tags
output : une listye dees enroits ou doivent etre placé les 1 pour cette sous partie du vecteur
"""

# Uni/Bi/Trigram
def tri_bi_uni_features(x, y, start):
    res = []
    num_tri_uni_bi_features = number_of_tags ** 3 + number_of_tags ** 2 + number_of_tags  #indice de a quel emplacement la partie trigram du sous vcteur f1,f2,f3 va commencer
    bigram_start = number_of_tags ** 3  #pareil pour vecteur bigram
    unigram_start = number_of_tags ** 3 + number_of_tags ** 2  #pareil pour vecteur unigram

    idx_current = tags_as_list.index(y)   #index du mot wi
    idx_last = tags_as_list.index(x[4])  #index du mot wi-1
    idx_before_last = tags_as_list.index(x[3]) #index du mot wi-3

    res.append(start + idx_before_last * (number_of_tags ** 2) + idx_last * number_of_tags + idx_current)  #on note les emplacements des 1 dans la futur partie du vecteur (le sous vecteur f1,f2,f3)
    res.append(start + bigram_start + idx_last * number_of_tags + idx_current)
    res.append(start + unigram_start + idx_current)
    return res, num_tri_uni_bi_features


"""
meme fonctionelent pour construire la sous partie du vecteur qui concerne f5
"""
def previous_word_features(x, y, start):
    res = []
    num_features = number_of_words * number_of_tags + number_of_tags

    try:
        last_word = x[1]
        last_word_index = words_as_list.index(last_word)
        tag_index = tags_as_list.index(y)
        res.append(start + last_word_index * number_of_tags + tag_index)
    except ValueError:
        pass
    return res, num_features


# Defining Other features
NUMBERS = [str(i) for i in range(10)]
number_features = [
    lambda x, y, start, number=number: ([start], 1) if str(number) in str(x[2]) and y == 'CD' else ([], 1) for number in
    NUMBERS]
proper_noun_feature = lambda x, y, start: ([start], 1) if str(x[2][0]).isupper() and y == 'NNP' else (
    [], 1)  # First Letter is capital
all_capital_feature = lambda x, y, start: ([start], 1) if str(x[2]).isupper() and y == 'NNP' else (
    [], 1)  # All letters are capital (Ex: USA)
pattern_ness = lambda x, y, start: ([start], 1) if x[2][-4:] == "ness" and y == 'NN' else (
    [], 1)
assocc_JJ_NN = lambda x, y, start: ([start], 1) if x[4] == "NN" and y == 'JJ' else (
    [], 1)
assocc_NN_JJ = lambda x, y, start: ([start], 1) if x[4] == "JJ" and y == 'NN' else (
    [], 1)

kappa_features = lambda x, y, start: ([start], 1) if x[2][-5:] == "kappa" and y == 'NN' else (
    [], 1)
virus_features = lambda x, y, start: ([start], 1) if x[2][-5:] == "virus" and y == 'NN' else (
    [], 1)
ol_features = lambda x, y, start: ([start], 1) if x[2][-2:] == "ol" and y == 'NN' else (
    [], 1)
lexical_features = lambda x, y, start: ([start], 1) if x[1] == "an" and ((y == 'NN') or (y == 'JJ')) else (
    [], 1)



"""
permet pour chaque history et tags de donner une liste d'index o seront placé tous les 1 dans le futur vecteur binaire, permet 
aussi de calculer la taille grlobal du futur vecteur binaire. 
input: l'history, les tags 
output : renvoi la liste des indexes ou sont placés les 1 dans le futur vecteur binaire, permet de calculer la longueur global du vecteur 
"""
def model1_feat_func(x, y):
    global opt_number_of_features
    final = []
    start = 0
    funcs = [word_tag_features, pre_suf_tag_features, tri_bi_uni_features, proper_noun_feature, all_capital_feature,
             pattern_ness, assocc_JJ_NN, assocc_NN_JJ, previous_word_features]
    funcs += number_features
    for i, func in enumerate(funcs):
        feats, feat_len = func(x, y, start)
        final += feats
        start += feat_len
    opt_number_of_features = start
    return final


def model2_feat_func(x, y):
    global opt_number_of_features
    final = []
    start = 0
    funcs = [word_tag_features, pre_suf_tag_features, tri_bi_uni_features, proper_noun_feature, all_capital_feature,
             pattern_ness, assocc_JJ_NN, assocc_NN_JJ, previous_word_features, kappa_features, virus_features,
             ol_features, lexical_features]
    funcs += number_features
    for i, func in enumerate(funcs):
        feats, feat_len = func(x, y, start)
        final += feats
        start += feat_len
    opt_number_of_features = start
    return final


def model_feat_func(x, y):
    if model == 1:
        return model1_feat_func(x, y)
    else:
        return model2_feat_func(x, y)
