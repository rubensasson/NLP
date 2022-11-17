from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple


WORD = 0
TAG = 1


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107"]  # the feature classes used in the code
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test

    def get_word_tag_pair_count(self, file_path) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1

                    #f100
                    if (cur_word, cur_tag) not in self.feature_rep_dict["f100"]:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] += 1

                    #f101 and f102
                    if len(cur_word) >= 4:
                        # for prefix
                        if (cur_word[0:4], cur_tag) not in self.feature_rep_dict["f102"]:
                            self.feature_rep_dict["f102"][(cur_word[0:4], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f102"][(cur_word[0:4], cur_tag)] += 1
                        # for suffix
                        if (cur_word[-4:], cur_tag) not in self.feature_rep_dict["f101"]:
                            self.feature_rep_dict["f101"][(cur_word[-4:], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f101"][(cur_word[-4:], cur_tag)] += 1

                    if len(cur_word) >= 3:
                        # for prefix
                        if (cur_word[0:3], cur_tag) not in self.feature_rep_dict["f102"]:
                            self.feature_rep_dict["f102"][(cur_word[0:3], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f102"][(cur_word[0:3], cur_tag)] += 1
                        # for suffix
                        if (cur_word[-3:], cur_tag) not in self.feature_rep_dict["f101"]:
                            self.feature_rep_dict["f101"][(cur_word[-3:], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f101"][(cur_word[-3:], cur_tag)] += 1

                    if len(cur_word) >= 2:
                        # for prefix
                        if (cur_word[0:2], cur_tag) not in self.feature_rep_dict["f102"]:
                            self.feature_rep_dict["f102"][(cur_word[0:2], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f102"][(cur_word[0:2], cur_tag)] += 1
                        # for suffix
                        if (cur_word[-2:], cur_tag) not in self.feature_rep_dict["f101"]:
                            self.feature_rep_dict["f101"][(cur_word[-2:], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f101"][(cur_word[-2:], cur_tag)] += 1

                    if len(cur_word) >= 1:
                        # for prefix
                        if (cur_word[0:1], cur_tag) not in self.feature_rep_dict["f102"]:
                            self.feature_rep_dict["f102"][(cur_word[0:1], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f102"][(cur_word[0:1], cur_tag)] += 1
                        # for suffix
                        if (cur_word[-1:], cur_tag) not in self.feature_rep_dict["f101"]:
                            self.feature_rep_dict["f101"][(cur_word[-1:], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f101"][(cur_word[-1:], cur_tag)] += 1


                    if word_idx == 0:
                        OneWordBack = '*'
                        TwoTagBack = '*'
                        OneTagBack = '*'

                    else:
                        OneWordBack, OneTagBack = split_words[word_idx - 1].split('_')

                    if word_idx == 1:
                        TwoTagBack = '*'

                    else :
                        if word_idx != 0:
                            TwoWordBack, TwoTagBack = split_words[word_idx - 2].split('_')

                    #Build f103
                    if (TwoTagBack, OneTagBack, cur_tag) not in self.feature_rep_dict['f103']:
                        self.feature_rep_dict['f103'][(TwoTagBack,OneTagBack,cur_tag)] = 1
                    else :
                        self.feature_rep_dict['f103'][(TwoTagBack, OneTagBack, cur_tag)] += 1

                    # Build f104
                    if (OneTagBack, cur_tag) not in self.feature_rep_dict['f104']:
                        self.feature_rep_dict['f104'][(OneTagBack, cur_tag)] = 1
                    else:
                        self.feature_rep_dict['f104'][(OneTagBack, cur_tag)] += 1

                    # Build f105
                    if (cur_tag) not in self.feature_rep_dict['f105']:
                        self.feature_rep_dict['f105'][(cur_tag)] = 1
                    else:
                        self.feature_rep_dict['f105'][(cur_tag)] += 1

                    # Build f106
                    if (OneWordBack, cur_tag) not in self.feature_rep_dict['f106']:
                        self.feature_rep_dict['f106'][(OneWordBack, cur_tag)] = 1
                    else:
                        self.feature_rep_dict['f106'][(OneWordBack, cur_tag)] += 1

                    size = len(split_words) - 1

                    if word_idx <= size:
                        if word_idx == size:
                            OneWordForward = '~'
                        else:
                            OneWordForward, OneTagForward = split_words[word_idx + 1].split('_')
                    # Build f107
                    if (OneWordForward, cur_tag) not in self.feature_rep_dict['f107']:
                        self.feature_rep_dict['f107'][(OneWordForward, cur_tag)] = 1
                    else:
                        self.feature_rep_dict['f107'][(OneWordForward, cur_tag)] += 1

                    """
                    #build CapitalLetter feature
                    capital = any(letter.isupper() for letter in cur_word)
                    if capital == True:
                        if (cur_word, cur_tag) not in set(self.feature_rep_dict["fCapital"].keys()):
                            self.feature_rep_dict["fCapital"][(cur_word,cur_tag)] = 1
                        else :
                            self.feature_rep_dict["fCapital"][(cur_word, cur_tag)] += 1

                    #build Number feature
                    number = any(letter.isdigit() for letter in cur_word)
                    if number == True:
                        if (cur_word, cur_tag) not in set(self.feature_rep_dict["fNumber"].keys()):
                            self.feature_rep_dict["fNumber"][(cur_word,cur_tag)] = 1
                        else :
                            self.feature_rep_dict["fNumber"][(cur_word, cur_tag)] += 1

                    """
                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)



class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object, from here we get histories, dictionary that count tag, word and pair(word, tag) and the feature_dict_list with f100
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # c'est un dictionaire dans lequel on met pour chaque features les index de celle qui apparaissent le plus de fois, (celle ou le nombre de fois est superioeur au Thresold
        self.feature_to_idx = {
            "f100": OrderedDict(),   #(CurrentWord, CurrentTag)
            "f101": OrderedDict(),   #(Prefix, CurrentTag)
            "f102": OrderedDict(),   #(Suffix, CurrentTag)
            "f103": OrderedDict(),   #(Tag(i-2), Tag(i-1), Tag(i))
            "f104": OrderedDict(),   #(Tag(i-1), Tag(i))
            "f105": OrderedDict(),   #(Tag(i))
            "f106": OrderedDict(),   #(word(i-1), Tag(i))
            "f107": OrderedDict()   #(word(i+1), Tag(i))
            #"fCapital" : OrderedDict(),
            #"fNumber"  : OrderedDict(),
        }
        self.represent_input_with_features = OrderedDict()    #
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx

        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]])\
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_word = history[0]
    c_tag = history[1]
    OneWordBack = history[2]
    OneTagBack = history[3]
    TwoTagBack = history[5]
    OneWordForward = history[6]
    features = []

    # f100
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    #f101 and f102
    if len(c_word) >=4:
        FourLetterPrefix = c_word[0:4]
        FourLetterSuffix = c_word[-4:]
        if (FourLetterPrefix, c_tag) in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][(FourLetterPrefix, c_tag)])
        if (FourLetterSuffix, c_tag) in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][(FourLetterSuffix, c_tag)])

    if len(c_word) >=3:
        ThreeLetterPrefix = c_word[0:3]
        ThreeLetterSuffix = c_word[-3:]
        if (ThreeLetterPrefix, c_tag) in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][(ThreeLetterPrefix, c_tag)])
        if (ThreeLetterSuffix, c_tag) in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][(ThreeLetterSuffix, c_tag)])

    if len(c_word) >=2:
        TwoLetterPrefix = c_word[0:2]
        TwoLetterSuffix = c_word[-2:]
        if (TwoLetterPrefix, c_tag) in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][(TwoLetterPrefix, c_tag)])
        if (TwoLetterSuffix, c_tag) in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][(TwoLetterSuffix, c_tag)])

    if len(c_word) >=1:
        OneLetterPrefix = c_word[0:1]
        OneLetterSuffix = c_word[-1:]
        if (OneLetterPrefix, c_tag) in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][(OneLetterPrefix, c_tag)])
        if (OneLetterSuffix, c_tag) in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][(OneLetterSuffix, c_tag)])



    #f103
    if (TwoTagBack, OneTagBack, c_tag) in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][(TwoTagBack, OneTagBack, c_tag)])

    #f104
    if (OneTagBack, c_tag) in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][(OneTagBack, c_tag)])

    # f105
    if (c_tag) in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][(c_tag)])

    # f106
    if (OneWordBack, c_tag) in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][(OneWordBack, c_tag)])

    # f107
    if (OneWordForward, c_tag) in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][(OneWordForward, c_tag)])
    """
    #fCapital
    if (c_word, c_tag) in dict_of_dicts["fCapital"]:
        features.append(dict_of_dicts["fCapital"][(c_word, c_tag)])

    #fNumber
    if (c_word, c_tag) in dict_of_dicts["fNumber"]:
        features.append(dict_of_dicts["fNumber"][(c_word, c_tag)])
    """
    return features


def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
