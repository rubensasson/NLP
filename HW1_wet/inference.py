from preprocessing import read_test
from tqdm import tqdm
import numpy as np


def return_important_index(history, feature2id):
    important_index = []
    c_word = history[0]
    c_tag = history[1]
    OneWordBack = history[2]
    OneTagBack = history[3]
    TwoTagBack = history[5]
    OneWordForward = history[6]
    if (c_word, c_tag) in list(feature2id.feature_to_idx['f100'].keys()):
        important_index.append(feature2id.feature_to_idx['f100'][(c_word, c_tag)])

    for i in range(1, 5):
        if len(c_word) >= i:
            prefix = c_word[0:i]
            suffix = c_word[-i:]
            if (suffix, c_tag) in list(feature2id.feature_to_idx['f101'].keys()):
                important_index.append(feature2id.feature_to_idx['f101'][(suffix, c_tag)])
            if (prefix, c_tag) in list(feature2id.feature_to_idx['f102'].keys()):
                important_index.append(feature2id.feature_to_idx['f102'][(prefix, c_tag)])

    if (TwoTagBack, OneTagBack, c_tag) in list(feature2id.feature_to_idx['f103'].keys()):
        important_index.append(feature2id.feature_to_idx['f103'][(TwoTagBack, OneTagBack, c_tag)])

    if (OneTagBack, c_tag) in list(feature2id.feature_to_idx['f104'].keys()):
        important_index.append(feature2id.feature_to_idx['f104'][(OneTagBack, c_tag)])

    if (c_tag) in list(feature2id.feature_to_idx['f105'].keys()):
        important_index.append(feature2id.feature_to_idx['f105'][(c_tag)])

    if (OneWordBack, c_tag) in list(feature2id.feature_to_idx['f106'].keys()):
        important_index.append(feature2id.feature_to_idx['f106'][(OneWordBack, c_tag)])

    if (OneWordForward, c_tag) in list(feature2id.feature_to_idx['f107'].keys()):
        important_index.append(feature2id.feature_to_idx['f107'][(OneWordForward, c_tag)])

    """
    if (c_word,c_tag) in list(feature2id.feature_to_idx['fCapital'].keys()):
        important_index.append(feature2id.feature_to_idx['fCapital'][(c_word,c_tag)])

    if (c_word,c_tag) in list(feature2id.feature_to_idx['fNumber'].keys()):
        important_index.append(feature2id.feature_to_idx['fNumber'][(c_word,c_tag)])
    """

    return important_index


def compute_score(history, feature2id, pre_trained_weight):
    vector_of_idx = return_important_index(history, feature2id)
    score = 0
    for idx in vector_of_idx:
        score += pre_trained_weight[idx]
    return np.exp(score)


def compute_denom_firstK(feature2id, pre_trained_weight, all_tag, sentence):
    scores = []
    for tag in all_tag:
        h = (sentence[0], tag, '*', '*', '*', '*', sentence[1])
        scores.append(compute_score(h, feature2id, pre_trained_weight))
    return sum(scores)


# def give_probability(q, pi, k, possibleTag, tag_u):


def memm_viterbi(sentence, pre_trained_weights, feature2id, beam_size=2):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    all_tag = list(feature2id.feature_statistics.tags_counts.keys())
    nb_tag = len(all_tag)
    all_tag.append('*')

    possibleTags = [['*'], ['*']]

    pi = dict()
    pi[(-1, '*', '*')] = 1
    bp = dict()

    sentence = sentence[2:]
    n = len(sentence)
    print(sentence)
    for k in range(0, n - 1):
        print(k)
        score_per_round = {}
        if k == 0:
            denom = compute_denom_firstK(feature2id, pre_trained_weights, all_tag, sentence)
        for tag_u in possibleTags[k + 1]:
            for tag_v in all_tag:

                if k == 0:
                    possible_histories = [(sentence[k], tag_v, '*', '*', '*', '*', sentence[k + 1])]

                elif k == 1:
                    possible_histories = [(sentence[k], tag, sentence[k - 1], tag_v, '*', tag_u, sentence[k + 1]) for
                                          tag in all_tag]

                else:
                    possible_histories = [
                        (sentence[k], tag, sentence[k - 1], tag_v, sentence[k - 2], tag_u, sentence[k + 1]) for tag in
                        all_tag]


                if k == 0:
                    scores = [compute_score(h, feature2id, pre_trained_weights) for h in possible_histories]
                    q = [scores[i] / denom for i in range(0, len(scores))]
                    probabilities = [pi[(k - 1, possibleTags[k][i], tag_u)] * q[i] for i in range(len(possibleTags[k]))]
                    index_best_proba = all_tag.index(tag_v)
                    bp[(k, tag_u, tag_v)] = all_tag[index_best_proba]
                    pi[(k, tag_u, tag_v)] = probabilities[0]
                    print(k, tag_u, tag_v)

                else:
                    scores = [compute_score(h, feature2id, pre_trained_weights) for h in possible_histories]
                    q = [scores[i] / sum(scores) for i in range(0, len(scores))]
                    probabilities = [pi[(k - 1, possibleTags[k][i], tag_u)] * q[i] for i in range(len(possibleTags[k]))]
                    index_best_proba = np.argmax(probabilities)
                    bp[(k, tag_u, tag_v)] = possibleTags[k][index_best_proba]
                    pi[(k, tag_u, tag_v)] = probabilities[index_best_proba]

                score_per_round[tag_u, tag_v] = pi[(k, tag_u, tag_v)]

        next_possible_tags = set()
        curr_scores = sorted(score_per_round, key=(lambda key: score_per_round[key]), reverse=True)
        for (u, v) in curr_scores:
            next_possible_tags.add(v)
            if len(next_possible_tags) >= beam_size:
                break
        possibleTags.append(list(next_possible_tags))


    print(possibleTags)
    # searching for the last tag and the last former tag
    real_tags = ['*'] * n
    print(len(real_tags))
    max = 0.0
    for tag_u in possibleTags[n - 1]:
        for tag_v in possibleTags[n]:
            if pi[(n - 2, tag_u, tag_v)] > max:
                max = pi[(n - 2, tag_u, tag_v)]
                real_tags[-1] = tag_v
                if n > 1:
                    real_tags[-2] = tag_u
    # moving backward
    print(real_tags[-1], real_tags[-2])
    for i in range(n - 3, -1, -1):
        real_tags[k] = bp[(i + 2, real_tags[i + 1], real_tags[i + 2])]

    print(real_tags)

    # print(' we have the set of (k, u, v) = (' + str(k) + ', ' + str(tag_u) + ', ' + str(tag_v) + ') and possible histories = ' + str(possible_histories) + 'probabilities ' + str(
    # probabilities))
    # print('The best tag for the set of history is ' + str(bp[(k, tag_u, tag_v)]) + ' and the probability is  ' + str(pi[(k, tag_u, tag_v)]))


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()