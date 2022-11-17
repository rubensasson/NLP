import numpy as np

from model_instantation import Y_viterbi


"""
Calcul le score de choisir un tag pour une history avec l'aide du vecteur v 
"""
def exp_scores(v, history, feat_func, HASH_TABLE, Y):
    correct_ind = [[int(HASH_TABLE[i]) for i in feat_func(history, tag) if HASH_TABLE[i] > 0] for tag in Y]  #retourne les index de la ou on doit mettre les 1 dans le vecteur binaire pour tout ls tags
    norm_scores = [v[ind].sum() for ind in correct_ind]  #on fait la somme des coeff de v seulement pour les index concernés
    return np.exp(norm_scores)   #on le met sous exponenteil, il servira de dénominateur pour le calcul du score


def viterbi(sentence, opt_v, feat_func, HASH_TABLE, beam_size=2, tagged=False):
    active_tags = list() 
    active_tags.append([len(Y_viterbi) - 1])  # index of '*' 
    active_tags.append([len(Y_viterbi) - 1])  # index of '*' 
    corpus = sentence.split()  #on sépare les phrases dy test set, ils n'ont pas encore de tag, on assigne les tags * au mots w(i-2) et W(i-1)

    if tagged:
        words = [w_t.split('_')[0] for w_t in corpus] 
        true_tags = [w_t.split('_')[1] for w_t in corpus] 
    else:
        words = corpus 

    pi_viterbi = dict()   #on crée un dictionaire qui estime pour chaque mots de la phrase toutes les probabilité d'avoir tel tag pour tel mot
    pi_viterbi[(-1, len(Y_viterbi) - 1, len(Y_viterbi) - 1)] = 1 #on associe le tag * au 2premiers (faux) word de la sentence
    bp = dict() #
    word_tag_len = len(corpus) 

    # Forward step - We run on the best B active tags only for u and t - defined several iterations before.
    for k, word in enumerate(words):  #on passe sur toutes les phrases du data set
        curr_scores = {}     #on ouvre un dictionaire des scores
        for u in active_tags[k + 1]: #on passe sur les tag actif, au debut il n'y a que '*' dans les tags actifs
            for v in range(len(Y_viterbi)): #pour tout les tags possible
                if k == 0:
                    histories = [('*', '*', word, '*', '*', Y_viterbi[v], k)] #on recrée une histoire en assignant les tag * * au deux faux premier mots
                elif k == 1: #pareil que en haut
                    histories = [('*', words[k-1], word,  Y_viterbi[t], Y_viterbi[u], Y_viterbi[v], k) for t in active_tags[k]]
                else: #on reconstruit toutes les histoires possible avec les précedent mots (k-2), et on donne tous les tags possible a Y(i-2),
                    histories = [(words[k - 2], words[k - 1], words[k], Y_viterbi[t], Y_viterbi[u], Y_viterbi[v], k) for t in active_tags[k]]
                exp_scs = [exp_scores(opt_v, h, feat_func, HASH_TABLE, Y_viterbi) for h in histories]  #on calcul laquelle des histoiry a eu le meilleur score pour tous les tags possible qu'on a donner a Y(i-2)
                norms = [exp_scs[t].sum() for t in range(len(active_tags[k]))]

                # Computing probabilities for all t's
                probs = [pi_viterbi[(k - 1, active_tags[k][t], u)] * exp_scs[t][v] / norms[t] for t in range(len(active_tags[k]))]  #nous retourne la probabilité pour chaque tag qu'il soit le bon ou pas
                best_ind = np.argmax(probs)  #on prend l'indexe du tag ayant eu la proba maximum
                bp[(k, u, v)] = active_tags[k][best_ind] #pour le mot de la phrase numero k, on lui assigne l'index du meilleur tag pour lui
                pi_viterbi[(k, u, v)] = probs[best_ind]  #probabilité d'avoir la sequence de tag qui fini par les tag u et v a la position k
                curr_scores[(u, v)] = pi_viterbi[(k, u, v)]
        next_active_tags = set()

        # Ranking scores and insert the best B tags in active_tags[k+2]
        curr_scores = sorted(curr_scores, key=(lambda key: curr_scores[key]), reverse=True)
        for elem in curr_scores:
            next_active_tags.add(elem[1])
            if len(next_active_tags) >= beam_size:
                break
        active_tags.append(list(next_active_tags))

    # Finding t_n and t_n-1 - Backward step. Searching only in active_tags[k]
    tags = np.zeros(len(words))
    best_last_pi = 0.0
    for u in active_tags[word_tag_len]:
        for v in active_tags[word_tag_len + 1]:
            if pi_viterbi[(word_tag_len - 1, u, v)] > best_last_pi:
                best_last_pi = pi_viterbi[(word_tag_len - 1, u, v)]
                tags[-1] = v
                try:
                    tags[-2] = u
                except IndexError:
                    pass

    for k in range(word_tag_len - 3, -1, -1):
        tags[k] = bp[(k + 2, tags[k + 1], tags[k + 2])]

    if tagged:
        return words, [Y_viterbi[i] for i in [int(i) for i in tags]], true_tags
    else:
        return words, [Y_viterbi[i] for i in [int(i) for i in tags]]


