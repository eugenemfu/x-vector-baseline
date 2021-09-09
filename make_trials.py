import numpy as np
import pickle
from tqdm import tqdm
import sys
from count_eer import count_eer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--plda",
    type=str,
    help="Path to PLDA pkl model",
    required=True,
)
parser.add_argument(
    "-e",
    "--embeddings",
    type=str,
    help="Path to embeddings .npz file",
    required=True,
)
parser.add_argument(
    "-s",
    "--skip",
    type=float,
    help="Skip rate",
    required=False,
    default=1.0,
)
args = parser.parse_args()

SEED = 1000
np.random.seed(SEED)

skip_rate = args.skip

SNORM = False
cohort_size = 500
TOP = 200

mean3eer = False
enroll_size = 3

# y_test = np.load(args.labels)  # '../mfcc/voxceleb2_test/speakers.npy'
# y_test = y_test[:y_test.shape[0]]
embeddings_test, y_test = np.load(args.embeddings).values()  # 'lists/embeddings_voxceleb2_test_53.npy'
test_size = y_test.shape[0]
# print(test_size)
# for i in range(test_size):
#     print(y_test[i], end=' ')
# print()

if SNORM:
    embeddings_train, y_train = np.load('lists/embeddings_mix_val_mix35.npz').values()
    train_size = y_train.shape[0]
    # print(train_size)
    cohort_idx = np.random.choice(train_size, size=cohort_size, replace=False)
    embeddings_cohort = embeddings_train[cohort_idx]

with open(args.plda, 'rb') as f:  # 'models/plda_53.pkl'
    lda, plda = pickle.load(f)

if SNORM:
    embeddings_cohort_reduced = lda.transform(embeddings_cohort)
    transformed_cohort = plda.transform(embeddings_cohort_reduced, from_space='D', to_space='U_model')

embeddings_test_reduced = lda.transform(embeddings_test)
transformed_test = plda.transform(embeddings_test_reduced, from_space='D', to_space='U_model')

enroll_idx = np.zeros(0, dtype=int)
verify_idx = np.zeros(0, dtype=int)
if mean3eer:
    last_class = 0
    last_class_start_point = 0
    for i in range(test_size + 1):
        if i == test_size or y_test[i] != last_class:
            class_size = i - last_class_start_point
            rand_idx = last_class_start_point + \
                       np.random.choice(class_size, size=class_size, replace=False)
            enroll_idx = np.concatenate((enroll_idx, rand_idx[:enroll_size]))
            verify_idx = np.concatenate((verify_idx, rand_idx[enroll_size:]))
            if i != test_size:
                last_class = y_test[i]
                last_class_start_point = i


threshold = 0 #-0.55
n_same = 0
k_same = 0
n_diff = 0
k_diff = 0
ITERATIONS = 1000000

scores = []
sames = []

#f = open('lists/trials.txt', 'w')


for enroll_pos in tqdm(range(len(enroll_idx) // 3 if mean3eer else test_size)):
    if mean3eer:
        enroll_i = enroll_idx[enroll_pos * 3: enroll_pos * 3 + 3]
        y_enroll = y_test[enroll_i]
        assert y_enroll[0] == y_enroll[1] == y_enroll[2]
        y_enroll = y_enroll[0]
    else:
        enroll_i = enroll_pos
        y_enroll = y_test[enroll_i]

    for verify_i in (verify_idx if mean3eer else range(enroll_i + 1, test_size)):
        skip = np.random.random()
        if skip > skip_rate:
            continue
        same = (y_enroll == y_test[verify_i])
        enroll_emb = np.mean(transformed_test[enroll_i], axis=0) if mean3eer \
            else transformed_test[enroll_i]
        verify_emb = transformed_test[verify_i]
        score = plda.calc_same_diff_log_likelihood_ratio(enroll_emb[None, :], verify_emb[None, :])

        if SNORM:
            cohort_scores_enroll = np.array([plda.calc_same_diff_log_likelihood_ratio(
                enroll_emb[None, :], emb[None, :])
                for emb in transformed_cohort])
            cohort_scores_enroll = cohort_scores_enroll[
                np.argpartition(cohort_scores_enroll, cohort_size - TOP)[-TOP:]]
            cohort_scores_verify = np.array([plda.calc_same_diff_log_likelihood_ratio(
                verify_emb[None, :], emb[None, :])
                for emb in transformed_cohort])
            cohort_scores_verify = cohort_scores_verify[
                np.argpartition(cohort_scores_verify, cohort_size - TOP)[-TOP:]]
            #print(cohort_scores_enroll)
            mean_0 = np.mean(cohort_scores_enroll)
            mean_1 = np.mean(cohort_scores_verify)
            std_0 = np.std(cohort_scores_enroll)
            std_1 = np.std(cohort_scores_verify)
            #print(score, mean_0, mean_1, std_0, std_1)
            score = 0.5 * ((score - mean_0) / std_0 + (score - mean_1) / std_1)

        if score > threshold:
            if same:
                n_same += 1
            else:
                k_diff += 1
                n_diff += 1
        else:
            if same:
                k_same += 1
                n_same += 1
            else:
                n_diff += 1

        #f.write(f'{y_test[enroll_i]} {y_test[verify_i]} {same}\n')
        scores.append(score)
        sames.append(same)

    #print(f'\r{enroll_pos + 1} speakers done')

#f.close()

np.save(f'lists/scores_for_eer.npy', np.array(scores))
np.save(f'lists/sames_for_eer.npy', np.array(sames, dtype=bool))

count_eer(scores, sames)

print(f'Total same pairs: {n_same:<8}  err rate: {k_same / n_same}')
print(f'Total diff pairs: {n_diff:<8}  err rate: {k_diff / n_diff}')