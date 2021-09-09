import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

def count_eer(scores, sames, left=-10, right=10):
    assert len(scores) == len(sames)
    n = len(scores)

    num_points = 200
    thresholds = np.linspace(left, right, num_points)
    frr = np.zeros(num_points)
    far = np.zeros(num_points)
    closest_dist = 1
    closest_j = -1

    for j in tqdm(range(num_points)):
        #print(f'\r{j}/{num_points}', end='')
        n_same = 0
        k_same = 0
        n_diff = 0
        k_diff = 0
        for i in range(n):
            score = scores[i]
            same = sames[i]
            if score > thresholds[j]:
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
        frr[j] = k_same/n_same
        far[j] = k_diff/n_diff
        if abs(frr[j] - far[j]) < closest_dist:
            closest_dist = abs(frr[j] - far[j])
            closest_j = j
        # print(frr[j], far[j])

    print(f'\rThreshold: {thresholds[closest_j]:1.2f},  '
          f'EER: {(frr[closest_j] + far[closest_j]) / 2 * 100 :1.2f},  '
          f'tol: {abs(frr[closest_j] - far[closest_j]) / 2 * 100 :1.2f}')

    return thresholds, frr, far, closest_j


if __name__ == '__main__':
    scores = np.load('lists/scores_for_eer.npy')
    sames = np.load('lists/sames_for_eer.npy')

    thresholds, frr, far, closest_j = count_eer(scores, sames) if len(sys.argv) < 2 else \
        count_eer(scores, sames, float(sys.argv[1]), float(sys.argv[2]))

    plt.plot(frr, far)
    plt.scatter(frr[closest_j], far[closest_j], s=10)
    plt.show()

    plt.plot(thresholds, frr)
    plt.plot(thresholds, far)
    plt.show()