"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import numpy as np


def create_train_dataset():
    ############## Task 1
    n_train = 100000
    max_train_card = 10
    X_train = []
    y_train = []

    for _ in range(n_train):
        card = np.random.randint(1, max_train_card + 1)
        x = list(np.random.randint(1, 10 + 1, card))
        y = np.sum(x)
        x = [0 for _ in range(max_train_card - card)] + x
        X_train.append(x)
        y_train.append(y)

    return np.array(X_train), np.array(y_train)


def create_test_dataset():
    ############## Task 2
    min_test_card = 5
    max_test_card = 100
    increments = 5
    n_by_increment = 10000
    X_test = []
    y_test = []
    for card in range(min_test_card, max_test_card + 1, increments):
        partial_x = []
        partial_y = []
        for _ in range(n_by_increment):
            x = list(np.random.randint(1, 10 + 1, card))
            y = np.sum(x)
            partial_x.append(x)
            partial_y.append(y)
        X_test.append(np.array(partial_x))
        y_test.append(np.array(partial_y))

    return X_test, y_test
