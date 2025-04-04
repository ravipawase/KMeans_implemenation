import numpy as np
from timeit import timeit
from timeit import Timer


def calculate_distance(point1, point2):
    """

    :param point1:
    :param point2:
    :return:
    """

    # check point1 and point2 are 1d arrays and have same length
    if not isinstance(point1, np.ndarray) and isinstance(point2, np.ndarray) and point1.ndim == 1 and point2.ndim == 1 \
            and (len(point1) == len(point2)):
        raise ValueError('Expected 1d numpy arrays of same length')

    distance = 0
    for _ in range(len(point1)):
        distance += np.square(point1[_] - point2[_])
    distance = np.sqrt(distance)
    return distance

def calculate_distance_list_comprehension(point1, point2):
    """

    :param point1:
    :param point2:
    :return:
    """

    # check point1 and point2 are 1d arrays and have same length
    if not isinstance(point1, np.ndarray) and isinstance(point2, np.ndarray) and point1.ndim == 1 and point2.ndim == 1 \
            and (len(point1) == len(point2)):
        raise ValueError('Expected 1d numpy arrays of same length')

    distance = [np.square(point1[_] - point2[_]) for _ in range(len(point1))]
    distance = np.sqrt(np.sum(distance))
    return distance


if __name__ == '__main__':
    from timeit import Timer

    t = Timer(lambda: calculate_distance(np.random.random(4), np.random.random(4)))
    print(t.timeit(100000))
    t = Timer(lambda: calculate_distance_list_comprehension(np.random.random(4), np.random.random(4)))
    print(t.timeit(100000))
