"""

This is example how to use function and pass arguments ot it.

"""

import numpy as np
from parallelwork.parallelize import Parallelize


if __name__ == '__main__':
    def test_func(data):
        return np.mean(data), np.sum(data)


    def test_func_withparam(data, axis):
        print('Axis: {}'.format(axis))
        return np.mean(data, axis=axis), np.sum(data, axis=axis)


    arr = np.arange(1, 100000001, 1).reshape(20000000, 5)

    print('Executing Parallel process without parameters to function with splits!')
    parallelizer = Parallelize(func=test_func, func_params={}, processes=-1)
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process without parameters to function without splits!')
    parallelizer = Parallelize(func=test_func, func_params={}, split_data=False, processes=-1)
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process with parameters to function with splits!')
    parallelizer = Parallelize(func=test_func_withparam, func_params={'axis': 0}, processes=-1)
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process with parameters to function without splits!')
    parallelizer = Parallelize(func=test_func_withparam, func_params={'axis': 0}, split_data=False, processes=-1)
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())
