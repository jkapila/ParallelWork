"""

This is example how to use function and pass arguments ot it.

"""

import numpy as np
import parallelwork as pw

import sys


if __name__ == '__main__':


    def test_func(data):
        return np.mean(data), np.sum(data)

    def test_func_withparam(data, axis):
        print('Axis: {}'.format(axis))
        return np.mean(data, axis=axis), np.sum(data, axis=axis)


    arr = np.random.rand(2000000, 50).astype('float16')

    print('Executing Parallel process without parameters to function with splits!')
    parallelizer = pw.ParallelizeData()
    parallelizer.func = test_func
    parallelizer.func_params = {}
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process without parameters to function without splits!')
    parallelizer = pw.ParallelizeData(split_data=False)
    parallelizer.func = test_func
    parallelizer.func_params = {}
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process with parameters to function with splits!')
    parallelizer = pw.ParallelizeData()
    parallelizer.func = test_func_withparam
    parallelizer.func_params = {'axis': 0}
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process with parameters to function without splits!')
    parallelizer = pw.ParallelizeData(split_data=False)
    parallelizer.func = test_func_withparam
    parallelizer.func_params = {'axis': 0}
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process with parameters to function with splits on Axis 1!')
    parallelizer = pw.ParallelizeData(split_data=True, split_axis=1, processes=4)
    parallelizer.func = test_func_withparam
    parallelizer.func_params = {'axis': 0}
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process with random splits!')
    parallelizer = pw.ParallelizeData(random_split=True, verbose=True)
    parallelizer.func = test_func_withparam
    parallelizer.func_params = {'axis': 1}
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())
