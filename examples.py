"""

This is example how to use function and pass arguments ot it.

"""

import numpy as np
import parallelwork as pw


if __name__ == '__main__':
    def test_func(data):
        return np.mean(data), np.sum(data)


    def test_func_withparam(data, axis):
        print('Axis: {}'.format(axis))
        return np.mean(data, axis=axis), np.sum(data, axis=axis)


    arr = np.arange(1, 100000001, 1).reshape(20000000, 5)

    print('Executing Parallel process without parameters to function with splits!')
    parallelizer = pw.Parallelize()
    parallelizer.func = test_func
    parallelizer.func_params = {}
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process without parameters to function without splits!')
    parallelizer = pw.Parallelize(split_data=False)
    parallelizer.func = test_func
    parallelizer.func_params = {}
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process with parameters to function with splits!')
    parallelizer = pw.Parallelize()
    parallelizer.func = test_func_withparam
    parallelizer.func_params = {'axis': 0}
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process with parameters to function without splits!')
    parallelizer = pw.Parallelize(split_data=False)
    parallelizer.func = test_func_withparam
    parallelizer.func_params = {'axis': 0}
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())

    print('\nExecuting Parallel process with parameters to function with splits on Axis 1!')
    parallelizer = pw.Parallelize(split_data=True, split_axis=1, processes=4)
    parallelizer.func = test_func_withparam
    parallelizer.func_params = {'axis': 0}
    parallelizer.execute(arr)
    print('Results:')
    print(parallelizer.results())
