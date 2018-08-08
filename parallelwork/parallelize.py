from __future__ import print_function, division

import numpy as np

import time
import multiprocessing


class Parallelize(object):

    def __init__(self, func, func_params, processes=-1, split_data=True, agg_func=None, agg_func_params=None,
                 func_exec_text=None, split_eval_func=None, timer=True, verbose=False):

        self.func = func
        self.func_params = func_params
        if processes == -1:
            cpu_units = multiprocessing.cpu_count() * 2
            self.processes = cpu_units if cpu_units < 8 else 8
        else:
            self.processes = processes

        self.data = None
        self.split_data = split_data
        self.data_list = []
        self.outputs = []

        self.agg_func = agg_func
        self.agg_func_params = agg_func_params

        self.func_exec_text = func_exec_text
        self.split_eval_func = split_eval_func
        self.timer = timer
        self.verbose = verbose

    def _data_splitter(self, data):

        if len(self.data_list) != 0:
            print('There are {} splits available. Removing them!'.format(len(self.data_list)))
            del self.data_list
            self.data_list = []

        if self.split_data:

            for l in np.array_split(data, self.processes):
                if self.verbose:
                    print('Split Shape: {}'.format(l.shape))
                if self.split_eval_func is not None:
                    print('Split Evaluation:')
                    self.split_eval_func(l)

                self.data_list.append(l)

            assert len(self.data_list) == self.processes

        else:

            self.data_list = [data]

            assert len(self.data_list) == 1

    def _processor(self, sub_data):

        output = self.func(sub_data, **self.func_params)
        if self.func_exec_text is not None:
            print(self.func_exec_text)
        else:
            if self.verbose:
                print('Function Executed!')
        return output

    def execute(self, data):

        t = time.time()
        self._data_splitter(data)
        if self.timer and self.split_data:
            print("Time to Split data {:>5.4f}!".format(time.time() - t))
            t = time.time()

        pool = multiprocessing.Pool(processes=self.processes, maxtasksperchild=4)
        self.outputs = pool.map(self._processor, self.data_list, chunksize=100000)

        if self.verbose:
            print('Active children count: %d ' % len(multiprocessing.active_children()))

        pool.close()
        pool.join()
        if self.timer:
            print('Time to execute function: {:>5.4f}!'.format(time.time() - t))

        if self.verbose:
            print('Execution Completed Successfully!\n')

    def results(self, aggregate=False):
        if aggregate:
            if self.agg_func is not None:

                return self.agg_func(self.outputs, **self.agg_func_params)
            else:
                print('No aggregation function found! Returning output as it is!')
                pass

        return self.outputs


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
