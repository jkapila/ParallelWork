from __future__ import print_function, division

import numpy as np

import time
import multiprocessing


class Parallelize(object):

    def __init__(self, processes=-1, split_data=True, split_axis=0, split_by=None, timer=True, verbose=False):
        """

        :param processes: Number of Processes to run.(Default is -1 (uses all 2 * all cores max up to 8))
        :param split_data: Split data or not (Default is True). No splitting means no parallel processing.
        :param split_axis: Axis on which data should be made to split.(Default is 0 (row wise))
        :param split_by: Column number, on whose value data should be split. If given, this supersedes split_axis option.
        :param timer: To show executions time while processing. (Default is True)
        :param verbose: Be verbose. (Default is False)
        """

        if processes == -1:
            cpu_units = multiprocessing.cpu_count() * 2
            self.processes = cpu_units if cpu_units < 8 else 8
        else:
            self.processes = processes

        self.data = None
        self.split_data = split_data

        if split_by is not None:
            self.split_by = split_by
            self.split_axis = None
        else:
            self.split_by = None
            self.split_axis = split_axis

        self.data_list = []
        self.outputs = []

        self.timer = timer
        self.verbose = verbose

        # Alterable functions and values
        self.func = None
        self.func_params = {}
        self.agg_func = None
        self.agg_func_params = {}
        self.func_exec_text = None
        self.split_func = None
        self.split_eval_func = None

    def __validate_func(self):
        if self.func is None:
            raise NotImplementedError('Function on data cannot be blank!')
        else:
            pass

    def _data_splitter(self, data):

        if len(self.data_list) != 0:
            print('There are {} splits available. Removing them!'.format(len(self.data_list)))
            del self.data_list
            self.data_list = []

        if self.split_data:

            if self.split_func is None:

                if self.split_by is None:
                    self.data_list = [l for l in np.array_split(data, self.processes, axis=self.split_axis)]
                    assert len(self.data_list) == self.processes

                else:

                    if self.split_by < data.shape[1]:
                        unq = np.unique(data[:,self.split_by])

                        if len(unq) <= self.processes:
                            self.data_list = [data[np.where(data[:,self.split_by] == val),:] for val in unq]
                            assert len(self.data_list) <= self.processes
                        else:
                            print('This split by: {} is causing splits more than processes assigned!')
                            print('Splitting unique value in equal size by number of processes!')
                            for l in np.array_split(data[:,self.split_by],self.processes, axis=1):
                                dat_ = np.isin(data[:,self.split_by],l)
                                self.data_list.append(data[np.where(dat_), :])
                            assert len(self.data_list) == self.processes

            else:
                self.data_list = self.split_func(data)

            assert isinstance(self.data_list, list)

            for l in self.data_list:
                if self.verbose:
                    print('Split Shape: {}'.format(l.shape))
                if self.split_eval_func is not None:
                    print('Split Evaluation:')
                    self.split_eval_func(l)

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
        self.__validate_func()
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
