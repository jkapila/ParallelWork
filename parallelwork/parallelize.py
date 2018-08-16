from __future__ import print_function, division

import numpy as np

import time
import multiprocessing


class ParallelizeData(object):

    def __init__(self, processes=-1, split_data=True, split_axis=0, split_by=None, keep_data=False,
                 random_split=False, random_split_by_row=None, random_split_by_col=None, keep_col=None,
                 timer=True, verbose=False):
        """

        The Parallelize data
        :param processes: Number of Processes to run.(Default is -1 (uses all 2 * all cores max up to 8))
        :param split_data: Split data or not (Default is True). No splitting means no parallel processing.
        :param split_axis: Axis on which data should be made to split.(Default is 0 (row wise))
        :param split_by: Column number, on whose value data should be split.
                            If given, this supersedes split_axis option.
        :param keep_data: To keep data on execute (Default is False)
        :param random_split: To make a random split (Default is False)
        :param random_split_by_row: Percentage of rows to keep in random split. Values should be greater than 0.
                                (Defaults to 0.75 if random_split is True)
        :param random_split_by_col: Percentage of columns to keep in random split. Values should be greater than 0.
                        (Defaults to 0.8 if random_split is True)
        :param keep_col: Column to keep in each split. Should be list of indexes. Not used when split_by is given.
        :param timer: To show executions time while processing. (Default is True)
        :param verbose: Be verbose. (Default is False)
        """

        # Cpu process to be used
        if processes == -1:
            cpu_units = multiprocessing.cpu_count() * 2
            self.processes = cpu_units if cpu_units < 8 else 8
        else:
            self.processes = processes

        self.split_data = split_data

        if split_by is not None:
            self.split_by = split_by
            self.split_axis = None
        else:
            self.split_by = None
            self.split_axis = split_axis

        # Random Splitting
        self.random_split = random_split
        if random_split_by_col is None:
            self.random_split_by_col = 0.8
        else:
            self.random_split_by_col = random_split_by_col

        if random_split_by_row is None:
            self.random_split_by_row = 0.75
        else:
            self.random_split_by_row = random_split_by_row

        # todo: assert and restructure random splitting criteria

        # Keeping specific columns it all splits
        if keep_col is None:
            self.keep_col = []
        else:
            assert isinstance(keep_col, list)
            self.keep_col = keep_col

        # keeping data with the object
        self.keep_data = keep_data

        # Generic behaviour
        self.timer = timer
        self.verbose = verbose

        # Data variables
        self.data = None
        self.data_list = []
        self.outputs = []

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

    def __random_splitter(self, data):
        splits = self.processes
        col_to_keep = self.keep_col
        data_list = []
        n, m = data.shape

        for i in range(splits):

            if self.random_split_by_row >= 1.0:
                rand_ind = np.random.choice(n, np.ceil(self.random_split_by_row * n).astype('int32'),
                                            replace=True)
            elif 1.0 > self.random_split_by_row > 0.0:
                rand_ind = np.random.choice(n, np.ceil(self.random_split_by_row * n).astype('int32'), replace=False)
            else:
                raise ValueError('Random Split cannot have -ve values!')

            if self.random_split_by_col >= 1.0:
                rand_col = np.random.choice(m,
                                            np.ceil(self.random_split_by_col * (m - len(col_to_keep))).astype('int32'),
                                            replace=True).tolist()
            elif 1.0 > self.random_split_by_col > 0.0:
                rand_col = np.random.choice(m,
                                            np.ceil(self.random_split_by_col * (m - len(col_to_keep))).astype('int32'),
                                            replace=False).tolist()
            else:
                raise ValueError('Random Split cannot have -ve values!')

            rand_col.extend(col_to_keep)
            rand_col = sorted(rand_col, key=lambda x: x)
            t = time.time()
            dd = data[rand_ind, :]
            dd = dd[:, rand_col]
            if self.timer:
                print('Time taken to assing random split is :{:>5.6f}'.format(time.time() - t))
            data_list.append(dd)

        assert len(data_list) == splits

        if self.verbose:
            for l in data_list:
                print('Random Subset Shape: {}'.format(l.shape))

        return data_list

    def __data_value_splitter(self, data):
        splits = self.processes
        split_by = self.split_by
        data_list = []
        if split_by < data.shape[1]:
            unq = np.unique(data[:, split_by])

            if len(unq) <= splits:
                data_list = [data[np.where(data[:, split_by] == val), :] for val in unq]
                assert len(data_list) <= splits
            else:
                print('This split by: {} is causing splits more than processes assigned!')
                print('Splitting unique value in equal size by number of processes!')
                for l in np.array_split(data[:, split_by], splits, axis=1):
                    dat_ = np.isin(data[:, split_by], l)
                    data_list.append(data[np.where(dat_), :])
                assert len(data_list) == splits

        if self.verbose:
            for l in data_list:
                print('Split Shape: {}'.format(l.shape))

        return data_list

    def __generic_splitter(self, data):

        splits = self.processes
        data_list = []
        col_to_keep = self.keep_col

        if self.split_axis == 1 and len(col_to_keep) != 0:
            n, m = data.shape
            col_to_split = [l for l in range(m) if l not in col_to_keep]
            dd = data[:, col_to_split]
            kept_col = data[:, col_to_keep]

            for l in np.array_split(dd, splits, axis=1):
                data_list.append(np.concatenate((l, kept_col), axis=1))

        elif len(col_to_keep) == 0:
            data_list = [l for l in np.array_split(data, splits, axis=self.split_axis)]

        assert len(data_list) == splits

        if self.verbose:
            for l in data_list:
                print('Split Shape: {}'.format(l.shape))

        return data_list

    def _data_splitter(self, data):

        if self.keep_data:
            self.data = data

        if len(self.data_list) != 0:
            print('There are {} splits available. Removing them!'.format(len(self.data_list)))
            del self.data_list
            self.data_list = []

        if self.split_data:

            if self.split_func is None and not self.random_split:

                if self.split_by is None:
                    self.data_list = self.__generic_splitter(data)

                else:
                    self.data_list = self.__data_value_splitter(data)

            elif self.random_split and self.split_func is None:

                self.data_list = self.__random_splitter(data)

            elif self.split_func is not None:
                self.data_list = self.split_func(data)

            else:
                pass

            assert isinstance(self.data_list, list)

            for l in self.data_list:
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

        if self.verbose:
            print('Total {} data splits to be processed.'.format(len(self.data_list)))

        try:
            pool = multiprocessing.Pool(processes=self.processes, maxtasksperchild=4)
            for op in pool.imap(self._processor, self.data_list, chunksize=100000):
                self.outputs.append(op)

            if self.verbose:
                print('Active children count: %d ' % len(multiprocessing.active_children()))

            pool.close()
            pool.join()
        except Exception as e:
            print('Unable to process data in parallel.\nCaught Exception {}'.format(e))

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
