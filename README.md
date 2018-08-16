# ParallelWork

Parallelization can be achieved by two methods in Python:

1) Data Parallelization
2) Task Parallelization

This library supports Data Parallelization (for 2d arrays only) via:
 * Splitting via Axes - Row wise or Column Wise
 * Splitting via Unique Values
 * Random Subset of Data
 * Splitting by user defined function


<hr></hr>
<br></br>


How to use this library ?

```{python}
import numpy as np
import parallelwork as pw

# write and task function
def test_func(data):
    return np.mean(data), np.sum(data)

# generate a huge array
arr = np.arange(1, 100000001, 1).reshape(20000000, 5)

# setup parallelize class for execution
pworker = pw.ParallelizeData()

# declare what function to execute
pworker.func = test_func

# provide data to the executer
pworker.execute(arr)

# get results after execution
print(pworker.results())

```


One can declare various fucntions on the data and output as follows:

```{python}
import parallelwork as pw

pworker = pw.ParallelizeData()

# Main task fucntion to execute on data
pworker.func = None

# Parameters to function can be passed as dictionary
pworker.func_params = {}

# Aggregate function to run on output
pworker.agg_func = None
pworker.agg_func_params = {}

# Arbitary text to print if function runs successfully
pworker.func_exec_text = None

# Arbitary splitting function. The output of this fucntion should be a list.
# Also it is not calibarated with number of processe hence results may vary.
pworker.split_func = None

# A function to exectue on splitted data once the split is done.
pworker.split_eval_func = None

```


To install you can do:
```
# to install via pip
pip install git+https://github.com/jkapila/ParallelWork.git

# or download the git and just do
python setup.py

```


**Note 1: The results are stored inside class.
So it may consume memory after execution.**

**Note 2: one should be very careful with dtypes as very huge data values
are not easyly pickalable and tends to cause errors.**

**For more example looks a examples.py !**


Next Steps:
1) Blog for usage methods and use cases
2) Make Async mapper parallel processing
3) Task parallelization with list
4) Data + Task Paralell processing
5) Sparse Matrices as input (Scipy and others)
6) Spark as Backend Support
7) Dask + Numba as Backend Support
8) Write Tests
9) Use manager for higher performance on heaviery functions
10) Native support for list and tuples


Dependency:
* Python 3
* NumPy
