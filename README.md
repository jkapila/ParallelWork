# ParallelWork

Package for doing parallelization by two methods in Python:

1) Data Parallelization
2) Task Parallelization

Data Parallelization is supported (for 2d arrays only) via:
 * Splitting via Axes - Row wise or Column Wise
 * Splitting via Unique Values

Task Parallelization is in WIP.

<hr></hr>


How to use this library ?

```{python}
import numpy as np
from parallelwork import Parallelize

# write and task function
def test_func(data):
    return np.mean(data), np.sum(data)

# generate a huge array
arr = np.arange(1, 100000001, 1).reshape(20000000, 5)

# setup parallelize class for execution
parallelizer = Parallelize(func=test_func, func_params={}, processes=-1)

# provide data to the executer
parallelizer.execute(arr)

# get results after execution
print(parallelizer.results())

```

**Note: The results are stored inside class. So it may consume memory after execution.**