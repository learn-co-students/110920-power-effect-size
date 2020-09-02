# Power and Effect Size Checkpoint


```python
#run this cell without changes

import numpy as np
import pandas as pd
from statsmodels.stats import power
from scipy import stats
```

#### 1. What is the relationship between power and the false-negative rate, $\beta$ ?

=== BEGIN MARK SCHEME ===


'''
Power is simply 1 less the false-negative rate, 1 - beta.
'''

=== END MARK SCHEME ===

#### 2. Calculate Cohen's *d* for the following data.  

Suppose we have the following samples of ant body lengths in centimeters, taken from two colonies that are distinct but spatially close to one another.

Calculate Cohen's *d* for these groups.

(The variance of each group has been provided for you.)


```python
#run this cell without changes

group1 = np.array([2.101, 2.302, 2.403])
group2 = np.array([2.604, 2.505, 2.506])

group1_var = np.var(group1, ddof=1)
group2_var = np.var(group2, ddof=1)
```


```python
### BEGIN SOLUTION


from test_scripts.test_class import Test
test = Test()


pooled_stdev = np.sqrt(
    (2*group1_var + 2*group2_var) 
    / 
    4
)

d = (group2.mean() - group1.mean()) / pooled_stdev
d

test.save()



### END SOLUTION
```




    2.3265865171551505




```python
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS

### BEGIN HIDDEN TESTS

from test_scripts.test_class import Test
test = Test()

test.run_test()


### END HIDDEN TESTS
```

#### 3. Is this a large effect size? What conclusion should we draw about these two samples?

=== BEGIN MARK SCHEME ===


'''
This is a huge effect size, but we have to be careful about drawing
conclusions about the relationship between the two colonies (that
the colonies contain distinct populations of ants, for example), since
we have such small sample sizes.
'''

=== END MARK SCHEME ===

#### 4. We decide we want to collect more data to have a more robust experiment. 

#### Given the current effect size, calculate how many observations we need (of each ant colony) if we want our test to have a power rating of 0.9 and a false-positive rate of 5%.


```python
### BEGIN SOLUTION


from test_scripts.test_class import Test
test = Test()


power.TTestIndPower().solve_power(effect_size=d,
                                  alpha=0.05,
                                 power=0.9)

test.save()



### END SOLUTION
```




    5.066906633232539




```python
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS

### BEGIN HIDDEN TESTS

from test_scripts.test_class import Test
test = Test()

test.run_test()


### END HIDDEN TESTS
```

Suppose we gather more data on our ants and then re-visit our calculations. 


```python
#run this cell without changes

col1 = pd.read_csv('data/colony1')
col2 = pd.read_csv('data/colony2')
```

#### 5. Do a two-tailed t-test on the lengths of ants in these two colonies.


```python
### BEGIN SOLUTION


from test_scripts.test_class import Test
test = Test()


stats.ttest_ind(col1['col1_length'], col2['col2_length'])

test.save()



### END SOLUTION
```




    Ttest_indResult(statistic=-0.6607977673021191, pvalue=0.5088181960071383)




```python
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS

### BEGIN HIDDEN TESTS

from test_scripts.test_class import Test
test = Test()

test.run_test()


### END HIDDEN TESTS
```

#### 6. What should we conclude about these two collections of ant body lengths?

=== BEGIN MARK SCHEME ===


'''
The p-value is now quite large, and so we cannot reject the null
hypothesis that says that there is no difference between the body
lengths in the two ant colonies.
'''

=== END MARK SCHEME ===
