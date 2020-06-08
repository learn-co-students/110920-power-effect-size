# Power and Effect Size Checkpoint


```python
#run this cell without changes

import numpy as np
import pandas as pd
from statsmodels.stats import power
from scipy import stats
```

#### 1. What is the relationship between power and the false-negative rate, $\beta$ ?


```python

```


```python
# __SOLUTION__

# Power is simply 1 less the false-negative rate, 1 - beta.
```

#### 2. Calculate Cohen's *d* for the following data.  

Suppose we have the following samples of ant body lengths in centimeters, taken from two colonies that are distinct but spatially close to one another.

Calculate Cohen's *d* for these groups.

(The variance of each group has been provided for you.)


```python
group1 = np.array([2.101, 2.302, 2.403])
group2 = np.array([2.604, 2.505, 2.506])

group1_var = np.var(group1, ddof=1)
group2_var = np.var(group2, ddof=1)
```


```python
# __SOLUTION__

pooled_stdev = np.sqrt(
    (2*group1_var + 2*group2_var) 
    / 
    4
)

d = (group2.mean() - group1.mean()) / pooled_stdev
d
```




    2.3265865171551505



#### 3. Is this a large effect size? What conclusion should we draw about these two samples?


```python

```


```python
# __SOLUTION__

# This is a huge effect size, but we have to be careful about drawing
# conclusions about the relationship between the two colonies (that
# the colonies contain distinct populations of ants, for example), since
# we have such small sample sizes.
```

#### 4. We decide we want to collect more data to have a more robust experiment. 

#### Given the current effect size, calculate how many observations we need (of each ant colony) if we want our test to have a power rating of 0.9 and a false-positive rate of 5%.


```python

```


```python
# __SOLUTION__

power.TTestIndPower().solve_power(effect_size=d,
                                  alpha=0.05,
                                 power=0.9)
```




    5.066906633232539



Suppose we gather more data on our ants and then re-visit our calculations. 


```python
#run this cell without changes

col1 = pd.read_csv('data/colony1')
col2 = pd.read_csv('data/colony2')
```

#### 5. Do a two-tailed t-test on the lengths of ants in these two colonies.


```python

```


```python
# __SOLUTION__

stats.ttest_ind(col1['col1_length'], col2['col2_length'])
```




    Ttest_indResult(statistic=-0.6607977673021191, pvalue=0.5088181960071383)



#### 6. What should we conclude about these two collections of ant body lengths?


```python

```


```python
# __SOLUTION__

# The p-value is now quite large, and so we cannot reject the null
# hypothesis that says that there is no difference between the body
# lengths in the two ant colonies.
```
