# Power and Effect Size Checkpoint


```python
# run this cell without changes

import numpy as np
import pandas as pd
from statsmodels.stats import power
from scipy import stats
```

#### 1. What is the relationship between power and the false-negative rate, $\beta$ ?

=== BEGIN MARK SCHEME ===

Power is simply 1 less the false-negative rate, 1 - beta.

=== END MARK SCHEME ===

#### 2. Calculate Cohen's *d* for the following data.  

Suppose we have the following samples of ant body lengths in centimeters, taken from two colonies that are distinct but spatially close to one another.

Calculate Cohen's *d* for these groups and assign it to the variable `d`.

(The variance of each group has been provided for you.)


```python
# run this cell without changes

colony1 = np.array([2.101, 2.302, 2.403])
colony2 = np.array([2.604, 2.505, 2.506])

colony1_var = np.var(colony1, ddof=1)
colony2_var = np.var(colony2, ddof=1)
```


```python
# Replace None with appropriate code
d = None
### BEGIN SOLUTION

n1, n2 = len(colony1), len(colony2)

pooled_stdev = np.sqrt(
    ((n1-1)*colony1_var + (n2-1)*colony2_var) 
    / 
    (n1 + n2 - 2)
)

d = (colony2.mean() - colony1.mean()) / pooled_stdev

def Cohen_d_1(group1, group2):
    """
    This function comes from the Effect Size lab
    """

    # Compute Cohen's d.

    # group1: Series or NumPy array
    # group2: Series or NumPy array

    # returns a floating point number 

    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    return d

def Cohen_d_2(group1, group2):
    """
    This function comes from Greg's lecture
    """

    """
    Computes Cohen's d.
    """
    
    # group1: Series or NumPy array
    # group2: Series or NumPy array

    # returns a floating point number 

    diff = group1.mean() - group2.mean()

    n1 = len(group1)
    n2 = len(group2)
    var1 = group1.var(ddof=1)
    var2 = group2.var(ddof=1)

    # Calculate the pooled variance
    pooled_var = ((n1-1) * var1 + (n2-1) * var2) / (n1 + n2 - 2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    return d
### END SOLUTION

print(d)
```

    2.3265865171551505



```python
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS

# d should be a floating point number
assert type(d) == float or type(d) == np.float64

### BEGIN HIDDEN TESTS

# Still give them points for negative d value
d = np.abs(d)

# Setup variables for tests
n1, n2 = len(colony1), len(colony2)
standardizer_1 = (n1 * colony1_var + n2 * colony2_var) / (n1 + n2)
standardizer_2 = ((n1-1) * colony1_var + (n2-1) * colony2_var) / (n1 + n2 - 2)

assert d == (colony2.mean() - colony1.mean()) / np.sqrt(standardizer_1) or \
       d == (colony2.mean() - colony1.mean()) / np.sqrt(standardizer_2)


### END HIDDEN TESTS
```

#### 3. Is this a large effect size? What conclusion should we draw about these two samples?

=== BEGIN MARK SCHEME ===

This is a huge effect size, but we have to be careful about drawing conclusions about the relationship between the two colonies (that the colonies contain distinct populations of ants, for example), since we have such small sample sizes.

=== END MARK SCHEME ===

#### 4. We decide we want to collect more data to have a more robust experiment. 

#### Given the current effect size, calculate how many observations we need (of each ant colony) if we want our test to have a power rating of 0.9 and a false-positive rate of 5%.


```python
# Replace None with appropriate code
observations_needed = None
### BEGIN SOLUTION
from test_scripts.test_class import Test
test = Test()

observations_needed = power.TTestIndPower().solve_power(effect_size=d,
                                  alpha=0.05,
                                 power=0.9)

test.save(observations_needed, "observations_needed")

### END SOLUTION
print(observations_needed)
```

    5.066906633232539



```python
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS

# No need to cast the result to an int. observations_needed should be a floating point number
assert type(observations_needed) == float or type(observations_needed) == np.float64

### BEGIN HIDDEN TESTS

from test_scripts.test_class import Test
test = Test()

test.run_test(observations_needed, "observations_needed")

### END HIDDEN TESTS
```

Suppose we gather more data on our ants and then re-visit our calculations. 


```python
# run this cell without changes

col1 = pd.read_csv('data/colony1', index_col=0)
col2 = pd.read_csv('data/colony2', index_col=0)
```


```python
# run this cell without changes
col1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.101</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.302</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.341</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.626</td>
    </tr>
  </tbody>
</table>
</div>




```python
# run this cell without changes
col2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col2_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.604000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.505000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.506000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.383960</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.578711</td>
    </tr>
  </tbody>
</table>
</div>



#### 5. Do a two-tailed t-test on the lengths of ants in these two colonies.


```python
### BEGIN SOLUTION

stats.ttest_ind(col1['col1_length'], col2['col2_length'])

### END SOLUTION
```

What should we conclude about these two collections of ant body lengths?

=== BEGIN MARK SCHEME ===

The p-value is now quite large, and so we cannot reject the null hypothesis that says that there is no difference between the body lengths in the two ant colonies.

=== END MARK SCHEME ===
