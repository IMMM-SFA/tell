import random
import pandas as pd

# Ensure reproducibility
random.seed(123)

# Your example data
name = ['RI', 'NH', 'MA', 'RI', 'NH', 'MA','RI', 'NH', 'MA','RI', 'NH', 'MA']
year = [2015, 2015, 2015, 2016, 2016, 2016, 2017, 2017, 2017, 2018, 2018, 2018]
population = random.sample(range(10000, 300000), 12)

# Build DataFrame
df = pd.DataFrame({'name': name,
                   'year': pd.to_datetime(year, format='%Y'),
                   'pop': population})

# Reshape
df = df.pivot(index='year', columns='name', values='pop')
print(df)



# Build an hourly DatetimeIndex
idx = pd.date_range(df.index.min(), df.index.max(), freq='H')
print(len(idx))


# Reindex and interpolate with cubicspline as an example
res = df.reindex(idx).interpolate('cubicspline')

# Inspect
print(res.head().round(1))