import pandas as pd

df = pd.read_csv('./nyc-taxi-trip-duration/train.csv')

#Descriptive
# print(df.shape)
# print(df.head())
# pd.set_option('float_format', '{:f}'.format)
# print(df.describe())
# print(df.info())

#Explore the target
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

target='trip_duration'
print(df[target].mean())
print(df[target].describe())
print(sns.kdeplot(df[target]))

fig, ax = plt.subplots(1,2)
fig.set_size_inches(12,6)
ax[0].hist(df[target], bins=30, label='original', color='red')
ax[1].hist(np.log1p(df[target]), bins=30, label='log')
fig.suptitle(target)
fig.legend()


