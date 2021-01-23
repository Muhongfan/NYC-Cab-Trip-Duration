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

target='trip_duration'
print(df[target].mean())
print(df[target].describe())
sns.kdeplot(df[target])


