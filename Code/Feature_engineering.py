import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import time
from matplotlib.image import imread
from tempfile import NamedTemporaryFile

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
import shap

from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from shapely.geometry import Point, LineString
import geopandas as gpd
from geopandas import GeoSeries
import geopandas as gpd
from shapely.geometry import Point
from geopandas.tools import geocode
from haversine import haversine, Unit
import geo.sphere
from geopy.geocoders import Nominatim

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import mixture
from scipy.spatial.distance import pdist, squareform, euclidean, cosine
from scipy.cluster.hierarchy import linkage, dendrogram


#Exploratory Data Analysis
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,8)
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


#Descriptive
df = pd.read_csv('./nyc-taxi-trip-duration/train.csv')

# print(df.shape)
# print(df.head())
# pd.set_option('float_format', '{:f}'.format)
# print(df.describe())
# print(df.info())


#Explore the target/ Data Preprocessing
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

print(np.log1p(df[target]).value_counts())
df['trip_duration_log'] = np.log1p(df['trip_duration'])
print(df.head())


#Pairplots
print(df.dtypes.value_counts())
# Get numerical features
numerical_cols = df.dtypes[df.dtypes != 'object'].index
# Remove the target columns
numerical_cols = numerical_cols.drop(['trip_duration_log', 'trip_duration'])
print(numerical_cols)
cols = list(numerical_cols) + ['trip_duration_log']
print(sns.pairplot(df.loc[:, cols]))

#Plot Distribution of all Features and Label¶
for col in df[numerical_cols].columns:
    fig, ax = plt.subplots(figsize=(5,5))
    df[col].hist(ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of feature: {col}')

#Nulls¶
null_summary = pd.concat((df.isnull().sum(), df.isnull().sum()/df.shape[0]), axis=1)
null_summary.columns = ['actual', 'pct']
null_summary['dtype'] = df.dtypes
print(null_summary)

# Correlation Matrix
# Start by creating a random variable to see where this falls in the correlation matrix
df['random'] = np.random.random(df.shape[0])
target_corr = df.corr()['trip_duration_log'].sort_values()
target_corr
ax = sns.heatmap(df.corr(), annot=True, linewidths=0.5, cmap ='YlGnBu')

#Feature Engineering
#Time-based Features¶
import holidays

class FeaturizeTime(object):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.copy()
        # Engineer temporal features
        X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])
        X['month'] = X['pickup_datetime'].dt.month
        X['day'] = X['pickup_datetime'].dt.day
        X['day_of_week'] = X['pickup_datetime'].dt.dayofweek
        X['hour_of_day'] = X['pickup_datetime'].dt.hour
        X['minute_of_hour'] = X['pickup_datetime'].dt.minute

        # Get holidays in NY state in 2016
        us_holidays = holidays.US(state='NY', years=2016)
        X['is_holiday'] = X['pickup_datetime'].dt.date.apply(lambda x: 1 if x in us_holidays.keys() else 0)

        # Drop time column in final df
        X = X.drop(['pickup_datetime'], axis=1)
        print('Time-based features transformed')
        return X

# Calculate Distance Between Coordinates
# The original dataset includes latitude and longitude for pickup and dropoff locations. I will use these to engineer the following location-based features:
# Great Circle Distance between pickup and dropoff locations (great_circle_distance): I will calculate the great circle distance using geo-py's API between the pickup and dropoff lat/long coordinates - this is the shortest distance between two points on a sphere.
# Manhattan Distance between pickup and dropoff locations (manhattan_distance): I will calculate the manhattan distance between the pickup and dropoff lat/long coordinates
# Bearing between pickup and dropoff locations (bearing): I will calculate the bearing using geo-py's API between the pickup and dropoff lat/long coordinates - this is the direction from the pickup location to dropoff location.
from haversine import haversine, Unit
import geo.sphere
class Calculateistances(object):
    def fit(self, X, y):
        return self
    def transform(self, X):
        X.copy()

        #find the minimum distance between pickup and dropoff coordinates
        X['great_circle_distance'] = X.apply(
            lambda x: self._calculate_great_circle_distance(x['pickup_latitude'], x['pickup_longitude'],
                                                            x['dropoff_latitude'], x['dropoff_longitude']),axis=1)

        #caluculate manhattan distance between pickup and dropoff
        X['manhattan_distance'] = X.apply(
            lambda x: self._calculate_manhattan_distance(x['pickup_latitude'], x['pickup_longitude'],
                                                         x['dropoff_latitude'], x['dropoff_longitude']),axis=1)

        #calculate bearing between pickup and dropoff
        X['bearing'] = X.apply(lambda x: self._calculate_bearing(x['pickup_latitude'], x['pickup_longitude'],
                                                                 x['dropoff_latitude'], x['dropoff_longitude']),axis=1)
        print('Distance features calculated')
        return X



    def _calculate_great_circle_distance(self, pickup_lat, pickup_long, dropoff_lat, dropoff_long):
        pickup = [pickup_lat, pickup_long]
        dropoff = [dropoff_lat, dropoff_long]
        distance = geo.sphere.distance(pickup, dropoff)
        return distance

    def _calculate_manhattan_distance(self, pickup_lat, pickup_long, dropoff_lat, dropoff_long):
        pickup = [pickup_lat, pickup_long]
        dropoff_a = [pickup_lat, dropoff_long]
        dropoff_b = [dropoff_lat, pickup_long]
        distance_a = geo.sphere.distance(pickup, dropoff_a)
        distance_b = geo.sphere.distance(pickup, dropoff_b)
        return distance_a + distance_b

    def _calculate_bearing(self, pickup_lat, pickup_long, dropoff_lat, dropoff_long):
        pickup = [pickup_lat, pickup_long]
        dropoff = [dropoff_lat, dropoff_long]
        bearing = geo.sphere.bearing(pickup, dropoff)
        return bearing


from sklearn.cluster import KMeans, MiniBatchKMeans
class GetClusterDensity():
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.pickup_time_clusters = pd.DataFrame()
        self.dropoff_time_clusters = pd.DataFrame()

    def fit(self, X, y):
        # I. Fit pickup cluster
        df_pickup = X[['pickup_latitude', 'pickup_longitude']].copy()

        ## Ia. Initialize K-means
        self.clf_pickup = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=10000, max_iter=300, random_state=1)
        self.clf_pickup.fit_predict(df_pickup)
        df_pickup['pickup_cluster'] = self.clf_pickup.labels_
        df_pickup['pickup_cluster'] = df_pickup['pickup_cluster'].astype(str)
        X = pd.merge(X, df_pickup[['pickup_cluster']], left_index=True, right_index=True, how='left')
        self.clf_pickup.fit(df_pickup.drop(['pickup_cluster'], axis=1))

        ## Ib. Calculate number of rides grouped by cluster and time as proxy for "density" of traffic by location and time
        pickup_time_clusters = pd.DataFrame(
            X.groupby(['month', 'day', 'hour_of_day', 'pickup_cluster'])['pickup_latitude'].count()).reset_index()
        pickup_time_clusters = pickup_time_clusters.rename(columns={'pickup_latitude': 'num_rides_by_pickup_group'})
        pickup_time_clusters_agg = pd.DataFrame(
            pickup_time_clusters.reset_index().groupby(['month', 'day'])['num_rides_by_pickup_group'].sum().round(4))
        pickup_time_clusters_agg = pickup_time_clusters_agg.rename(
            columns={'num_rides_by_pickup_group': 'agg_rides_per_day'})
        pickup_time_clusters = pd.merge(pickup_time_clusters.set_index(['month', 'day', 'hour_of_day']),
                                        pickup_time_clusters_agg, left_index=True, right_index=True)
        pickup_time_clusters['perc_rides_per_day_by_pickup_group'] = pickup_time_clusters['num_rides_by_pickup_group'] / \
                                                                     pickup_time_clusters['agg_rides_per_day']
        pickup_time_clusters = pickup_time_clusters.reset_index()
        pickup_time_clusters['pickup_group'] = pickup_time_clusters['pickup_cluster'].map(str) + str(',') + \
                                               pickup_time_clusters['month'].map(str) + str(',') + pickup_time_clusters[
                                                   'day'].map(str) + \
                                               str(',') + pickup_time_clusters['hour_of_day'].map(str)
        self.pickup_time_clusters = pickup_time_clusters

        # II. Fit dropoff cluster
        df_dropoff = X[['dropoff_latitude', 'dropoff_longitude']].copy()

        ## IIa. Initialize K-means
        self.clf_dropoff = MiniBatchKMeans(n_clusters=20, batch_size=10000, max_iter=300, random_state=1)
        self.clf_dropoff.fit_predict(df_dropoff)
        df_dropoff['dropoff_cluster'] = self.clf_dropoff.labels_
        df_dropoff['dropoff_cluster'] = df_dropoff['dropoff_cluster'].astype(str)
        X = pd.merge(X, df_dropoff[['dropoff_cluster']], left_index=True, right_index=True, how='left')
        self.clf_dropoff.fit(df_dropoff.drop(['dropoff_cluster'], axis=1))

        ## IIb. Calculate number of rides grouped by cluster and time as proxy for "density" of traffic by location and time
        dropoff_time_clusters = pd.DataFrame(
            X.groupby(['month', 'day', 'hour_of_day', 'dropoff_cluster'])['dropoff_latitude'].count()).reset_index()
        dropoff_time_clusters = dropoff_time_clusters.rename(columns={'dropoff_latitude': 'num_rides_by_dropoff_group'})
        dropoff_time_clusters_agg = pd.DataFrame(
            dropoff_time_clusters.reset_index().groupby(['month', 'day'])['num_rides_by_dropoff_group'].sum().round(4))
        dropoff_time_clusters_agg = dropoff_time_clusters_agg.rename(
            columns={'num_rides_by_dropoff_group': 'agg_rides_per_day'})
        dropoff_time_clusters = pd.merge(dropoff_time_clusters.set_index(['month', 'day', 'hour_of_day']),
                                         dropoff_time_clusters_agg, left_index=True, right_index=True)
        dropoff_time_clusters['perc_rides_per_day_by_dropoff_group'] = dropoff_time_clusters[
                                                                           'num_rides_by_dropoff_group'] / \
                                                                       dropoff_time_clusters['agg_rides_per_day']
        dropoff_time_clusters = dropoff_time_clusters.reset_index()
        dropoff_time_clusters['dropoff_group'] = dropoff_time_clusters['dropoff_cluster'].map(str) + str(',') + \
                                                 dropoff_time_clusters['month'].map(str) + str(',') + \
                                                 dropoff_time_clusters['day'].map(str) + \
                                                 str(',') + dropoff_time_clusters['hour_of_day'].map(str)
        self.dropoff_time_clusters = dropoff_time_clusters

        return self

    def transform(self, X):
        # III. Predict pickup cluster
        df_pickup = X[['pickup_latitude', 'pickup_longitude']].copy()

        ## IIIa. Add cluster label
        df_pickup['pickup_cluster'] = self.clf_pickup.predict(df_pickup)
        df_pickup['pickup_cluster'] = df_pickup['pickup_cluster'].astype(str)

        ## IIIb. Merge cluster label back to original dataframe
        X = pd.merge(X, df_pickup[['pickup_cluster']], left_index=True, right_index=True, how='left')

        ## IIIc. Merge to create "num_rides_by_pickup_group" and "perc_rides_by_pickup_group" features
        X['pickup_group'] = X['pickup_cluster'].map(str) + str(',') + X['month'].map(str) + str(',') + \
                            X['day'].map(str) + str(',') + X['hour_of_day'].map(str)
        X = pd.merge(X, self.pickup_time_clusters[
            ['pickup_group', 'num_rides_by_pickup_group', 'perc_rides_per_day_by_pickup_group']], on='pickup_group',
                     how='left')
        X = X.drop(['pickup_group', 'pickup_cluster'], axis=1)
        print('Pickup Clusters Found')

        # IV. Predict dropoff cluster
        df_dropoff = X[['dropoff_latitude', 'dropoff_longitude']].copy()

        ## IVa. Add cluster label
        df_dropoff['dropoff_cluster'] = self.clf_dropoff.predict(df_dropoff)
        df_dropoff['dropoff_cluster'] = df_dropoff['dropoff_cluster'].astype(str)

        ## IVb. Merge cluster label back to original dataframe
        X = pd.merge(X, df_dropoff[['dropoff_cluster']], left_index=True, right_index=True, how='left')

        ## IVc. Merge to create "num_rides_by_pickup_group" and "perc_rides_by_pickup_group" features
        X['dropoff_group'] = X['dropoff_cluster'].map(str) + str(',') + X['month'].map(str) + str(',') + \
                             X['day'].map(str) + str(',') + X['hour_of_day'].map(str)
        X = pd.merge(X, self.dropoff_time_clusters[
            ['dropoff_group', 'num_rides_by_dropoff_group', 'perc_rides_per_day_by_dropoff_group']], on='dropoff_group',
                     how='left')
        X = X.drop(['dropoff_group', 'dropoff_cluster'], axis=1)
        print('Dropoff Clusters Found')

        return X

#Gaussian Mixture Model¶
class GetClusterProbability():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        # Pickup locations
        self.pickup_cols = ['pickup_latitude', 'pickup_longitude']
        self.scaler_pickup = StandardScaler()
        pickup_scaled = self.scaler_pickup.fit_transform(X[self.pickup_cols])
        self.gmm_pickup = mixture.GaussianMixture(n_components=self.n_components).fit(pickup_scaled)

        # Dropoff locations
        self.dropoff_cols = ['dropoff_latitude', 'dropoff_longitude']
        self.scaler_dropoff = StandardScaler()
        dropoff_scaled = self.scaler_dropoff.fit_transform(X[self.dropoff_cols])
        self.gmm_dropoff = mixture.GaussianMixture(n_components=self.n_components).fit(dropoff_scaled)

        return self

    def transform(self, X):
        # Pickup locations
        pickup_scaled = self.scaler_pickup.transform(X[self.pickup_cols])
        preds_pickup = pd.DataFrame(self.gmm_pickup.predict_proba(pickup_scaled))
        preds_pickup = preds_pickup.add_prefix('gmm_pickup_')
        X = pd.merge(X, preds_pickup, left_index=True, right_index=True)
        print('GMM for pickup done')

        # Dropoff locations
        dropoff_scaled = self.scaler_dropoff.transform(X[self.dropoff_cols])
        preds_dropoff = pd.DataFrame(self.gmm_dropoff.predict_proba(dropoff_scaled))
        preds_dropoff = preds_dropoff.add_prefix('gmm_dropoff_')
        X = pd.merge(X, preds_dropoff, left_index=True, right_index=True)
        print('GMM for dropoff done')

        return X

#Dummify Categorical Features¶
class DummifyCategoricals():
    def fit(self, X, y):
        self.cols_to_encode = ['vendor_id', 'store_and_fwd_flag']
        self.encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')

        transformed_array = self.encoder.fit_transform(X[self.cols_to_encode]).toarray()
        self.transformed_colnames = [f'{prefix}_{value}'
                                     for prefix, values in zip(self.cols_to_encode, self.encoder.categories_)
                                     for value in values]

        return self

    def transform(self, X):
        transformed_array = self.encoder.transform(X[self.cols_to_encode]).toarray()
        transformed_df = pd.DataFrame(transformed_array, columns=self.transformed_colnames)

        X = pd.concat((X.drop(self.cols_to_encode, axis=1).reset_index(drop=True),
                       transformed_df.reset_index(drop=True)), axis=1)
        print('Categorical variables dummified')
        return X

#Super Naive Model¶
print(np.sqrt(mean_squared_log_error(df['trip_duration'], np.repeat(df['trip_duration'].mean(), df.shape[0]))))
print(np.sqrt(mean_squared_log_error(np.exp(df['trip_duration_log']), np.exp(np.repeat(df['trip_duration_log'].mean(), df.shape[0])))))

target = 'trip_duration_log'

#Naive Model
#No feature engineering;
# just modeling with cross validation. Use a small sample of 10k observations to see how random forest does

rf_cv_mean = np.sqrt(-cross_val_score(estimator=RandomForestRegressor(),
                X=df[numerical_cols].sample(10000, random_state=1),
                y=df[target].sample(10000, random_state=1),
                scoring='neg_mean_squared_error').mean())
print(f'Random Forest CV Mean RMSLE: {rf_cv_mean}')

#with Feature Engineering

train, df_test = train_test_split(df)

df_train, df_valid = train_test_split(train)

print(df_train.shape, df_valid.shape, df_test.shape)

# Remove original target from df_train and df_test
df_train = df_train.drop(['trip_duration'], axis=1)
df_valid = df_valid.drop(['trip_duration'], axis=1)
df_test = df_test.drop(['trip_duration'], axis=1)

print(df_train.head())

initial_feats = ['vendor_id',
                 'pickup_datetime',
                 'passenger_count',
                 'pickup_longitude',
                 'pickup_latitude',
                 'dropoff_longitude',
                 'dropoff_latitude',
                 'store_and_fwd_flag']
from sklearn.pipeline import make_pipeline

#Data Preprocessing
half_pipeline = make_pipeline(
                        FeaturizeTime(),
                         CalculateDistances(),
                         GetClusterDensity(n_clusters=20),
                         GetClusterProbability(n_components=20),
                         DummifyCategoricals()
                        )

df_train_feat = half_pipeline.fit_transform(df_train[initial_feats])
df_valid_feat = half_pipeline.transform(df_valid[initial_feats])
df_test_feat = half_pipeline.transform(df_test[initial_feats])


from sklearn.metrics import mean_squared_log_error, mean_squared_error
#Define Error Metric Calculation
def calc_rmsle(y_true_log, y_preds_log):
    return np.sqrt(mean_squared_log_error(np.exp(y_true_log), np.exp(y_preds_log)))



#Pipeline LightGBM Model
lgb = LGBMRegressor()
lgb.fit(df_train_feat, df_train[target], eval_set=[(df_valid_feat, df_valid[target])],
            eval_metric='rmse', early_stopping_rounds=10, verbose=1)
y_preds_lgbm = lgb.predict(df_test_feat)
calc_rmsle(df_test[target], y_preds_lgbm)

#Hyperparameter Tuning and Feature Selection
#Feature Selection
best_feats = df_train_feat.columns[np.argsort(lgb.feature_importances_)[::-1]]
print(best_feats)

features_intersection = [col for col in df_train_feat[df_train_feat.columns.intersection(df_test_feat.columns)].columns]
best_feats_intersection = [feat for feat in best_feats if feat in features_intersection]
print(len(best_feats_intersection))
print(best_feats_intersection)
print(best_feats_intersection[42:])

train_scores = []
test_scores = []

for i in range(1, len(best_feats_intersection)):
    lgb.fit(df_train_feat.loc[:, best_feats_intersection[:i]], df_train[target])
train_preds = lgb.predict(df_train_feat.loc[:, best_feats_intersection[:i]])
train_scores.append(calc_rmsle(df_train[target], train_preds))
test_preds = lgb.predict(df_test_feat.loc[:, best_feats_intersection[:i]])
test_scores.append(calc_rmsle(df_test[target], test_preds))
print(f'RMSLE recorded for round {i}')

len(test_scores)

#print forward feature selection
plt.plot(range(1, len(best_feats_intersection)), train_scores, label='train RMSLE')
plt.plot(range(1, len(best_feats_intersection)), test_scores, label='test RMSLE')

plt.xticks(np.arange(1,len(best_feats_intersection),2))
plt.xlabel('num features used in LGBM Regressor Model')
plt.ylabel('RMSLE Score')
plt.legend()
plt.title('Forward Feature Selection')
plt.ylim(0.39,0.45)
plt.savefig('./plots/step_forward_feat_sel.png', bbox_inches='tight')

min(test_scores)
best_n_feats = test_scores.index(min(test_scores)) + 1
print(best_n_feats)
X_train = df_train_feat.loc[:, best_feats_intersection[:best_n_feats]]
X_valid = df_valid_feat.loc[:, best_feats_intersection[:best_n_feats]]
X_test = df_test_feat.loc[:, best_feats_intersection[:best_n_feats]]

y_train = df_train[target]
y_valid = df_valid[target]
y_test = df_test[target]
print(X_train.shape, X_valid.shape, X_test.shape)

#Hyperparameter Tuning
params = {
    'num_leaves': [256,512,1024],
    'max_depth': [8,10,12],
    'n_estimators': [500],
    'subsample': [0.8],
    'feature_fraction': [0.9],
    'lambda_l1': [0.2],
    'learning_rate': [0.1]
}
def calc_rmsle(y_true_log, y_preds_log):
    return np.sqrt(mean_squared_log_error(np.exp(y_true_log), np.exp(y_preds_log)))

rmsle_scorer = metrics.make_scorer(calc_rmsle, greater_is_better=False)


lgb = LGBMRegressor(eval_metric='rmse')
reg = GridSearchCV(lgb, params, scoring=rmsle_scorer, verbose=True)
reg.fit(X_train, y_train)

print(reg.best_score_)

print(reg.best_params_)

lgb_cv_results = pd.DataFrame.from_dict(reg.cv_results_).sort_values(by='mean_test_score', ascending=False)
print(lgb_cv_results)


#Final Model
best_params_1 = {'feature_fraction': 0.9, 'lambda_l1': 0.2, 'learning_rate': 0.1, 'max_depth': 12,
               'n_estimators': 500, 'num_leaves': 512, 'subsample': 0.8}

final_lgb_1 = LGBMRegressor(eval_metric='rmse', **best_params_1)
final_lgb_1.fit(X_train, y_train,
              early_stopping_rounds=10,
              eval_set=[(X_train, y_train),(X_valid, y_valid)])

y_preds_2 = final_lgb_2.predict(X_test)
best_params_3 = {'feature_fraction': 0.9, 'lambda_l1': 0.2, 'learning_rate': 0.1, 'max_depth': 10,
               'n_estimators': 500, 'num_leaves': 256, 'subsample': 0.8}
final_lgb_3 = LGBMRegressor(eval_metric='rmse', **best_params_3)
final_lgb_3.fit(X_train, y_train,
              early_stopping_rounds=10,
              eval_set=[(X_train, y_train),(X_valid, y_valid)])
y_preds_3 = final_lgb_3.predict(X_test)
best_params_4 = {'feature_fraction': 0.9, 'lambda_l1': 0.2, 'learning_rate': 0.1, 'max_depth': 12,
               'n_estimators': 500, 'num_leaves': 1024, 'subsample': 0.8}
final_lgb_4 = LGBMRegressor(eval_metric='rmse', **best_params_4)
final_lgb_4.fit(X_train, y_train,
              early_stopping_rounds=10,
              eval_set=[(X_train, y_train),(X_valid, y_valid)])


y_preds_4 = final_lgb_4.predict(X_test)
best_params_5 = {'feature_fraction': 0.9, 'lambda_l1': 0.2, 'learning_rate': 0.1, 'max_depth': 12,
               'n_estimators': 500, 'num_leaves': 256, 'subsample': 0.8}
final_lgb_5 = LGBMRegressor(eval_metric='rmse', **best_params_5)
final_lgb_5.fit(X_train, y_train,
              early_stopping_rounds=10,
              eval_set=[(X_train, y_train),(X_valid, y_valid)])

y_preds_5 = final_lgb_5.predict(X_test)
y_preds_final = 0.2 * y_preds_1 + 0.2 * y_preds_2 + 0.2 * y_preds_3 + 0.2 * y_preds_4 + 0.2 * y_preds_5
calc_rmsle(y_test, y_preds_final)

#Summary Plots¶
fig, ax = plt.subplots()
ax.scatter(y_test,y_preds_final, alpha=0.7)
ax.set_xlabel('Actual (log)')
ax.set_ylabel('Preds (log)')
ax.set_title('Final LightGBM Model: actuals vs. preds on log scale')
plt.savefig('./plots/residuals_plot.png', bbox_inches='tight')


#Feature Importances¶
explainer_1 = shap.TreeExplainer(final_lgb_1)
examples = X_test.sample(10000)
shap_values_1 = explainer_1.shap_values(examples)
fig = shap.summary_plot(shap_values_1, examples, plot_type='bar', show=False)
fig = shap.summary_plot(shap_values_1, examples, show=False)


