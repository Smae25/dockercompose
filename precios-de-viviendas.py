# In[82]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import psycopg2

 # In[83]:

 # Leer el archivo de descripción de los datos
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


 # In[84]:


df_train.head()


 # In[85]:


df_train.shape


 # In[86]:


df_train.info()


 # In[87]:


df_train.duplicated().sum()


 # In[88]:


pd.set_option('display.max_rows', None)
df_train.isnull().sum().sort_values(ascending = False)


 # In[89]:


df_test.isnull().sum().sort_values(ascending = False)


 # In[90]:


df_train.drop(columns=['PoolQC','MiscFeature','Alley','Fence','MasVnrType', 'FireplaceQu'], inplace=True)
df_test.drop(columns=['PoolQC','MiscFeature','Alley','Fence','MasVnrType', 'FireplaceQu'], inplace=True)


 # In[91]:


df_train.shape


 # In[92]:


X=df_train.drop(columns=['SalePrice'])
y=df_train.SalePrice


 # In[93]:


imputer = SimpleImputer(strategy='most_frequent', fill_value="most_frequent")
imputed_X_train = pd.DataFrame(imputer.fit_transform(X))

 # imputar valores nulos por moda en df_test
imputed_df_test = pd.DataFrame(imputer.transform(df_test))
imputed_df_test.columns = df_test.columns

imputed_X_train.columns = X.columns


 # In[94]:


pd.reset_option('display.max_rows')
imputed_X_train.isnull().sum().sort_values(ascending = False)


 # In[95]:


s=(imputed_X_train.dtypes=='object')
object_cols=list(s[s].index)

ordinal_encoder=OrdinalEncoder()
ordinal_encoder.fit(pd.concat([imputed_X_train[object_cols], imputed_df_test[object_cols]]))

imputed_X_train[object_cols]=ordinal_encoder.transform(imputed_X_train[object_cols])
imputed_df_test[object_cols]=ordinal_encoder.transform(imputed_df_test[object_cols])


 # In[96]:


pd.set_option('display.max_rows', None)
mi_scores = mutual_info_regression(imputed_X_train, y)
mi_series = pd.Series(mi_scores, index=imputed_X_train.columns)
mi_series_sorted = mi_series.sort_values(ascending=False)

print(mi_series_sorted)


 # In[97]:


imputed_X_train['ageAtRemod'] = imputed_X_train['YrSold']- imputed_X_train['YearRemodAdd']
imputed_X_train['ageAtSold'] = imputed_X_train['YrSold']-imputed_X_train['YearBuilt']
imputed_X_train['TotalBathrooms'] = imputed_X_train['BsmtFullBath'] + imputed_X_train['BsmtHalfBath'] + imputed_X_train['FullBath'] + imputed_X_train['HalfBath']
imputed_X_train['TotalPorchSF'] = imputed_X_train['OpenPorchSF'] + imputed_X_train['EnclosedPorch'] + imputed_X_train['3SsnPorch'] + imputed_X_train['ScreenPorch']
imputed_df_test['ageAtRemod'] = imputed_df_test['YrSold']- imputed_df_test['YearRemodAdd']
imputed_df_test['ageAtSold'] = imputed_df_test['YrSold']-imputed_df_test['YearBuilt']
imputed_df_test['TotalBathrooms'] = imputed_df_test['BsmtFullBath'] + imputed_df_test['BsmtHalfBath'] + imputed_df_test['FullBath'] + imputed_df_test['HalfBath']
imputed_df_test['TotalPorchSF'] = imputed_df_test['OpenPorchSF'] + imputed_df_test['EnclosedPorch'] + imputed_df_test['3SsnPorch'] + imputed_df_test['ScreenPorch']


 # In[98]:


mi_scores = mutual_info_regression(imputed_X_train, y)
mi_series = pd.Series(mi_scores, index=imputed_X_train.columns)
mi_series_sorted = mi_series.sort_values(ascending=False)

print(mi_series_sorted)


 # In[99]:


relevant_features_mask = mi_series_sorted > 0.01

relevant_columns = mi_series_sorted[relevant_features_mask].index.tolist()

df_relevant = imputed_X_train[relevant_columns]
df_test_relevant=imputed_df_test[relevant_columns]


 # In[100]:


df_relevant.select_dtypes(['int64', 'float64']).hist(figsize = (25,25), bins = 30)
plt.show()


 # In[101]:


scaler=MinMaxScaler()
df_relevant= pd.DataFrame(scaler.fit_transform(df_relevant))
df_test_relevant= pd.DataFrame(scaler.transform(df_test_relevant))


 # In[102]:


df_relevant.select_dtypes(['int64', 'float64']).hist(figsize = (25,25), bins = 30)
plt.show()


 # In[103]:


X_train, X_valid, y_train, y_valid = train_test_split(df_relevant, y, train_size=0.9, test_size=0.1,random_state=0)


 # In[104]:


param_grid = {
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100],
    
 }

estimator = XGBRegressor()

optimal_params = GridSearchCV(estimator=estimator, param_grid=param_grid, verbose=0)
optimal_params.fit(X_train, y_train, verbose=False)

model= optimal_params.best_estimator_


 # In[105]:


df_test_rel=imputed_df_test[relevant_columns]

df_test_rel.head()


 # In[106]:


df_test_rel.isnull().sum()

 # In[107]:


predicciones = model.predict(df_test_relevant)
df_test['SalePrice'] = predicciones
df= df_test[['Id','SalePrice']]
df.head(10)


 # In[108]:


df.to_csv('submission.csv', index=False)


 # In[109]:


submission_df = pd.read_csv('submission.csv')
print(submission_df.head())
print(submission_df.shape)