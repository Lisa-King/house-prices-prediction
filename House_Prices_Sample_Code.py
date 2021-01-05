#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats

#Load the training data
df_train=pd.read_csv('train.csv')

#using df_train to show the first and last 5 rows of the dataframe, and how many rows and columns;
#df_train
#Check the column headers (the names of each column)
print(df_train.columns)

#Descriptive statistics summary
#show the statistics summary infomation of "SalePrice"
df_train['SalePrice'].describe()

#histogram
sns.distplot(df_train['SalePrice'])
#sns.distplot() can generate the distribution of the variable, but only use one more plt.show() can show the picture;
plt.show()

#correlation matrix
corrmat = df_train.corr()
#fig, ax = plt.subplots(figsize = (a, b)) --figzise is used to set the figure size, a is the figure width and b is the figure hight in feet;
#fig, ax = plt.subplots(m, n, 1) -- m is rows for the subplots, n is columns for the subplots and 1 means the first subplots; so
#use it to define m*n subplots;
f, ax = plt.subplots(figsize=(12, 9))
#plt.show() now it's only show an empty window withe the defined size;
#plot the heatmap
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

#check if this variable is not relevant to the sale price
var = 'EnclosedPorch'
#axis=1 means extract by column and then concatenate the two columns data into a new dataframe; 
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#data.plot.scatter means plot the scotter figure; and define x and y axis variable and limit the y axis value between [0,800000];
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#check if there are duplicated rows in the dataframe or not; if the result is empty dataframe, it means there is no duplicated rows.
df_train[df_train.duplicated()==True]

#check column data types, show each column the datatypes;
res = df_train.dtypes
#show all int64 type columns;
print(res[res == np.dtype('int64')])
print(res[res == np.dtype('bool')])
print(res[res == np.dtype('object')])
print(res[res == np.dtype('float64')])

#standardize
#show all unique values of variable;
print(df_train["LotConfig"].unique())

# feature scaling, only apply to numeric data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(df_train[["GrLivArea","SalePrice"]])
sns.distplot(X_train[:,1],fit=norm)
plt.show()

#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm)
plt.show()

#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'],fit=norm)
plt.show()

#missing data
#sum the number of null value in each column and then sort the columns by descending order;
total = df_train.isnull().sum().sort_values(ascending=False)
#df_train.isnull().count() is the actually row number of the column, so all the columnn has 1460 rows; number of null value/total rows = percentage
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
#concatenate total and percent together by column since axis=1, and define the column names;
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#only extract the first 20 rows of missing_data;
missing_data.head(20)

#dealing with missing data
#select the index of percent of the missing_data > 0.15, and drop them by column; 1 means by column;
df_train = df_train.drop((missing_data[missing_data['Percent'] > 0.15]).index,1)
#drop rows in which the value of Electrical is null;
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
#after drop comluns and rows, df_train [1460 rows x 75 columns] -> [1459 rows x 75 columns]

#if not drop, we can impute
# check correlation between LotFrontage and LotArea
df_train['LotFrontage'].corr(df_train['LotArea'])
df_train['SqrtLotArea']=np.sqrt(df_train['LotArea'])
#using squareroot and the correlation is more clear;
df_train['LotFrontage'].corr(df_train['SqrtLotArea'])

cond = df_train['LotFrontage'].isnull()
df_train["LotFrontage"][cond]=df_train["SqrtLotArea"][cond]
print(df_train["LotFrontage"].isnull().sum())

#flag the missing data as missing
mis=df_train['GarageType'].isnull()
#set null value of GarageType as missing;
df_train["GarageType"][mis]="Missing"
df_train["GarageType"].unique()

#identify the outliers
fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(16, 4))
axes = np.ravel(axes)
col_name = ['GrLivArea','TotalBsmtSF','1stFlrSF','BsmtFinSF1','LotArea']

for i, c in zip(range(5), col_name):
    df_train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='r')

#delete outliers
print(df_train.shape)
#drop outliers to make the data distribution better;
df_train = df_train[df_train['GrLivArea'] < 4500]
df_train = df_train[df_train['LotArea'] < 100000]
df_train = df_train[df_train['TotalBsmtSF'] < 3000]
df_train = df_train[df_train['1stFlrSF'] < 2500]
df_train = df_train[df_train['BsmtFinSF1'] < 2000]

print(df_train.shape)

for i, c in zip(range(5,10), col_name):
    df_train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='b')

plt.show()