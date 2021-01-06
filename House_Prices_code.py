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


#Now feature selection part
print(df_train.info()) #still 76 features remain

#check distribution of all the inputs
df_train.hist(figsize=(20, 20), bins=20)
plt.show()

#Based on the distribution results, check suspicious inputs
#3SsnPorch has too many zeros
df_train['3SsnPorch'].describe()

#BedroomAbvGr is not normal distributed
np.unique(df_train['BedroomAbvGr'].values)
df_train.groupby('BedroomAbvGr').count()['Id']

#Two Basement bathroom variables, can be merged together
df_train.groupby('BsmtFullBath').count()['Id']
df_train.groupby('BsmtHalfBath').count()['Id']
df_train['Bathroom']=df_train['BsmtFullBath']+ df_train['BsmtHalfBath'] 

#Four basement area variables. Three can be dropped
df_train[['TotalBsmtSF', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF']].head()

# Three more porch related variables. We can merge them togher or just keep one.
df_train[['OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']].describe()

#Garage area and cars must be correlated. Use area or cars?
df_train.corr()['GarageArea']['GarageCars']

#garage year built can be also dropped because we have a varaible: house year built.
df_train.corr()['GarageYrBlt']['YearBuilt']

#KitchenAbvGr can be dropped as there are too many 1s
df_train['KitchenAbvGr'].describe() 

#Lot area has a small proportion houses which have large area, need to be filtered.
sns.distplot(df_train['LotArea'], bins=100)
plt.show() 
#can filter Lot Area above 50000

#Now let's check the correlation matrix
df_train.corr()['SalePrice'].sort_values()

#YearBuilt and YearRemodAdd seems correlated
df_train.corr()['YearBuilt']['YearRemodAdd'] 

#Only select numeric variables (including SalePrice)
num_attrs = df_train.select_dtypes([np.int64, np.float64]).columns.values
df_train_num= df_train[num_attrs]

#Merge two bathroom variables
df_train_num['Bath']= df_train_num['BsmtFullBath'] + df_train_num['BsmtHalfBath'] 

#Remove the above variables
df_train_num=df_train_num.drop(['Id','3SsnPorch','BsmtFinSF2','BsmtFinSF1','BsmtUnfSF','EnclosedPorch','ScreenPorch','GarageCars',
                                'GarageYrBlt','KitchenAbvGr','YearRemodAdd','BsmtFullBath', 'BsmtHalfBath'],axis=1)

#Get the correlation matrix
corr = df_train_num.corr()
#using lambda function to reset the values: set 1 if the corr > 0.7 and -1 if the corr < -0.7, otherwise 0;
#the benifits: it's more clear to get to know which variables are highly correlated with each other;
corr = corr.applymap(lambda x : 1 if x > 0.7 else -1 if x < -0.7 else 0)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, vmax=1, center=0,vmin=-1 ,  square=True, linewidths=.005)
plt.show()

#Identify two correlated variables
df_train_num=df_train_num.drop(['TotRmsAbvGrd','1stFlrSF'],axis=1)

#list the correlation values
df_train_num.corr()['SalePrice'].sort_values()

#select correlation >0.5
df_train_num=df_train_num[df_train_num.columns[df_train_num.corr()['SalePrice']>0.5]]
df_train_num.columns

#Build the model
from sklearn import linear_model
reg = linear_model.LinearRegression()

#Split the input and output
df_train_num_x=df_train_num.drop('SalePrice',axis=1) 
df_train_num_y=df_train_num['SalePrice']

#Train the model
reg.fit(df_train_num_x, df_train_num_y)

#Check the model coefficients
print('Coefficients: \n', reg.coef_)

#Get the prediction based on the training dataset
preds = reg.predict(df_train_num_x)

#Check the training dataset prediction performance
from sklearn import metrics
#Mean Absolute Error 
print('MAE:', metrics.mean_absolute_error(df_train_num_y, preds))
#Mean Squared Error
print('MSE:', metrics.mean_squared_error(df_train_num_y, preds))
#Root Mean Squared Error
print('RMSE:', np.sqrt(metrics.mean_squared_error(df_train_num_y, preds)))

#Plot the predictions and actuals
plt.scatter(df_train_num_y,preds)
plt.show()

#Check the error
sns.distplot((df_train_num_y-preds),bins=35)
plt.show()

#Load the test data
df_test=pd.read_csv('test.csv')
df_test['Bath']= df_test['BsmtFullBath'] + df_test['BsmtHalfBath'] 
df_test_num= df_test[['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea','FullBath', 'GarageArea','Id']]

#IMPORTANT: All the feature engineering & data cleaning steps we have done to the training variables, we have to do the same for the test dataset!!
#Before we can feed the data into our model, we have to check missing values again. Otherwise the code will give you an error.
df_test_num.isnull().sum()
df_test_num['TotalBsmtSF']=df_test_num['TotalBsmtSF'].fillna(np.mean(df_test_num['TotalBsmtSF']))
df_test_num['GarageArea']=df_test_num['GarageArea'].fillna(np.mean(df_test_num['GarageArea']))


#Predict the results for test dataset
submit= pd.DataFrame()
submit['Id'] = df_test_num['Id']
#select features 
preds_out = reg.predict(df_test_num[['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea','FullBath', 'GarageArea']])
submit['SalePrice'] = preds_out
#final submission  
submit.to_csv('test_submit.csv', index=False)

#Check output
#check yearly alignment
df_train['preds']=preds
df_yearly=df_train[['SalePrice','preds','YearBuilt']].groupby('YearBuilt').mean()
sns.lineplot(data=df_yearly)
plt.show()

#check Rates the overall material and finish of the house
df_yearly1=df_train[['SalePrice','preds','OverallQual']].groupby('OverallQual').mean()
sns.lineplot(data=df_yearly1)

#check Rates the overall condition of the house
df_yearly2=df_train[['SalePrice','preds','OverallCond']].groupby('OverallCond').mean()
sns.lineplot(data=df_yearly2)

#check Bedrooms
df_yearly3=df_train[['SalePrice','preds','BedroomAbvGr']].groupby('BedroomAbvGr').mean()
sns.lineplot(data=df_yearly3)


#using the same data features and data processing part but using Lasso model rather than linear model to check the result;
#Import packages
import pandas as pd
import numpy as np
from sklearn import linear_model

#Load the training data
df_train=pd.read_csv('train.csv')

#Confirmed inputs
df_train_num=df_train[['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea','FullBath', 'GarageArea','Id','SalePrice']]

#Normalise Area
df_train_num['GrLivArea']=np.log(df_train_num['GrLivArea'])
df_train_num['SalePrice']=np.log(df_train_num['SalePrice'])

#Remove outliers
df_train_num = df_train_num[df_train_num['TotalBsmtSF']<3000]
df_train_num = df_train_num[df_train_num['GrLivArea']<4500]
df_train_num = df_train_num[df_train_num['GarageArea']<1250]

#Build the model
ls = linear_model.LassoCV()

#Split the input and output
df_train_num_x=df_train_num.drop(['SalePrice','Id'],axis=1) 
df_train_num_y=df_train_num['SalePrice']

#Train the model
ls.fit(df_train_num_x, df_train_num_y)

#Check the model coefficients
print('Coefficients: \n', ls.coef_)

#Load the test data
df_test=pd.read_csv('test.csv')
df_test_num=df_test[['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea','FullBath', 'GarageArea','Id']]

#IMPORTANT: All the feature engineering & data cleaning steps we have done to the testing variables, we have to do the same for the test dataset!!
#Normalise Area
df_test_num['GrLivArea']=np.log(df_test_num['GrLivArea'])

#Before we can feed the data into our model, we have to check missing values again. Otherwise the code will give you an error.
df_test_num.isnull().sum()
df_test_num['TotalBsmtSF']=df_test_num['TotalBsmtSF'].fillna(np.mean(df_test_num['TotalBsmtSF']))
df_test_num['GarageArea']=df_test_num['GarageArea'].fillna(np.mean(df_test_num['GarageArea']))


#Predict the results for test dataset
submit= pd.DataFrame()
submit['Id'] = df_test_num.Id
#select features 
preds_out = ls.predict(df_test_num[['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea','FullBath', 'GarageArea']])
submit['SalePrice'] = np.exp(preds_out)
#final submission  
submit.to_csv('LassoCV_submission.csv', index=False)