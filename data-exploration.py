#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from scipy import stats

df_train = pd.read_csv("train.csv")
df_train.columns
df_train["SalePrice"].describe()
sns.distplot(df_train["SalePrice"])
print("Skewness: %f" %df_train["SalePrice"].skew())
print("Kurtosis: %f" %df_train["SalePrice"].kurt())

var = "GrLivArea"
data = pd.concat([df_train["SalePrice"],df_train[var]], axis =1)
data.plot.scatter(x = var, y ="SalePrice", ylim=(0.80000))
#linear relationship

var = 'TotalBsmtSF'
data = pd.concat([df_train["SalePrice"],df_train[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0.8))
#strong linear model

var = "OverallQual"
data = pd.concat([df_train["SalePrice"],df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x= var, y="SalePrice", data=data)
fig.axis(ymin=0,ymax=800000);
#Again, linear relationship with overallqual and saleprice


var = "YearBuilt"
data = pd.concat([df_train["SalePrice"],df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16,8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0,ymax=800000)
plt.xticks(rotation=90);
#It's not strong tendency but there is certain prone that customer pays more to new stuff

#Correlation Matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.8, square=True);

#Saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k,"SalePrice")["SalePrice"].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar= True, annot=True, square=True, fmt =".2f", annot_kws={"size":10},yticklabels = cols.values, xticklabels = cols.values)
plt.show()

sns.set()
cols = ["SalePrice", "OverallQual","GrLivArea","GarageCars","TotalBsmtSF", "FullBath", "YearBuilt"]
sns.pairplot(df_train[cols], height=2.5)
plt.show()

#Missing Values
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() /df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis=1, keys =["Total","Percent"])
missing_data.head(20)

#dealing with the missing data
#df_train = df_train.drop((missing_data[missing_data['total'] > 1]).index,1)
#df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum()

#Univarite analysis
#Outliers is a complex subject and it deserves more attention. 
#Here, we'll just do a quick analysis through the standard deviation of 'SalePrice' and a set of scatter plots.

saleprice_scaled = StandardScaler().fit_transform(df_train["SalePrice"][:,np.newaxis]);
low_range =  saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print("Outer range(low) is the distribution:")
print(low_range)
print("\nOuter range(high) is the distribution:")
print(high_range)

#bivariate analysis
var = "GrLivArea"
data = pd.concat([df_train["SalePrice"],df_train[var]], axis = 1)
data.plot.scatter(x=var, y="SalePrice", ylim = (0.8))

#deleting outlier points
df_train.sort_values(by="GrLivArea", ascending=False)[:2]
df_train = df_train.drop(df_train[df_train["Id"]==1299].index)
df_train = df_train.drop(df_train[df_train["Id"]==524].index)

#bivarite analysis
var = "TotalBsmtSF"
data = pd.concat([df_train["SalePrice"],df_train[var]], axis = 1)
data.plot.scatter(x = var, y="SalePrice", ylim = 0.80000)

df_train.sort_values(by="TotalBsmtSF", ascending=False)[:3]
df_train = df_train.drop(df_train[df_train["Id"]== 333].index)
df_train = df_train.drop(df_train[df_train["Id"]== 497].index)
df_train = df_train.drop(df_train[df_train["Id"]== 441].index)

#lets go deep 
#histogram and normal probability plot
sns.distplot(df_train["SalePrice"],fit = norm)
plt.show()
figure = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
#we need a simple data transformation to solve peakedness in SalePrice
#in case of positive skewness, log transformations usually works well.

#applying log transformation
df_train["SalePrice"] = np.log(df_train["SalePrice"])
#transformed histogram and normal probability plot
sns.distplot(df_train["SalePrice"], fit= norm)
plt.show()
figure = plt.figure
res = stats.probplot(df_train["SalePrice"], plot=plt)

#histogram and normal probability plot
sns.distplot(df_train["GrLivArea"],fit=norm)
plt.show()
figure = plt.figure
res = stats.probplot(df_train["GrLivArea"], plot = plt)

#data transformation
df_train["GrLivArea"] = np.log(df_train["GrLivArea"])
sns.distplot(df_train["GrLivArea"], fit= norm)
plt.show()
figure = plt.figure
res = stats.probplot(df_train["GrLivArea"],plot=plt)

#histogram and normal probabiliy plot
sns.distplot(df_train["TotalBsmtSF"],fit=norm)
plt.show()
figure = plt.figure
res = stats.probplot(df_train["TotalBsmtSF"], plot = plt)

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0

df_train["HasBsmt"] = pd.Series(len(df_train["TotalBsmtSF"]), index = df_train.index)
df_train["HasBsmt"] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

#scatter plot
plt.scatter(df_train["GrLivArea"],df_train["SalePrice"])
plt.show()
#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
#convert categorical variable into dummy
df_train = pd.get_dummies(df_train)
