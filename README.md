# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
# Importing Libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset
df=pd.read_csv('Churn_Modelling.csv')
df

#Checking for null values
df.isnull().sum()

#Checking for dulpicated values
df.duplicated()

#Dropping unwanted columns
df.drop('RowNumber',axis=1,inplace=True)
df.drop('CustomerId',axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df.drop('Surname',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df

#Normalising using MinMaxScaler
ms=MinMaxScaler()
df2=pd.DataFrame(ms.fit_transform(df))
df2

#Splitting the dataset - x
X=df2.iloc[:,:-1].values
X

#Splitting the dataset - y
y=df2.iloc[:,-1].values
y

# Training the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))

```

## OUTPUT:
### Read Dataset
<img width="820" alt="image" src="https://github.com/Shavedha/Ex.No.1---Data-Preprocessing/assets/93427376/7f8c460a-a539-48c0-8604-90338aaf6553">

### Checking for Null values
<img width="219" alt="image" src="https://github.com/Shavedha/Ex.No.1---Data-Preprocessing/assets/93427376/8bd296e0-ec51-408f-985b-279e2bfb5c1f">

### Checking for duplicated values
<img width="257" alt="image" src="https://github.com/Shavedha/Ex.No.1---Data-Preprocessing/assets/93427376/d2d566b9-c92b-480d-8360-ba5af2b0e8a2">

### Dropping off unwanted values
<img width="634" alt="image" src="https://github.com/Shavedha/Ex.No.1---Data-Preprocessing/assets/93427376/6d21e370-5540-4224-b011-541afa48088c">

### Normalised data using MinMaxScaler
<img width="409" alt="image" src="https://github.com/Shavedha/Ex.No.1---Data-Preprocessing/assets/93427376/ad503fb4-b6e0-4b0a-a933-b5a1b04cd873">

### Split values of X dataset
<img width="527" alt="image" src="https://github.com/Shavedha/Ex.No.1---Data-Preprocessing/assets/93427376/f705d82c-cc50-4efe-8c30-d33c0e822f5c">

### Split values of y dataset
<img width="353" alt="image" src="https://github.com/Shavedha/Ex.No.1---Data-Preprocessing/assets/93427376/3ad34b40-a092-4c7a-976b-eb6383c5380f">

### Training the dataset
<img width="476" alt="image" src="https://github.com/Shavedha/Ex.No.1---Data-Preprocessing/assets/93427376/cd39c2c1-ca02-4edd-83c9-7288a328f8ae">


## RESULT
Thus the given data is been processed successfully.
