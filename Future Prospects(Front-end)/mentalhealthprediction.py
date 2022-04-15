# -*- coding: utf-8 -*-
"""MentalHealthPrediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eL8FTWtwScnLBVuD_jx9kGupEIbRozEh
"""

#Importing the libraries
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

#Importing the dataset
from google.colab import files
uploaded=files.upload()

#Loading the dataset
df=pd.read_csv('data.csv')

#Printing 10 instances of the data
df.tail

#Getting the shape of the data
df.shape

#Exploring the data
df.describe()

#Checking if there is any null values are present or not
df.isna().sum()

"""In the above exploaration, we found out that the number of missing values in most cases are more than 5% of the data, that is if we do 5% of 286 = 14.3, which exceeds the limit of our missing values in most cases. Moreover its Multivariate Analysis, so we need to keep in account all the columns which are contributing to the analysis and hence forth we can't drop those columns. However we can drop which are less than that."""

#Dropping out those columns whose missing values are less than or equal to 14
df.drop(columns=['Internet_bi', 'Others_bi', 'Alone_bi', 'religion_bi', 'Doctor_bi', 'Phone_bi', 'Professional_bi', 'Relative_bi', 'Parents_bi', 'Friends_bi', 'Partner_bi', 'DepSev'], inplace=True)

#Here those features with have a greater effect on the data analysis.
df.isna().sum()

"""So with the useful data as of now, we can convert the categorical data into numerical values for analysis."""

#Converting inter_dom to Numerical data
df['inter_dom2']=pd.factorize(df.inter_dom)[0]       #Replacing them
df.drop(columns=['inter_dom'], inplace=True)         #Dropping the original column  
#df.rename(columns={"inter_dom2":"inter_dom"})        #Renaming it for convinience

#To check which values are being assigned we do this
#df.inter_dom2.value_counts()

#Converting Region to Numerical data
df['Region2']=pd.factorize(df.Region)[0]       #Replacing them
df.drop(columns=['Region'], inplace=True)         #Dropping the original column  
#df.rename(columns={"Region2":"Region"})        #Renaming it for convinience

#df.Region2.value_counts()

#Converting Gender to Numerical data
df['Gender2']=pd.factorize(df.Gender)[0]       #Replacing them
df.drop(columns=['Gender'], inplace=True)         #Dropping the original column  
#df.rename(columns={"Gender2":"Gender"})        #Renaming it for convinience

#Converting Academic to Numerical data
df['Academic2']=pd.factorize(df.Academic)[0]       #Replacing them
df.drop(columns=['Academic'], inplace=True)         #Dropping the original column

#Converting Stay_Cate to Numerical data
df['Stay_Cate2']=pd.factorize(df.Stay_Cate)[0]       #Replacing them
df.drop(columns=['Stay_Cate'], inplace=True)         #Dropping the original column

#Converting Japanese_cate to Numerical data
df['Japanese_cate']=pd.factorize(df.Japanese_cate)[0]       #Replacing them
df.drop(columns=['Japanese_cate'], inplace=True)         #Dropping the original column

#Converting English_cate to Numerical data
df['English_cate2']=pd.factorize(df.English_cate)[0]       #Replacing them
df.drop(columns=['English_cate'], inplace=True)         #Dropping the original column

#Converting Intimate to Numerical data
df['Intimate2']=pd.factorize(df.Intimate)[0]       #Replacing them
df.drop(columns=['Intimate'], inplace=True)         #Dropping the original column

#Converting Religion to Numerical data
df['Religion2']=pd.factorize(df.Religion)[0]       #Replacing them
df.drop(columns=['Religion'], inplace=True)         #Dropping the original column

#Converting Suicide to Numerical data
df['Suicide2']=pd.factorize(df.Suicide)[0]       #Replacing them
df.drop(columns=['Suicide'], inplace=True)         #Dropping the original column

#Converting Dep to Numerical data
df['Dep2']=pd.factorize(df.Dep)[0]       #Replacing them
df.drop(columns=['Dep'], inplace=True)         #Dropping the original column

#Converting DepType to Numerical data
df['DepType2']=pd.factorize(df.DepType)[0]       #Replacing them
df.drop(columns=['DepType'], inplace=True)         #Dropping the original column

#While we are converting our data, it is bound to get shuffled and hence we need to have proper display, 
#before we can get hold of the features and the target variable
pd.options.display.max_columns = None
display(df)

#Changing NaNs to 0
df=df.fillna(0)

#Splitting the data into labels and features
X1=df.iloc[:,0:34]
X2=df.iloc[:,35:37] 
X=X1.join(X2) #Features
#X = X.values.reshape(1,-1)
Y=df.iloc[:,34:35] #Label
#Y = Y.values.reshape(1,-1)
#print(Y)

#print(X1)
#print(X2)

#Splitting the training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)

#Visualizing the data
dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show

#Feature Selection using Extra Tree Classifier

# Building the model 
extra_tree_forest = ExtraTreesClassifier(n_estimators = 5, criterion ='entropy', max_features = 2) 
  
# Training the model 
extra_tree_forest.fit(X, Y) 
  
# Computing the importance of each feature 
feature_importance = extra_tree_forest.feature_importances_ 
  
# Normalizing the individual importances 
feature_importance_normalized = np.std([tree.feature_importances_ for tree in extra_tree_forest.estimators_], axis = 0) 

# Plotting a Bar Graph to compare the models 
plt.bar(X.columns, feature_importance_normalized) 
plt.xlabel('Feature Labels') 
plt.ylabel('Feature Importances') 
plt.title('Comparison of different Feature Importances') 
plt.show()

#Feature Selection Using Chi Square Test
selector = SelectKBest(k=25)
X_train_selected = selector.fit_transform(X_train,Y_train)

#Using Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_selected,Y_train)
# select the same features on the test set, predict, and get the test accuracy:
X_test_selected = selector.transform(X_test)
y_pred = lr.predict(X_test_selected)
accuracy_score(Y_test, y_pred)

#Using KNeighbors Classifier
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(X_train_selected, Y_train)  
X_test_selected = selector.transform(X_test)
y_pred = classifier.predict(X_test_selected)
accuracy_score(Y_test, y_pred)

#Creating the confusion matrix
confusion_matrix=confusion_matrix(Y_test,y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))

#Using Support Vector Machine
classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(X_train_selected, Y_train)  
X_test_selected = selector.transform(X_test)
y_pred = classifier.predict(X_test_selected)
accuracy_score(Y_test, y_pred)

#Creating the confusion matrix
confusion_matrix=confusion_matrix(Y_test,y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))

#Using Naive Bayes Classifier
classifier = GaussianNB()  
classifier.fit(X_train_selected, Y_train)  
X_test_selected = selector.transform(X_test)
y_pred = classifier.predict(X_test_selected)
accuracy_score(Y_test, y_pred)

#Creating the confusion matrix
confusion_matrix=confusion_matrix(Y_test,y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))