#!/usr/bin/env python
# coding: utf-8

# # **Cancer Prediction**
# 
# Attribute Information:
# 
# - Diagnosis (M = malignant, B = benign)
# 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# - radius (mean of distances from center to points on the perimeter)
# - texture (standard deviation of gray-scale values)
# - perimeter
# - area
# - smoothness (local variation in radius lengths)
# - compactness (perimeter^2 / area - 1.0)
# - concavity (severity of concave portions of the contour)
# - concave points (number of concave portions of the contour)
# - symmetry
# - fractal dimension ("coastline approximation" - 1)
# 
# Dataset : https://github.com/ybifoundation/Dataset/raw/main/Cancer.csv

# # **Q. Classification Predictive Model**

# In[1]:


# Step 1 : import library
import pandas as pd


# In[2]:


# Step 2 : import data
cancer =pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Cancer.csv')


# In[4]:


# Step 3 : define y and X


# In[3]:


cancer.columns


# In[4]:


y = cancer['diagnosis']


# In[5]:


X = cancer[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]


# In[6]:


# Step 4 : train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)


# In[10]:


# check shape of train and test sample
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[31]:


# Step 5 : select model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000)


# In[32]:


# Step 6 : train or fit model
model.fit(X_train,y_train)


# In[33]:


# Step 7 : predict model
y_pred=model.predict(X_test)


# In[34]:


# Step 8 : model accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[35]:


accuracy_score(y_test,y_pred)


# In[36]:


confusion_matrix(y_test,y_pred)


# In[37]:


print(classification_report(y_test,y_pred))


# In[ ]:




