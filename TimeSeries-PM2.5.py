#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# In[2]:


#load data
data = pd.read_excel('107年新竹站_20190315.xls')
data


# In[3]:


#extract Oct, Nov, Dec
date = data['日期'] >= "2018/10/01"
daten = pd.DataFrame(data[date])
daten.to_csv('temperature.csv', index=False, encoding = 'utf-8')
temp = pd.read_csv('temperature.csv')
temp


# In[4]:


#check PM2.5 index
(temp[temp.values == "PM2.5"].index)%18


# In[5]:


#make a list for 18 items and seperate train and test
temp_train = []
temp_test = []

for i in range(18):
    temp_train.append([])
    temp_test.append([])
    
for i in range(len(temp)):
    if temp.iloc[i,0] >= "2018/10/01" and temp.iloc[i,0] <= "2018/11/30":
        for j in range(3, 27):
            temp_train[i%18].append((temp.iloc[i,j]))
    elif temp.iloc[i,0] >= "2018/12/01":
        for j in range(3, 27):
            temp_test[i%18].append((temp.iloc[i,j]))
#transfer to dataframe
temp_train = DataFrame(temp_train)
temp_test = DataFrame(temp_test)


# In[6]:


#fill in rainfall missing value with 0
temp_train = temp_train.replace("NR", 0)
temp_test = temp_test.replace("NR",0)


# In[7]:


#fill in missing value with mean, train_data
for a in range(0, len(temp_train)):
    for i in range (1, len(temp_train.columns)):
        j = i-1
        k = i+1
        if str(temp_train.iloc[a, i]).rfind("#") !=-1 or str(temp_train.iloc[a,i]).rfind("*") !=-1 or str(temp_train.iloc[a,i]).rfind("x")!=-1 or str(temp_train.iloc[a,i]).rfind("A")!=-1 or pd.isnull(temp_train.iloc[a,i]) == True:
            while str(temp_train.iloc[a,j]).rfind("#") != -1 or str(temp_train.iloc[a,j]).rfind("*") !=-1 or str(temp_train.iloc[a,j]).rfind("x")!=-1 or str(temp_train.iloc[a,j]).rfind("A")!= -1 or pd.isnull(temp_train.iloc[a,j]) == True:
                j = j-1
            while str(temp_train.iloc[a,k]).rfind("#") != -1 or str(temp_train.iloc[a,k]).rfind("*") !=-1 or str(temp_train.iloc[a,k]).rfind("x")!=-1 or str(temp_train.iloc[a,k]).rfind("A")!= -1 or pd.isnull(temp_train.iloc[a,k]) == True:
                k = k+1
            temp_train.iloc[a, i] = str((float(temp_train.iloc[a,j]) + float(temp_train.iloc[a,k])) / 2)
            #print(temp_train.iloc[a, i], temp_train.iloc[a, k], temp_train.iloc[a, j])


# In[8]:


#fill in missing value with mean, test_data
for a in range(0, len(temp_test)):
    for i in range (1, len(temp_test.columns)):
        j = i-1
        k = i+1
        if str(temp_test.iloc[a, i]).rfind("#") !=-1 or str(temp_test.iloc[a,i]).rfind("*") !=-1 or str(temp_test.iloc[a,i]).rfind("x")!=-1 or str(temp_test.iloc[a,i]).rfind("A")!=-1 or pd.isnull(temp_test.iloc[a,i]) == True:
            while str(temp_test.iloc[a,j]).rfind("#") != -1 or str(temp_test.iloc[a,j]).rfind("*") !=-1 or str(temp_test.iloc[a,j]).rfind("x")!=-1 or str(temp_test.iloc[a,j]).rfind("A")!= -1 or pd.isnull(temp_test.iloc[a,j]) == True:
                j = j-1
            while str(temp_test.iloc[a,k]).rfind("#") != -1 or str(temp_test.iloc[a,k]).rfind("*") !=-1 or str(temp_test.iloc[a,k]).rfind("x")!=-1 or str(temp_test.iloc[a,k]).rfind("A")!= -1 or pd.isnull(temp_test.iloc[a,k]) == True:
                k = k+1
            temp_test.iloc[a, i] = str((float(temp_test.iloc[a,j]) + float(temp_test.iloc[a,k])) / 2)
            #print(temp_test.iloc[a, i], temp_test.iloc[a, k], temp_test.iloc[a, j])


# In[9]:


#split train_x
temp_train_x = temp_train.drop(index = 9)
period = 6
length = len(temp_train_x.columns) - period
train_x = []
for i in range(length):
    train_x.append([])
for i in range(length):    
    for j in range(len(temp_train_x.index)):
        for k in range(period):
            train_x[i].append(temp_train_x.iloc[j, i+k])


# In[10]:


#split train_y
temp_train_y = temp_train.iloc[9,:]
train_y = []
for i in range(6, len(temp_train_y)):
    train_y.append(temp_train_y[i])


# In[11]:


#split test_x
temp_test_x = temp_test.drop(index = 9)
period = 6
length = len(temp_test_x.columns) - period
test_x = []
for i in range(length):
    test_x.append([])
for i in range(length):    
    for j in range(len(temp_test_x.index)):
        for k in range(period):
            test_x[i].append(temp_test_x.iloc[j, i+k])


# In[12]:


#split test_y
temp_test_y = temp_test.iloc[9,:]
test_y = []
for i in range(6, len(temp_test_y)):
    test_y.append(temp_test_y[i])


# In[13]:


#transfer str to int
train_x = np.float64(train_x)
train_y = np.float64(train_y)
test_x = np.float64(test_x)
test_y = np.float64(test_y)


# In[16]:


#train linear model
lr_model = LinearRegression()
lr_model.fit(train_x, train_y)
pred_y = model.predict(test_x)


# In[17]:


#caculate error
mean_absolute_error(test_y, pred_y)


# In[19]:


#train random forest model
rf_model = RandomForestRegressor(max_depth=3, random_state=0)
rf_model.fit(train_x, train_y)
pred_y = rf_model.predict(test_x)


# In[20]:


mean_absolute_error(test_y, pred_y)

