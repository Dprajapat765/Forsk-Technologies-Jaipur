# -*- coding: utf-8 -*-
"""
Created on Tue May 21 03:44:44 2019

@author: Dinesh Prajapat
"""


"""
Code Challenge
  Name: 
    Titanic Analysis
  Filename: 
    titanic.py
  Dataset:
    training_titanic.csv
  Problem Statement:
      It’s a real-world data containing the details of titanic ships 
      passengers list.
      Import the training set "training_titanic.csv"
  Answer the Following:
      How many people in the given training set survived the disaster ?
      How many people in the given training set died ?
      Calculate and print the survival rates as proportions (percentage) 
      by setting the normalize argument to True.
      Males that survived vs males that passed away
      Females that survived vs Females that passed away
      
      Does age play a role?
      since it's probable that children were saved first.
      
      Another variable that could influence survival is age; 
      since it's probable that children were saved first.

      You can test this by creating a new column with a categorical variable Child. 
      Child will take the value 1 in cases where age is less than 18, 
      and a value of 0 in cases where age is greater than or equal to 18.
 
      Then assign the value 0 to observations where the passenger 
      is greater than or equal to 18 years in the new Child column.
      Compare the normalized survival rates for those who are <18 and 
      those who are older. 
    
      To add this new variable you need to do two things
        1.     create a new column, and
        2.     Provide the values for each observation (i.e., row) based on the age of the passenger.
    
  Hint: 
      To calculate this, you can use the value_counts() method in 
      combination with standard bracket notation to select a single column of
      a DataFrame
"""
      

# importing the library 
import pandas as pd

# reading the data of the file
df2 = pd.read_csv("D:/forsk_internship/SUMMER_BOOTCAMP_TRAINING/DAY_12/training_titanic.csv") 

# counting the survied people of the ship
survived = df2["Survived"].value_counts()

# replacing the 0 and 1 with lable
survived.index=['Not Survived','Survived']
print("Survived People in the Training:",survived[1])

# counting the people are died
survived.index=['Died','Survived']
print("Died People in the Training:",survived[0])


# showing the percentage of the survived ratio
percent = df2['Survived'].value_counts(normalize=True)
print("Percentage of survival: ",percent[1])


# count the total males

male = df2.groupby(['Survived','Sex'])

male2 = male.groups
male2 = male.count()

male2.keys()



# create a new column in the data set with the name of Child

'''

==================performing this code using apply method===================

def child_or_not(child_age):
    if child_age > 18:
        return 0
    else:
        return 1

# using apply method for adding the data to the child column
df2['Child'] = df2['Age'].apply(child_or_not)


'''

'''
===================perfirming the code using the lambda method ===============
'''

df2['Age'] = df2['Age'].fillna(df2['Age'].mean()).astype(pd.np.int64)

df2['child_new'] = df2['Age'].map(lambda i: 0 if i > 18 else 1)

child = df2.groupby(['Survived','child_new'])

child.groups







"""
Code Challenge
  Name: 
      Exploratory Data Analysis - Automobile
  Filename: 
      automobile.py
  Dataset:
      Automobile.csv
  Problem Statement:
      Perform the following task :
      1. Handle the missing values for Price column
      2. Get the values from Price column into a numpy.ndarray
      3. Calculate the Minimum Price, Maximum Price, Average Price and Standard Deviation of Price
"""

import pandas as pd
import numpy as np
atm = pd.read_csv("D:/forsk_internship/SUMMER_BOOTCAMP_TRAINING/DAY_12/Automobile.csv")

# check the column that have atleast one missing values iin price column
atm[atm['price'].isnull()]

#1.  repacing the missing values with the mean of that column  and convert it into the int64
atm['price'] = atm['price'].fillna(atm['price'].mean()).astype(pd.np.int64)
print(atm['price'])
#2.  getting tthe values from price column in nd array
arr = np.array(atm['price'])
print(arr)

#3.  calculating the minimum , maximum , average ad the standerd diviations of the price
print("Minimum of the Price Column:",atm['price'].min())

print("Maximum of the Price Column:", atm['price'].max())

print("Average/Mean of the Price Column:",atm['price'].mean())

print("Standerd Deviation of the Price Column:",atm['price'].std())



















