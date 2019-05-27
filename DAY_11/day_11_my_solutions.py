# -*- coding: utf-8 -*-
"""
Created on Mon May 20 03:35:19 2019

@author: Dinesh Prajapat
"""
"""
Code Challenge
  Name: 
    Space Seperated data
  Filename: 
    space_numpy.py
  Problem Statement:
    You are given a 9 space separated numbers. 
    Write a python code to convert it into a 3x3 NumPy array of integers.
  Input:
    6 9 2 3 5 8 1 5 4
  Output:
      [[6 9 2]
      [3 5 8]
      [1 5 4]]
  
"""

# importing the library
import numpy as np

# creating a list saperated with space
list1 = [6,9,2,3,5,8,1,5,4]

# converting the list into nd array 
list2 = np.array(list1)

# giving the shape of 3 X 3 array

list2 = list2.reshape(3,3)

# printing the list
print(list2)




"""
Code Challenge
  Name: 
    Random Data
  Filename: 
    random_data.py
  Problem Statement:
    Create a random array of 40 integers from 5 - 15 using NumPy. 
    Find the most frequent value with and without Numpy.
  Hint:
      Try to use the Counter class
      
"""

''' with the numpy '''
# importing the library
import numpy as np

# generating the 40 digits from the range of 5 to 15
num = np.random.randint(5,15,40)

# using the library counting the frequency of digits
new = np.bincount(num).argmax()
print("Number which occure most in range(with numpy):",new)

''' without numpy solutions '''

# converting the nd array to the list
list2 = list(num)

# finding the highest appearance of the digit to get the maximum out of it.
new2 = max(set(num),key=list2.count)

print("Number which occure most in range(without numpy):",new2)





"""
Code Challenge
  Name: 
    E-commerce Data Exploration
  Filename: 
    ecommerce.py
  Problem Statement:
      To create an array of random e-commerce data of total amount spent per transaction. 
      Segment this incomes data into 50 buckets (number of bars) and plot it as a histogram.
      Find the mean and median of this data using NumPy package.
      Add outliers 
          
  Hint:
      Execute the code snippet below.
      import numpy as np
      import matplotlib.pyplot as plt
      incomes = np.random.normal(100.0, 20.0, 10000)
      print (incomes)
 
    outlier is an observation that lies an abnormal distance from other values 
    
"""



import numpy as np

import matplotlib.pyplot as plt

incomes = np.random.normal(100.0, 20.0, 10000)
#print (incomes)

# sorting the income data 
incomes.sort()

print(incomes)

income2 = list(incomes)
#in the code ploting the histogram visual of the incomes
plt.hist(income2,bins=50)
# extra 
#plt.hist(income2,bins=[25,35,45,55,65,75,85,95,105,115,125,135,145,155,165,175,185,195])

# finding the mean of the incomes

print("Mode is:",np.mean(income2))

# finding the median of the incomes

print("Median is:",np.median(income2))





"""
Code Challenge
  Name: 
    Normally Distributed Random Data
  Filename: 
    normal_dist.py
  Problem Statement:
    Create a normally distributed random data with parameters:
    Centered around 150.
    Standard Deviation of 20.
    Total 1000 data points.
    
    Plot the histogram using matplotlib (bucket size =100) and observe the shape.
    Calculate Standard Deviation and Variance. 
"""


import matplotlib.pyplot as plt

import numpy as np

# generating the normal distributed data from the parameters

#data = np.random.normal(150,20,1000) #new data size

data = np.random.normal(150,20,1000)
# sorting the data
data.sort()

# ploting the histogram visuals
plt.hist(income2,bins=100)

# calculatinig the daviance of the data
print("Daviance is:",np.std(data))

# calculating the varience of the data
print("Varience is:",np.var(data))

