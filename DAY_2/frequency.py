# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:50:39 2019

@author: Dinesh Prajapat
"""
# creating a empty dictionary for storing the values
dict1 = {}

# taking input from the user
user_input = input("enter any string :")


for item in user_input:
    if item not in dict1:
        dict1[item] = 1
    else:
        dict1[item] = dict1[item]+1

print(dict1)


 