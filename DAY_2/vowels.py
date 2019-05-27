# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:45:43 2019

@author: Dinesh Prajapat
"""

# remove the vowels form the list using loop


list2 = []

state_name = [ 'Alabama', 'California', 'Oklahoma', 'Florida']

vowels  = list("AEIOUaeiou")

for i in state_name:
    temp_list = []
    for j in i:
        if j not in vowels:
            temp_list.append(j)
    list2.append("".join(temp_list))
print(list2)


















