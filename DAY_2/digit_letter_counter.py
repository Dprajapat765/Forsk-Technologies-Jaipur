# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:05:28 2019

@author: Dinesh Prajapat
"""

digit_dict1 = {"digit":0,"letter":0}

user_input = input("enter the string:")

for item in user_input:
    if item.isdigit():
        digit_dict1["digit"] = digit_dict1["digit"] + 1
    else:
        digit_dict1["letter"] = digit_dict1["letter"] +1
        
print(digit_dict1)

        