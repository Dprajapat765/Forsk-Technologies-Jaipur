# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:48:03 2019

@author: Dinesh Prajapat
"""

day1= ('Monday', 'Wednesday', 'Thursday', 'Saturday')

day2 = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')

new_list = []

list1=list(day1)

list2 = list(day2)

for item in list2:
    if item not in list1:
        list1.append(item)
print(tuple(list1))