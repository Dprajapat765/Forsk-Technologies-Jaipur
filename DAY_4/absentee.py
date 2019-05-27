# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:36:12 2019

@author: Dinesh Prajapat
"""


n = 1
while n<=25:
    user = input("enter the {} student name:".format(n))
    
    n = n+1
    
    if user =="":
        break
    
    with open("student.txt",'at') as my_file:
        my_file.write(user+'\n')
    

with open("student.txt",'rt') as file:
    for row in file:
        my_row = file.readlines()
        print(my_row)
        









