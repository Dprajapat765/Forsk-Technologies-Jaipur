# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:33:18 2019

@author: Dinesh Prajapat
"""

number = 0

while number < 100:
    number = number + 1
    if number % 3 ==0 and number % 5 == 0:
        print("FizzBuzz")
    elif (number % 3 == 0):
        print("Fizz")
    elif (number % 5 == 0):
        print("Buzz")
    else:
        print(number)
    


