# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:54:25 2019

@author: Dinesh Prajapat
"""

'''  hands on 1 '''
# Create a list of number from 1 to 20 using range function. 
# Using the slicing concept print all the even and odd numbers from the list 


list1 = list(range(0,21))

# printing the even number from the list using the silicing concept

list1[0::2]

# printing the odd number from the list using the slicing concept

list1[1::2]

''' hands on 2 '''

# Make a function to find whether a year is a leap year or no, return True or False 

# created a function
def leap_year(year):
    num = True
    if year % 400 ==0 or year % 4 == 0 and year % 100 != 0:
        print(num)
    else:
        return False

# taking the input from the user
year = int(input("enter the year:"))
# calling the function
leap_year(year)





''' hands on 3 '''
# Make a function days_in_month to return the number of days in a specific month of a year

# importing the library
import calendar 


# defining the function 
def days_in_month(year,month):
    get_days = calendar.monthlen(year,month)
    return get_days

# taking input of the month in the 
year = int(input("enter the year:"))

month = int(input("enter the month:"))

days_in_month(year,month)







