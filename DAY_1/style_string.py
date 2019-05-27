# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:10:02 2019

@author: Dinesh Prajapat
"""

# lets try tostyle the string 
print("We will change the string to uppercase. lowercase and into camel case")

user_string = input("Enter teh string to perform following operations: ")

# give the user to choose what kind of operaion he want to perform
print("Enter the choice -- \n 1. change to lowercase \n 2. change to uppercase \n 3. change to camelcase")

# user choice
user_choice = input("enter your choice:")



if user_choice == "1":
    print(user_string.lower())
elif user_choice == "2":
    print(user_string.upper())
elif user_choice == "3":
    print(user_string.capitalize())
