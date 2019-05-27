# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:41:57 2019

@author: Dinesh Prajapat
"""

my_dict = {}

while True:
    user = input("enter the product name and price saperated with comma  :").split()
    if not user:
        break
    if user!="":
        if "".join(user[:-1]) in my_dict:
            my_dict["".join(user[:-1])] = my_dict["".join(user[:-1])] + int(user[-1])
        else:
            my_dict["".join(user[:-1])] = int(user[-1])
        
print(my_dict)
        

dict2={}
while True:
    user2 =input("ente the product name and price saperated with comma : ").split()
    if not user2:
        break
    if user!="":
        if user2[0] in dict2:
            dict2[user2[0]]=dict2[user2[0]]+int(user2[1])
        else:
            dict2[user2[0]]=int(user2[1])
print(dict2)
    
    
    

    