# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:35:51 2019

@author: Dinesh Prajapat
"""
import csv

# defining the function for read the file and print them
def print_file():
    with open("zoo.csv",'r') as my_file:
        my_file = csv.reader(my_file,delimiter = ',')
        for row in my_file:
            print(row)
            
print_file()


# definging the function for the displaying the animals name in group


def group_anim():
    with open("zoo.csv",'r') as anim:
        anim = csv.reader(anim, delimiter = ',')
        next(anim)
        new_list = []
        for row in anim:
            if row[0] not in new_list:
                new_list.append(row[0])
        print(new_list)    

                
group_anim()




# printing teh total number of water need by all the animal


dict1={}

with open("D:/forsk_internship/SUMMER_BOOTCAMP_TRAINING/DAY_4/CHALLANGES/zoo.csv",'rt') as water:
    water = csv.reader(water,delimiter=',')
    next(water)
    for  i in water:
        if  i[0] not in dict1:
            dict1[i[0]] = 1
        else:
            dict1[i[0]] += int(i[2])
    print(dict1)
   
    
    
    
    
# printing the water need of all the animals 

list1 = []
with open("D:/forsk_internship/SUMMER_BOOTCAMP_TRAINING/DAY_4/CHALLANGES/zoo.csv",'rt') as water:
    water = csv.reader(water,delimiter=',')
    next(water)

        
        
            
        
    
    
    
        





    
    
    
    
    
    
    
#    my_list = []
#    for row in water:
#        if row[0] not in my_list:
#            my_list.append(row[0])
#            for col in my_list:
#                if row[2] not in my_list:
#                    my_list.append(row[2])
#                
#    print(my_list)

#
#dict1={}
#with open("zoo.csv",'rt') as water:
#    str1=water.readlines()
#    
#    for i in str1:
#        if i.split()[0] not in dict1:
#            dict1[i.split()[0]]=int(i.split()[2])
#        else:
#            dict1[i.split()[0]]+=int(i.split()[2])
#    print(dict1)
#        
#        
