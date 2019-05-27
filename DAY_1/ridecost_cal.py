# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:29:38 2019

@author: Dinesh Prajapat
"""



# distance in kilometers

total_distance = 80

# average of the vehicle in km/litre

average  = 18

# cost of the iesel per litre in indian rupee
cost_of_diesel = 80

# calculating the fuel consumed by the car in given distance
fuel_consumed = total_distance / average

#printing the fuel consumed by the vehicle
print(fuel_consumed)

# now calculating the cost of driving per day 

cost_of_driving = fuel_consumed * cost_of_diesel 

# printing the cost of the driving
print(cost_of_driving)