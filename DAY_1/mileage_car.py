# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:21:34 2019

@author: Dinesh Prajapat
"""

# distance travelled by the car in kilometer
distance_travel = 100

# fuel consumed in the travelling in liter
fuel_consumed = 5 

# calculating te mielage of the car by dividing the kilometers by the litres
mileage_of_car = distance_travel / fuel_consumed

# printing the result
print(mileage_of_car)


''' try using the input form the user '''

distance_travel2 = int(input("Enter the distance travelled by the car:"))

fuel_consumed2 = int(input("Enter the amount of fuel consumed by the car:"))

mileage_of_car2 = distance_travel2 / fuel_consumed2

print(mileage_of_car2)