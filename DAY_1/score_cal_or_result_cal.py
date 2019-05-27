# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:39:58 2019

@author: Dinesh Prajapat
"""

# calculating the result or weighted score of the students
print("You have asked the score of the 3 assignment and 2 exams. Each with max score 100")

assignment1 = int(input("Enter the score of first assignment:"))

assignment2 = int(input("Enter the score of second assignment:"))

assingment3 = int(input("Enter the score of third assignment:"))

half_term = int(input("Enter the score of the First exam:"))

yearly = int(input("Enter the score of the fiel Exam:"))

# calculating the weighted score of the student in all the exams

weighted_score = (assignment1 + assignment2 + assingment3)  * 0.1 + (half_term + yearly) * 0.35

#printing the weighted score
print(weighted_score)