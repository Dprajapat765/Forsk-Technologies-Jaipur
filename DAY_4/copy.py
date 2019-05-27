# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:30:20 2019

@author: Dinesh Prajapat
"""

with open('student.txt','rt') as file:
    with open('student2.txt','wt') as new_file:
        for line in file:
            new_file.write(line)





