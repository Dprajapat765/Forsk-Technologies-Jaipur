# -*- coding: utf-8 -*-
"""
Created on Mon May 20 05:42:55 2019

@author: Dinesh Prajapat
"""
# importing the libraries
from bs4 import BeautifulSoup as bs
import requests

# getting the url and text of the website
url = "https://en.wikipedia.org/wiki/List_of_state_and_union_territory_capitals_in_India"

# getting the text
data = requests.get(url).text

# adding the beautiful soup here
data2 = bs(data,"lxml")

print(data2)

# lets prettify the data of the website
data2.prettify()

# get the data of the table using find all attribute
data2.findAll("table")

# now create some list to store the data of the website
A = []
B = []

# now use the for loop to get the table data form row of the table
for raw in data2.findAll("tr"):
    # getting the table data
    col = data2.find("td")
    head = data2.find("th")
    
        


