# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:13:50 2019

@author: Dinesh Prajapat
"""




"""
Code Challenge 

import Automobile.csv file.

Using MatPlotLib create a PIE Chart of top 10 car makers according to the number 
of their cars and explode the largest car maker


"""

# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# reading the csv file 
data = pd.read_csv("D:/forsk_internship/SUMMER_BOOTCAMP_TRAINING/DAY_13/Automobile.csv")

# getting the values of the data
new = data['make'].value_counts()

# getting the top 10 values
series = new.values[0:10]

lable = new.index[0:10]

explodes = (0.2,0,0,0,0,0,0,0,0,0)
# using the matplot lib ploting the pie chart
plt.pie(series,explode=explodes,labels=lable,autopct='%1.1f%%')






"""
Code Challenge
  Name: 
    Baltimore City Analysis
  Filename: 
    baltimore.py
  Problem Statement:
    Read the Baltimore_City_Employee_Salaries_FY2014.csv file 
    and perform the following task :

    0. remove the dollar signs in the AnnualSalary field and assign it as a float
    0. Group the data on JobTitle and AnnualSalary, and aggregate with sum, mean, etc.
       Sort the data and display to show who get the highest salary
    0. Try to group on JobTitle only and sort the data and display
    0. How many employess are there for each JobRoles and Graph it
    0. Graph and show which Job Title spends the most
    0. List All the Agency ID and Agency Name 
    0. Find all the missing Gross data in the dataset 
    
  Hint:

import pandas as pd
import requests
import StringIO as StringIO
import numpy as np
        
url = "https://data.baltimorecity.gov/api/views/2j28-xzd7/rows.csv?accessType=DOWNLOAD"
r = requests.get(url)
data = StringIO.StringIO(r.content)

dataframe = pd.read_csv(data,header=0)

dataframe['AnnualSalary'] = dataframe['AnnualSalary'].str.lstrip('$')
dataframe['AnnualSalary'] = dataframe['AnnualSalary'].astype(float)

# group the data
grouped = dataframe.groupby(['JobTitle'])['AnnualSalary']
aggregated = grouped.agg([np.sum, np.mean, np.std, np.size, np.min, np.max])

# sort the data
pd.set_option('display.max_rows', 10000000)
output = aggregated.sort(['amax'],ascending=0)
output.head(15)


aggregated = grouped.agg([np.sum])
output = aggregated.sort(['sum'],ascending=0)
output = output.head(15)
output.rename(columns={'sum': 'Salary'}, inplace=True)


from matplotlib.ticker import FormatStrFormatter

myplot = output.plot(kind='bar',title='Baltimore Total Annual Salary by Job Title - 2014')
myplot.set_ylabel('$')
myplot.yaxis.set_major_formatter(FormatStrFormatter('%d'))



"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# reading the csv file 
batli = pd.read_csv("D:/forsk_internship/SUMMER_BOOTCAMP_TRAINING/DAY_13/Baltimore_City_Employee_Salaries_FY2014.csv")
new2 = batli['AnnualSalary'].astype(np.float64)


new2 = new2.to_frame()




"""
Code Challenge
  Name: 
    SSA Analysis
  Filename: 
    ssa.py
  Problem Statement:
    (Baby_Names.zip)
    The United States Social Security Administration (SSA) has made available 
    data on the frequency of baby names from 1880 through the 2010. 
    (Use Baby_Names.zip from Resources)  
    
    Read data from all the year files starting from 1880 to 2010, 
    add an extra column named as year that contains year of that particular data
    Concatinate all the data to form single dataframe using pandas concat method
    Display the top 5 male and female baby names of 2010
    Calculate sum of the births column by sex as the total number of births 
    in that year(use pandas pivot_table method)
    Plot the results of the above activity to show total births by sex and year  
     
"""



# reading the files using for loop
df1 = pd.DataFrame['Name','Sex','Number','Year']
for i in range(1880,2011):
    file_name = 'yob'+str(i)+'.txt'

    













