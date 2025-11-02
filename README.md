8# MFC-Practicals
# Practical 1: Create and transform vectors and matrices (the transpose vector (matrix) conjugate
#transpose of a vector (matrix)<br>
<br>
#Program to transpose a matrix<br>
#using Numpy<br>
<br>
```
#NR: Number of Rows
#NC: Number of Columns

import numpy as np

NR=int(input("enter no. of rows: "))
NC=int(input("enter no. of columns: "))

print("Enter the entries in a single line (separated by space): ")

# Single line separated by space
entries= list(map(int, input().split()))

#For Printing the matrix
A=np.array(entries).reshape(NR,NC)

#For transposing the matrix
Transpose= np.transpose(A)
print('transpose of matrix A:-', Transpose)
```
<br>
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/db69a7f3-6dd6-4ecc-9fab-2c329eab92bf" />





