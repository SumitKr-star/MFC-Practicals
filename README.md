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
<img width="419" height="150" alt="image" src="https://github.com/user-attachments/assets/4c0381ab-7b12-438d-a7a0-7e352f801ab1" />
<br>
<br>

# Practical 2: 

```
#NR = Number of Rows
#NC = Number of Columns

import numpy as np

NR = int(input("Enter the number of Rows: "))
NC = int(input("Enter the number of Columns: "))

print("Enter the entries in a single line (separated by space): ")

# Single line separated by space 
entries = list(map(int, input().split()))

#For Printing the Matrix

matrix = np.array (entries).reshape(NR, NC)
print("Matrix X is as follows:", '\n', matrix)


# For finding the Rank of a Matrix
rank = np.linalg.matrix_rank(matrix)
print("The Rank of a Matrix is:", rank)
```





