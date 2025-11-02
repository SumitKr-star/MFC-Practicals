# MFC-Practicals
# Practical 1: Create and transform vectors and matrices (the transpose vector (matrix) conjugate.
<br>
#Program to transpose a matrix<br>
#using Numpy<br>

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
OUTPUT:-<br>
<br>
&nbsp;&nbsp;&nbsp;<img width="419" height="150" alt="image" src="https://github.com/user-attachments/assets/4c0381ab-7b12-438d-a7a0-7e352f801ab1" />
<br>
<br>

# Practical 2: Generate teh matrix into echelon form and find its rank. 
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
<br>
OUTPUT:-<br>
<br>
&nbsp;&nbsp;&nbsp;<img width="441" height="144" alt="image" src="https://github.com/user-attachments/assets/4d8e6475-7154-4ce8-9d08-9a9dce15d20f" />





