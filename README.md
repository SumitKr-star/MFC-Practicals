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
<br>
<br>

# Practical 3: Find Co-factors, Determinant, Adjoint and Inverse of a matrix. 

```
import numpy as np

nr=int(input("enter the number of rows"))

nc=int(input("enter the number of coloums"))

print('enter the entries in a single line seperated by spaces:')

entries=list(map(int, input().split()))

A=np.array(entries).reshape(nr,nc)

print("matrix x is a follows;", '\n',A)

A_inverse-np.linalg.inv(A)

transpose_of_A_inverse=np.transpose (A_inverse)

determinant_of_A=np.linalg.det (A)

cofactor_of_A=np.dot (transpose_of_A_inverse, determinant_of_A) print("the cofactor of a matrix is:", '\n',cofactor_of_A)

print("the determinant of a matrix is:",'\n', determinant_of_A)

Adjoint_of_A=np.transpose (cofactor_of_A)

print("the Adjoint of a matrix is:", '\n', Adjoint_of_A)

```

#Practical 4:- 



