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

# Practical 4:- Solve a system of Homogeneous and non-homogeneous equations using Gauss elimination method

```
import numpy as np

#Coefficient Matrix (A)

print("Enter the dimensions of coefficient matrix (A):")

NR = int(input("Enter the number of rows: ")) 
NC = int (input("Enter the number of columns:"))

print("Enter the elements of coefficient matrix (A) in a single line (separated by space):")

coeff entries list (map(float,input().split()))

#Create Coefficient Matrix

Coefficient Matrix = np.array(coeffentries).reshape (NR, NC)

print("\nCoefficient Matrix (A) is as follows:\n", Coefficient Matrix, "\n")

Column Matrix (B)

print("Enter the elements of column matrix (B) in a single line (separated by space):")

column entries list (map (float, input().split()))

Column Matrix np.array(column_entries).reshape (NR, 1) print("\nColumn Matrix (B) is as follows:\n", Column Matrix, "\n")

#Solution of System of Equations using Gauss elimination method Solution np.linalg.solve (Coefficient Matrix, Column Matrix) print("Solution of the system of equations using Gauss elimination method:\n")

print (Solution)

```
<br>

# Practical 5:- Solve a system of Homogeneous equations using the Gauss Jordan method.

```
import numpy as np

Coefficient Matrix (A)

print("Enter the dimensions of coefficient matrix (A):")

NR = int(input("Enter the number of TOMS: ")) NC = int (input("Enter the number of columns:"))

print("Enter the elements of coefficient matrix (A) in a single line (separated by space): ") 

coeff entries list (map(float,input().split()))

#Create Coefficient Matrix

Coefficient Matrix =  np.array(coeff_entries).reshape (NR, NC) print("\nCoefficient Matrix (A) is as follows:\n", Coefficient Matrix, "\n")

Column Matrix (B)

print("Enter the elements of column matrix (B) in a single line (separated by space):") 
column entries = list(map(float, input().split()))

Column Matrix = np.array(column entries).reshape (NR, 1) 
print("\nColumn Matrix (B) is as follows:\n", Column_Matrix, "\n")

#Solution of System of Equations using Gauss-Jordan (Matrix Inversion) Inv_of Coefficient Matrix np.linalg.inv(Coefficient_Matrix) Solution of the system of Equations np.matmul (Inv of Coefficient Matrix, Column Matrix)

print("Solution of the system of equations using Gauss-Jordan method") print (Solution_of_the_system_of_Equations)

```
<br>

# Practical 6:- Generate basis of column space, null space, row space and left null space of a matrix space.

```

import numpy as np

# Coefficient Matrix (A) Elements
print("Enter the dimensions of Matrix (A):")
NR = int(input("Enter the number of rows: "))
NC = int(input("Enter the number of columns: "))

print("Enter the elements of Matrix (A) in a single line (separated by space):")
Entries = list(map(float, input().split()))

# Create Matrix A
A_np = np.array(Entries).reshape(NR, NC)
print("\nMatrix (A) is as follows:\n", A_np, "\n")

# Convert NumPy array to SymPy Matrix
A = Matrix(A_np)

# Null Space of A
NullSpace_list = A.nullspace()   # Returns a list of vectors

if NullSpace_list:
    NullSpace = Matrix.hstack(*NullSpace_list)  # Combine vectors into a matrix
else:
    NullSpace = Matrix([])  # Empty matrix if null space is zero vector only

print("Null Space of Matrix (A) is:\n", NullSpace, "\n")

# Check whether NullSpace satisfies A * NullSpace = 0
print("Checking whether NullSpace satisfies A * NullSpace = 0 ...\n")
if NullSpace_list:
    print("A * NullSpace =\n", A * NullSpace, "\n")
else:
    print("A * NullSpace = [0] (Trivial Null Space)\n")

# Python Code for Rank and Nullity of Matrix
NoC = A.shape[1]             # Number of columns
rank = A.rank()              # Rank of matrix
nullity = NoC - rank         # Nullity of matrix

print("Rank of Matrix (A) =", rank)
print("Nullity of Matrix (A) =", nullity)

```
<br>
<br>

# Practical 7:- Check the linear dependence of vectors. Generate a linear combination of given vectors of Rn/matrices of the same size and find the transition matrix of given matrix space.

```

import numpy as np

# ---- Input vectors ----
n = int(input("Enter number of vectors: "))
m = int(input("Enter dimension of each vector: "))

vecs = [list(map(float, input(f"Vector {i+1}: ").split())) for i in range(n)]
A = np.column_stack(vecs)

print("\nMatrix A:\n", A)

# ---- Check Linear Dependence ----
r = np.linalg.matrix_rank(A)
print("\nRank =", r)

print("Linearly Independent" if r == A.shape[1] else "Linearly Dependent")

# ---- Linear Combination ----
coef = list(map(float, input("\nEnter coefficients: ").split()))
lc = np.sum([coef[i] * A[:, i] for i in range(n)], axis=0)
print("\nLinear Combination:", lc)

# ---- Transition Matrix ----
s = int(input("\nEnter size of square matrices: "))
print("Enter B1:")
B1 = np.array([list(map(float, input().split())) for _ in range(s)])

print("Enter B2:")
B2 = np.array([list(map(float, input().split())) for _ in range(s)])

try:
    P = np.linalg.inv(B2) @ B1
    print("\nTransition Matrix:\n", P)
except np.linalg.LinAlgError:
    print("B2 is not invertible.")

```
<br>
<br>

# Practical 8:-Find the orthonormal basis of a given vector space using the Gram-Schmidt orthogonalization process Code:

```
import numpy as np

# --- Function to take input vectors ---
def input_vectors():
    n = int(input("Enter number of vectors: "))
    m = int(input("Enter dimension of each vector: "))
    vectors = []
    for i in range(n):
        v = list(map(float, input(f"Enter elements of vector {i+1}: ").split()))
        vectors.append(np.array(v))
    return vectors

# --- Gram-Schmidt Orthonormalization ---
def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        for u in orthonormal:
            v = v - np.dot(v, u) * u       # Make orthogonal
        v = v / np.linalg.norm(v)          # Normalize
        orthonormal.append(v)
    return orthonormal

# --- Main Program ---
vectors = input_vectors()
basis = gram_schmidt(vectors)

print("\nOrthonormal Basis:")
for i, b in enumerate(basis, 1):
    print(f"u{i} =", np.round(b, 3))
```

<br>
<br>

# Practical 9:-Check the diagonalizable property of matrices and find the corresponding eigenvalue and verify the Cayley- Hamilton theorem.
<br>
```
import numpy as np

# --- Take matrix input from user ---
def input_matrix():
    n = int(input("Enter size of square matrix: "))
    A = []
    for i in range(n):
        A.append(list(map(float, input(f"Row {i+1}: ").split())))
    return np.array(A)

# --- Find eigenvalues and eigenvectors ---
def eigen_info(A):
    values, vectors = np.linalg.eig(A)
    return values, vectors

# --- Check diagonalizable or not ---
def check_diagonalizable(A):
    values, vectors = eigen_info(A)
    if np.linalg.matrix_rank(vectors) == len(A):
        print("\nMatrix is Diagonalizable.")
        print("Eigenvalues:", np.round(values, 3))
    else:
        print("\nMatrix is NOT Diagonalizable.")
        print("Eigenvalues:", np.round(values, 3))

# --- Verify Cayley-Hamilton theorem ---
def verify_cayley_hamilton(A):
    p = np.poly(A)        # coefficients of characteristic polynomial
    n = len(A)

    result = sum([p[i] * np.linalg.matrix_power(A, n - i - 1) for i in range(n)]) + p[-1] * np.eye(n)

    if np.allclose(result, 0):
        print("Cayley-Hamilton Theorem Verified.")
    else:
        print("Cayley-Hamilton Theorem Not Verified.")

# --- Main Program ---
A = input_matrix()
check_diagonalizable(A)
verify_cayley_hamilton(A)
```
<br>
# Practical 10:-Application of Linear algebra: Coding and decoding of messages using nonsingular matrices. eg code "Linear Algebra is fun" and then decode it.
<br>
```
import numpy as np

# --- Encode function ---
def encode(msg, key):
    msg = msg.upper().replace(" ", "")
    nums = [ord(c) - 64 for c in msg]      # A=1, B=2, ...
    
    while len(nums) % len(key) != 0:       # padding if needed
        nums.append(0)
    
    nums = np.array(nums).reshape(-1, len(key))
    coded = nums.dot(key)
    return coded

# --- Decode function ---
def decode(coded, key):
    inv_key = np.linalg.inv(key)
    decoded = coded.dot(inv_key).flatten()
    text = ''.join(chr(int(round(n)) + 64) if n > 0 else ' ' for n in decoded)
    return text

# --- Main ---
message = "LINEAR ALGEBRA IS FUN"
key = np.array([[2, 1], [1, 1]])          # simple nonsingular matrix

print("Original Message:", message)

coded = encode(message, key)
print("\nEncoded Matrix:\n", np.round(coded))

decoded = decode(coded, key)
print("\nDecoded Message:", decoded)

```
<br>
<br>
# Practical 11:-Compute Gradient of a scalar field.
<br>
```
import sympy as sp

def gradient():
    x, y = sp.symbols('x y')     # two variables
    f = input("Enter scalar function f(x, y): ")   # example: x**2 + 3*y
    f = sp.sympify(f)
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)
    print("\nGradient of f(x, y):")
    print("∇F = ({fx}, {fy})")

# --- Main ---
gradient()

```
<br>
<br>
# Practical 12:-Compute Divergence of a vector field.
<br>

```
import sympy as sp

def divergence():
    # define variables
    x, y, z = sp.symbols('x y z')

    # take vector field input
    Fx = input("Enter Fx: ")   # example: x*y
    Fy = input("Enter Fy: ")   # example: y*z
    Fz = input("Enter Fz: ")   # example: z*x

    # convert to symbolic form
    Fx, Fy, Fz = sp.sympify(Fx), sp.sympify(Fy), sp.sympify(Fz)

    # compute divergence
    div = sp.diff(Fx, x) + sp.diff(Fy, y) + sp.diff(Fz, z)

    # print result
    print("\nDivergence of F = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z")
    print("Divergence =", div)

# --- Main ---
divergence()

```
<br>
<br>
# Practical 13:-Compute Curl of a vector field.
<br>
```
import sympy as sp

def curl():
    # Define symbols
    x, y, z = sp.symbols('x y z')

    # Read components of the vector field as strings and convert to SymPy
    Fx = sp.sympify(input("Enter Fx: "))
    Fy = sp.sympify(input("Enter Fy: "))
    Fz = sp.sympify(input("Enter Fz: "))

    # Partial derivatives for curl F = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
    curl_x = sp.diff(Fz, y) - sp.diff(Fy, z)
    curl_y = sp.diff(Fx, z) - sp.diff(Fz, x)
    curl_z = sp.diff(Fy, x) - sp.diff(Fx, y)

    print("curl F =", (curl_x, curl_y, curl_z))

# Run
curl()

```