# This is a file with all the metods to solve a system of linear equasions and
# systems of differential equasions.
# Author: Armin Grobbelaar
# Student number: u9018178 (WTW383)
import numpy as np
import math 
np.set_printoptions(suppress=True)
#Systems of linear equasions
#Gauss elimination with back substitution
def guass(A, b):
    n=len(A)
    A=np.hstack((A, b))
    
    #Elimination process
    for i in range(0, n-1):
        
        # step 2
        p=i
        while p <= n and A[p][i] ==0:
            p=p+1
            
        if p == n+1:
            raise ValueError("Matrix A is singular")
        
        # step 3
        if p!=i:
            A[[i, p]] = A[[p, i]]
        
        print(A)
        # step 4 => to do: step 5 &6
        for j in range(i+1, n):
            m = A[j][i]/A[i][i]
            for k in range(i, n+1):
                A[j][k] = A[j][k] - m*A[i][k]
    print("")
    print(A)    
    #step7
    if A[n-1][n-1] == 0:
        raise ValueError("Matrix A is singular")
        
    #Back Subsitution
    x_n = (A[n-1][n]/A[n-1][n-1])
    x = np.zeros(n)
    x[n-1] = x_n
    sum=0
    for i in range(n-1, -1, -1):
        sum=0
        for j in range(i+1, n):
            sum=sum + A[i][j]*x[j]
            
        x[i] = (A[i][n] - sum)/A[i][i]
        sum=0
    print("")
    print(x)
    

#Gauss-Jordan Hybrid elimination
def guass_jordan_hybrid(A, b):
    n=len(A)
    A=np.hstack((A, b))
    
    print(A)
    print("")
    
    #Elimination process
    for i in range(0, n-1):
        
        p=i
        while p <= n and A[p][i] == 0:
            p=p+1
            
        if p == n+1:
            raise ValueError("Matrix A is singular")
        
        if p!=i:
            A[[i, p]] = A[[p, i]]
        
        for j in range(i+1, n):
            m = A[j][i]/A[i][i]
            for k in range(i, n+1):
                A[j][k] = A[j][k] - m*A[i][k]
                
    print(A)
    print("")
    
    if A[n-1][n-1] == 0:
        raise ValueError("Matrix A is singular")
    
    for i in range(0, n):
        const = A[n-i-1][n-i-1]   
        for j in range(i+1, n):
            m = A[n-j-1][n-i-1] / const
            for k in range(0, n+1):
                A[n-j-1][k] = A[n-j-1][k] - m*A[n-i-1][k]
    
    #Get final answer
    print("")
    print(A)
    x=[]
    for p in range(0, n):
        x.append(A[p][n] / A[p][p])
        
    print(x)
    
A = np.array([[3.33, 159, 10.3], [2.22, 16.7, 9.61], [-1.56, 5.18, 1.69]])
b = np.array([[795], [0.965], [2.71]])


#Factor LU -> LU Factorisation
def lu_factorisation(A):
    n = len(A)
    
    L = np.zeros((n, n), float)
    np.fill_diagonal(L, 1)
    
    U = np.zeros((n, n), float)
    
    U[0][0] = A[0][0] / L[0][0]
    if (L[0][0] * U[0][0] == 0):
        raise ValueError("Factorisation impossible")
    
    for j in range(1, n):
        U[0][j] = A[0][j] / L[0][0]
        L[j][0] = A[j][0] / U[0][0]
        
        
    for i in range(1, n-1):

        for j in range(i+1, n):
            sums=0
            for k in range(0, i):
                sums=sums+L[i][k]*U[k][i]
            U[i][i] = (A[i][i] - sums) / L[i][i]
            if(L[i][i] * U[i][i] == 0):
                raise ValueError("Factorisation impossible")
                
            for j in range(i+1, n):
                sum_l=0
                sum_u=0
                for k in range(0, i):
                    sum_u=sum_u+L[i][k]*U[k][j]
                    sum_l=sum_l+L[j][k]*U[k][i]
                       
                U[i][j] = (1/L[i][i])*(A[i][j] - sum_u)
                L[j][i] = (1/U[i][i])*(A[j][i] - sum_l)
                
    
    sum_n=0
    for k in range(0, n):
        sum_n=sum_n+L[n-1][k]*U[k][n-1]
        
    U[n-1][n-1] = (A[n-1][n-1] - sum_n) / L[n-1][n-1]
    if(L[n-1][n-1]*U[n-1][n-1] == 0):
        print("Note: A=LU, but A is Singular, A being the matrix inputted in the function")
    
    print("A:")
    print(A)
    print("")
    print("L:")
    print(L)
    print("")
    print("U:")
    print(U)

#Solve Ly=b    
def forward(L, b):
    n = len(L)
    y=np.zeros(n, float)
    
    y[0]=b[0] / L[0][0]
    
    for i in range(0, n):
        
        sums=0
        for j in range(0, i):
            sums += L[i][j]*y[j]
        
        y[i] = (b[i] - sums) / L[i][i]
    
    print("")
    print(y)

#Solve Ux=y    
def backward(U, y):
    
    n=len(U)
    x=np.zeros(n, float)
    
    x[n-1]= y[n-1] / U[n-1][n-1]
    
    for i in range(n-1, -1, -1):
        
        sums=0
        for j in range(i+1, n):
            sums += U[i][j]*x[j]
            
        x[i] = (y[i] - sums) / U[i][i]
        
    print("")
    print(x)

def jacobi_iterative_method(A, b, x_0, TOL, N):
    
    n=len(A)
    x=np.zeros(n, float)
    k=1
    while k < N or k==N:
    
        for i in range(0, n):
            
            sum0=0
            for j in range(0, i):
                sum0 += A[i][j] * x_0[j]
                
            sum1=0
            for j in range(i+1, n):
                sum1 += A[i][j] * x_0[j]
                
            x[i] = (1/A[i][i]) *(-sum0 - sum1 + b[i])
        
            error = np.linalg.norm(x - x_0) / np.linalg.norm(x_0)
            if (error < TOL or error==TOL):
            
             k += 1
             print("Current solution: ")
             print(x)
        
            for i in range(0, n):
             x_0[i] = x[i]
    
    if k>N:
        print("Maximum number of iterations exceeded.")
    
def gauss_seidel_iterative_method(A, b, x_0, TOL, N):
    
    n = len(A)
    k=1
    x = np.zeros(n, float)
    
    while k < N:
        for i in range(0, n):
            
            sum0=0
            for j in range(0, i):
                sum0 = sum0 + A[i][j] * x[j]
                
            sum1=0
            for j in range(i+1, n):
                sum1 = sum1 + A[i][j] * x_0[j]
                
            x[i] = (1/A[i][i]) *(-sum0 -sum1 +b[i])
            
            error = np.linalg.norm(x - x_0) / np.linalg.norm(x_0)
            if (error < TOL or error==TOL):
             
             k += 1
             print("Current solution: ")
             print(x)
        
            for i in range(0, n):
             x_0[i] = x[i]
             
             
    if k>N:
        print("Maximum number of iterations exceeded.")

#Solving differential equasions


def derivative(t, y):
  f=y-t**2+1
  return(f)  
  
def euler_method(a, b, N, initial_value):
    
    h= (b-a)/N
    t=a
    w=initial_value
    print("t:")
    print(t)
    print("w:")
    print(w)
    
    for i in range(1, N+1):
        
        w=w+h*(derivative(t, w))
        t=a+i*h
        print("")
        print("t:")
        print(t)
        print("w:")
        print(w)

    
