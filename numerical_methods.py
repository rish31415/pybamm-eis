# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:10:36 2022

@author: rish3
"""
import numpy as np
import scipy.sparse.linalg
from scipy.sparse import csr_matrix

def get_matrix_problem(w, N):
    # returns the matrix A and vector b for the matrix problem we need to solve
    # Uses ghost points
    j_hat = 1
    D = 1
    a1 = 0
    a2 = -j_hat/D
    second_derivative_matrix = np.diag(
        np.full(N+3, -2)) + np.diag(np.full(N+2, 1), -1) + np.diag(np.full(N+2, 1), 1)
    A = D*second_derivative_matrix - w*1.j*np.identity(N+3)/N**2

    # c_1 - c_-1 = 2*a1/N
    # c_N+1 - c_N-1 = 2*a2/N
    A[0][0] = -1
    A[0][1] = 0
    A[0][2] = 1
    A[N+2][N] = -1
    A[N+2][N+1] = 0
    A[N+2][N+2] = 1

    b = [2*a1/N] + list(np.zeros(N+1)) + [2*a2/N]
    return A, b

def solve(A, b, method, start_point = 'none'):
    #This file will contain different algorithms to solve Ax = b
    
    if str(start_point) == 'none':
        start_point = np.zeros(np.shape(b))
        
        
    if method == 'cg':
        c = conjugate_gradient(A, b, start_point)
    elif method == 'bicgstab':
        c = bicgstab(A, b, start_point)   
    elif method == 'standard':
        c = np.linalg.solve(A, b)
    
    return c

def conjugate_gradient(A, b, start_point):

    xk = np.array(start_point)
    rk = np.array(b) - A@xk
    pk = rk
    
    max_num_iter = min(100, len(b))
    rk1rk1 = np.dot(np.conj(rk), rk)
    tol = 10**-5
    
    for k in range(max_num_iter):
        Apk = A@pk
        rkrk = rk1rk1
        pkApk = np.dot(np.conj(pk), Apk)
        
        alpha_k = rkrk / pkApk
        
        xk = xk + alpha_k * pk
        
        if alpha_k*pk[-1] < tol:
            break
        else:
            rk = rk - alpha_k * Apk
            
            rk1rk1 = np.dot(np.conj(rk), rk)
            
            beta_k = rk1rk1 / rkrk
            
            pk = rk + beta_k * pk
        
    return xk

def empty(xk):
    pass

def bicgstab(A, b, start_point, callback = empty):

    xk = np.array(start_point)
    rk = np.array(b) - A@xk
    r0 = np.conj(rk)
    rhok = 1
    alpha_k = 1
    wk = 1
    
    pk = np.zeros(np.shape(b))
    vk = pk
    
    max_num_iter = 2*len(b)
    tol = 10**-5
    
    for k in range(1, max_num_iter+1):
        rhok1 = rhok
        rhok = np.dot(r0, rk)
        beta_k = (rhok/rhok1)*(alpha_k/wk)
        
        pk = rk + beta_k*(pk - wk*vk)
        vk = A@pk
        
        alpha_k = rhok/np.dot(r0, vk)
        h = xk + alpha_k*pk
        
        s = rk - alpha_k*vk
        t = A@s
        
        wk = np.dot(np.conj(t), s)/np.dot(np.conj(t), t)
        xk1 = xk
        xk = h + wk*s
        callback(xk)
        if np.abs(xk[-1] - xk1[-1]) < tol:
            break
        else:
            rk = s - wk*t  
    return xk

def prebicgstab(A, b, Kinv, start_point, callback = empty):

    xk = np.array(start_point)
    rk = np.array(b) - A@xk
    r0 = np.conj(rk)
    rhok = 1
    alpha_k = 1
    wk = 1
    
    pk = np.zeros(np.shape(b))
    vk = pk
    
    max_num_iter = 10*len(b)
    tol = 10**-5
    
    for k in range(1, max_num_iter+1):
        rhok1 = rhok
        rhok = np.dot(r0, rk)
        beta_k = (rhok/rhok1)*(alpha_k/wk)
        
        pk = rk + beta_k*(pk - wk*vk)
        y = Kinv@pk
        
        vk = A@y
        
        alpha_k = rhok/np.dot(r0, vk)
        
        h = xk + alpha_k * y

        s = rk - alpha_k*vk
        z = Kinv@s
        t = A@z
        
        wk = np.dot(np.conj(t), s)/np.dot(np.conj(t), t)
        
        xk1 = xk
        xk = h + wk*z
        callback(xk)
        if np.abs(xk[-1] - xk1[-1]) < tol:
            break
        else:
            rk = s - wk*t  
    return xk

def tdge(diag1, diag2, diag3, b):
    #diag2 and b must have 1 more entry than 1 and 3
    #diag2 cannot have any zeros
    #b is zeros followed by an entry
    
    n = len(diag1)
    alpha = diag2[n]
    for i in range(n):
        diag2[i+1] -= diag3[i]*diag1[i]/diag2[i]
    ghost_point = b[n]/diag2[n]
    ans = (b[n] - ghost_point*alpha)/diag1[n-1]
    return ans

def thomasMethod(diag1, diag2, diag3, b):
    #similar to previous but lists contain square matrices of equal size
    #diag2 and b must have 1 more entry than 1 and 3
    #diag2 cannot have any non-invertible matrices
    #b is zeros followed by k entries, k is block size
    m = len(diag1)
    for i in range(m):
        Q = np.linalg.solve(diag2[i], diag3[i])
        diag2[i+1] -= diag1[i]@Q
    
    Bm = diag2[m]
    k = np.shape(Bm)[0]
    y = b[-k:]
    
    ans = np.linalg.solve(Bm, y)[-1]
    
    return ans

def get_block_diagonals(M, J, k):
    '''
    get_block_diagonals(M, J, k)
    
    converts a scipy sparse csr matrix to a block diagonal storage form
    
    Parameters
    ----------
    
    M : scipy sparse csr matrix 
        Must be block diagonal
    J : scipy sparse csr matrix 
        Must be block tridiagonal
    k : int
        is the block size
    
    Returns
    -------
    
    M_diag : list
        list of all the block matrices on the diagonal of M
        
    (diag1, diag2, diag3) : tuple
        a tuple of lists of block matrices on the 3
        diagonals of J. 1 is below the main diagonal, 2 is the main diagonal, 3
        is above.
    '''
    n = np.shape(M)[0]
    m = int(n/k)
    diag1 = []
    diag2 = []
    diag3 = []
    M_diag = []
    for i in range(m-1):
        diag1.append(scipy.sparse.csr_matrix.todense(J[range(i*k, (i+1)*k), :][:, range((i+1)*k, (i+2)*k)]).astype('complex'))
        diag2.append(scipy.sparse.csr_matrix.todense(J[range(i*k, (i+1)*k), :][:, range((i)*k, (i+1)*k)]).astype('complex'))
        diag3.append(scipy.sparse.csr_matrix.todense(J[range((i+1)*k, (i+2)*k), :][:, range((i)*k, (i+1)*k)]).astype('complex'))
        M_diag.append(scipy.sparse.csr_matrix.todense(M[range(i*k, (i+1)*k), :][:, range((i)*k, (i+1)*k)]).astype('complex'))
    diag2.append(scipy.sparse.csr_matrix.todense(J[range((m-1)*k, m*k), :][:, range((m-1)*k, (m)*k)]).astype('complex'))
    M_diag.append(scipy.sparse.csr_matrix.todense(M[range((m-1)*k, m*k), :][:, range((m-1)*k, (m)*k)]).astype('complex'))
    return M_diag, (diag1, diag2, diag3)

def ILUpreconditioner(diag1, diag2, diag3):
    '''
    ILUpreconditioner(diag1, diag2, diag3)
    
    Creates an ILU preconditioner for A. A is block tridiagonal.
    
    Parameters
    ----------
    diag1 : list of matrices
        The lower diagonal of A.
    diag2 : list of matrices
        The main diagonal of A
    diag3 : list of matrices
        The upper diagonal of A

    Returns
    -------
    L : scipy sparse csr matrix
        A lower triangular matrix 
    U : scipy sparse csr matrix
        An upper triangular matrix 
        
        We have A equals approximately LU

    '''
    m = len(diag2)
    
    k = np.shape(diag1[0])[0]
    Si = diag2[0]
    Ss = [Si]
    Ti = np.linalg.solve(Si, diag3[0])
    Ts = [Ti]
    
    prev = []
    
    for i in range(2, m+1):
        current = [diag2[i-1], diag1[i-2], diag2[i-2], diag3[i-3], diag3[i-2]]
        if current == prev:
            Ss.append(Si)
            Ts.append(Ti)
        else:
            M = np.linalg.solve(current[2], current[3])
            H = current[1]@M
            G = np.linalg.solve(current[0], H)
            F = np.eye(k) + 2*G
            Si = current[0]@np.linalg.inv(F)
            Ss.append(Si)
            
            if i != m:
                Ti = np.linalg.solve(Si, current[4])
                Ts.append(Ti)
                prev = current

    L = scipy.sparse.bmat([[Ss[i] if i == j else diag1[j] if i-j==1
                else None for j in range(m)]
                for i in range(m)], format='csr')
    U = scipy.sparse.bmat([[np.eye(k) if i == j else Ts[i] if i-j==-1
            else None for j in range(m)]
            for i in range(m)], format='csr')
    
    return L, U
'''
A = np.array([[2+1.j, 1+3.j, 0, 0, 0], 
              [-10.j, 2, 3, 0, 5],
              [0, 1, 2, 3-4.j, 0],
              [2, 0, 4, 5, 6-1.j], 
              [1, 2, 3, 4, 5]])
b = np.array([1, 5, 2, 3, 2+1.j])
'''
'''
def run():
    num_iters = 0
    def callback(xk):
        nonlocal num_iters
        num_iters += 1
        
    
    A, b = get_matrix_problem(0.05, 100)
    start_point = solve(A, b, 'standard')
    
    
    A, b = get_matrix_problem(0.1, 100)
    
    c_exact = solve(A, b, 'standard')
    c = solve(A, b, 'bicgstab', start_point)
    import scipy.sparse.linalg
    
    c2 = scipy.sparse.linalg.bicgstab(A, b, x0 = start_point, callback = callback)
    
    e = c - c_exact 
    e2 = c2[0] - c_exact
    
    print(c)
    print(c_exact)
    print(e)
    
    print()
    #print(num_iters)
    #print(c2)
    #print(e2)
    
    change = start_point - c_exact
    #print(change)
    #print(str(e-change))
run()
'''

'''
#TDGE TEST
diag1 = np.full(5, 1, dtype = float)
diag2 = np.full(6, 3, dtype = float)
diag3 = diag1
c = tdge(diag1, diag2, diag3, [0, 0, 0, 0, 0, 3])
print(c)
'''

'''
#Thomas solve Test
diag1 = []
diag2 = []
diag3 = []
A1 = np.array([[1, 2], [3, 4]],dtype = 'float64')
B1 = np.array([[2, 4], [2, 9]], dtype = 'float64')
C1 = np.array([[0, 0], [0, 0]],dtype = 'float64')

diag1.append(A1)
diag2.append(B1)
diag3.append(C1)
diag2.append(B1)
b = np.zeros(4)
b[-1] = 2
ans = thomasMethod(diag1, diag2, diag3, b)
print("thomas method answer:" + str(ans))

M = np.array([[2, 4, 0, 0], [2, 9, 0, 0], [1, 2, 2, 4], [3, 4, 2, 9]])
ans = np.linalg.solve(M, b)
print("Numpy solve answer: " + str(ans))
'''
'''
#preconditioner test
A, b = get_matrix_problem(5, 1)
print(np.shape(A))
M_diags, J_diags = get_block_diagonals(csr_matrix(np.zeros([4, 4])), csr_matrix(A), 1)
L, U = ILUpreconditioner(J_diags[0], J_diags[1], J_diags[2])
print(J_diags)
print(L.todense())
print(U.todense())

print(A)
M = L@U
print(M.todense())
print(np.shape(M))
#CG required positive definite symmetric matrices 
#Make matrix symmetric?
#BiCG
#BiCGSTAB?
'''