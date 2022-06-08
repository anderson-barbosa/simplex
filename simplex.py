# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv

def _simplex_inverse(bbinv,new_col,pos):
    '''
    Computes the inverse of matrix given one of its columns `new_col`, the
    position of this column `pos` and the inverse of another matrix that
    only differs from it in this column. 

    Parameters
    ----------
    bbinv : numpy.ndarray
        Inverse of a matrix that only differs from the one to be inverted in
        one column.
    new_col : numpy.ndarray
        The different column.
    pos : numpy.ndarray
        Position of the different column.

    Returns
    -------
    numpy.ndarray
        The inverse of the matrix.

    '''
    e=np.identity(np.shape(bbinv)[0])
    v=np.matmul(bbinv,new_col)
    vpos=v[pos,0]
    
    for i in range(np.shape(e)[0]):
        if i!=pos:
            e[i,pos]=-v[i,0]/vpos
        else:
            e[i,pos]=1/vpos

    return np.matmul(e,bbinv)

def _simplex_iteration(bbinv,cbt,cnt,n,b):
    '''
    Performs one simplex iteration.
    
    Given the inverse of a feasible basis, determines which variable should 
    enter the basis and should leave it.

    Parameters
    ----------
    bbinv : numpy.ndarray
        Inverse of the basis matrix to be used.
    cbt : numpy.ndarray
        Cost vector for the basic variables.
    cnt : numpy.ndarray
        Cost vector for the non-basic variables..
    n : numpy.ndarray
        Coefficients fo the non-basic variables.
    b : numpy.ndarray
        Vector `b` of the problem.

    Returns
    -------
    int
        Identifier for the result of the iteration. It is 0 if the given basis
        is optimal, 1 if the problem is unbounded and 2 otherwise.
    int
        Index of the variable that should leave the basis. It is`None` if
        the given basis is optimal or the problem is unbounded.
    int
        Index of the variable that should enter the basis. It is`None` if
        the given basis is optimal or the problem is unbounded.
    numpy.ndarray
        Values of the basic variables.

    '''
    
    # Computes the matrices of the simplex standard form with respect to the
    # basis given
    y=np.matmul(bbinv,n)
    xb=np.matmul(bbinv,b)
    cz=cnt-np.matmul(cbt,y)
    
    # Determines the index of the variable that should enter the basis
    newb=0
    for i in range(np.shape(cz)[1]):
        if cz[0,i]>0:
            newb=i
            break
    else:
        # If cz<=0, the basis is optimal
        return 0,None,None,xb
    
    # Determines the index of the variable that shoud leave the basis
    l1=[]
    for i in range(np.shape(y)[0]):
        if y[i,newb]>0:
            l1.append(i)
    if len(l1)==0:
        # If y[i,newb]<=0 for all i outside the basis, the problem in
        # unbounded, then returns 1,None,None,None
        return 1,None,None,None
    newn=l1[0]
    min_ratio=xb[newn,0]/y[newn,newb]
    for i in l1[1:]:
        if xb[i,0]/y[i,newb]<min_ratio:
            newn=i
            min_ratio=xb[i,0]/y[i,newb]
            
    # Returns the changes to be made to the basis
    return 2,newn,newb,None

def simplex(a,b,ct,ib):
    '''
    Performs the simplex algorithm for the problem max ct*x, Ax=b, x>=0 from 
    the feasible basis defined by `ib`.

    Parameters
    ----------
    a : numpy.ndarray
        Matrix of coefficients.
    b : numpy.ndarray
        Column vector `b`.
    ct : numpy.ndarray
        Cost vector.
    ib : list
        Indexes of the basic variables.

    Returns
    -------
    float
        Optimal value of the objective function. It is float('inf') if the
        problem is unbounded.
    numpy.ndarray
        Optimal solution `x`. It is `None` if the problem in unbounded.
    list
        Indexes of the variables in the optimal basis found.  It is `None` if 
        the problem in unbounded.

    '''
    
    # Get the number of constraints and variables
    m,nv=np.shape(a)
    
    i_n=[i for i in range(nv) if i not in ib]
    i_b=ib
    
    # Computes the matrices of the standard form  with respect to the given
    # basis
    bb=a[:,ib]
    n=a[:,i_n]
    cbt=ct[:,ib]
    cnt=ct[:,i_n]
    bbinv=inv(bb)
    
    # Run until simplex converges
    while True:
        
        # Performs one simplex iteration from the current basis
        id_,newn,newb,xb=_simplex_iteration(bbinv,cbt,cnt,n,b)
        
        if id_==0:
            
            # Returns the optimal value of the objective function, the optimal
            # solution and the indexes of the basic variables
            zbar=np.matmul(cbt,xb)[0,0]
            x=np.zeros((nv,1))
            for i in range(m):
                x[i_b[i],0]=xb[i,0]
            return zbar,x,i_b
        
        elif id_==1:
            
            # Since the problem is unbounded, returns float('inf'),None,None
            return float('inf'),None,None
        
        elif id_==2:
            
            # Updates the basis using the result of the iteration
            temp=i_b[newn]
            i_b[newn]=i_n[newb]
            i_n[newb]=temp
            n=a[:,i_n]
            cbt=ct[:,ib]
            cnt=ct[:,i_n]
            bbinv=_simplex_inverse(bbinv,a[:,[i_b[newn]]],newn)
            
def two_phase_simplex(a,b,ct):
    '''
    Performs the two-phase simplex algorithm for the problem max ct*x, Ax=b, 
    x>=0.

    Parameters
    ----------
    a : numpy.ndarray
        Matrix of coefficients.
    b : numpy.ndarray
        Column vector `b`.
    ct : numpy.ndarray
        Cost vector.

    Returns
    -------
    float
        Optimal value of the objective function. It is float('inf') if the
        problem is unbounded and `None` if it is unfeasible.
    numpy.ndarray
        Optimal solution `x`. It is `None` if the problem in unbounded or 
        unfeasible.
    list
        Indexes of the variables in the optimal basis found. It is `None` if 
        the problem in unbounded or unfeasible.

    '''
    
    # Get the number of constraints and variables
    m,n=np.shape(a)
    
    # Make a copy of matrix a
    new_a=a.copy()
    new_b=b.copy()
    
    # Make b>=0
    for i in range(m):
        if b[i,0]<0:
            new_a[i,:]*=-1
            new_b[i,0]*=-1
    
    # Creates an artificial problem (PA) to get a basic feasible solution
    new_a=np.concatenate((a,np.identity(m)),axis=1)
    new_ct=np.concatenate((np.zeros((1,n)),-np.ones((1,m))),axis=1)
    
    # Apply simplex to the PA
    zbar1,x1,i_b1=simplex(new_a,b,new_ct,[n+i for i in range(m)])
    
    # If the optimal value of the PA is less than 0, que problem is unfeasible
    #Then returns None,None,None
    if zbar1<0:
        return None,None,None
    
    # Replace artificial variables in the basis found
    zbvars=[i for i in range(m) if i_b1[i]>=n]
    if len(zbvars)>0:
        nvar=0
        for i in zbvars:
            while nvar in i_b1:
                nvar+=1
            i_b1[i]=nvar
    
    # Runs simplex from the basis found
    return simplex(a,b,ct,i_b1)
    
    
    


    
    
    
    