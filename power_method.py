import numpy as np


def __find_p(x):
    return np.argwhere(np.isclose(np.abs(x), np.linalg.norm(x, np.inf))).min()


def __iterate(A, x, p):
    
    y = np.dot(A, x)       
    μ = y[p]      
    p = __find_p(y)     
    error = np.linalg.norm(x - y / y[p],  np.inf)
    x = y / y[p]
    
    return (error, p, μ, x) 


def power_method(A, inverse_a=False, tolerance=1e-10, max_iterations=10, ):
    
    if inverse_a:
        A = np.linalg.inv(A)

    n = A.shape[0]
    x = np.ones(n)
    
    p = __find_p(x)
    
    error = 1
    
    x = x / x[p]
    
    for i in range(max_iterations):
        
        if error < tolerance:
            break
            
        error, p, μ, x = __iterate(A, x, p)
        print("iteration", i)
        print("e-val (μ%s): %s" %(str(i), str(μ)))
        print("e-vec (x%s): %s" %(str(i), str(x)))
        print()
        
    if inverse_a:
        return (1 / μ, x)
    return (μ, x)


if __name__ == '__main__':
    A = np.array([[1, 2, 3, 2],
                    [2, 12, 13, 11],
                    [-2, 4, 0, 2],
                    [4, 5, 7, 2]])
    print(power_method(A, True))
