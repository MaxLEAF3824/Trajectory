def discret_lcss(t0, t1):
    """
    适用于离散点的lcss算法，不需要eps
    param t0 : len(t0) numpy_array
    param t1 : len(t1)  numpy_array
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if t0[i - 1] == t1[j - 1]:
                C[i][j] = C[i - 1][j - 1] + 1
            else:
                C[i][j] = max(C[i][j - 1], C[i - 1][j])
    lcss = 1 - float(C[n0][n1]) / min([n0, n1])
    return lcss


