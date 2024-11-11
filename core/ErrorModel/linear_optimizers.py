import cvxopt

#=======================================================================================================================
def l1blas (P, q):
    """
    Returns the solution u of the ell-1 approximation problem

        (primal) minimize ||P*u - q||_1

        (dual)   maximize    q'*w
                 subject to  P'*w = 0
                             ||w||_infty <= 1.

    This is a modification of the l1 solver from http://cvxopt.org/examples/mlbook/l1.html
    """

    m, n = P.size

    # Solve equivalent LP
    #
    #     minimize    [0; 1]' * [u; v]
    #     subject to  [P, -I; -P, -I] * [u; v] <= [q; -q]
    #
    #     maximize    -[q; -q]' * z
    #     subject to  [P', -P']*z  = 0
    #                 [-I, -I]*z + 1 = 0
    #                 z >= 0

    c = cvxopt.matrix(n*[0.0] + m*[1.0])
    h = cvxopt.matrix([q, -q])
    u = cvxopt.matrix(0.0, (m,1))
    Ps = cvxopt.matrix(0.0, (m,n))
    A = cvxopt.matrix(0.0, (n,n))

    def Fi(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):
        if trans == 'N':
            cvxopt.blas.gemv(P, x, u)
            y[:m] = alpha * ( u - x[n:]) + beta*y[:m]
            y[m:] = alpha * (-u - x[n:]) + beta*y[m:]
        else:
            cvxopt.blas.copy(x[:m] - x[m:], u)
            cvxopt.blas.gemv(P, u, y, alpha = alpha, beta = beta, trans = 'T')
            y[n:] = -alpha * (x[:m] + x[m:]) + beta*y[n:]

    def Fkkt(W):

        # Returns a function f(x, y, z) that solves
        #
        # [ 0  0  P'      -P'      ] [ x[:n] ]   [ bx[:n] ]
        # [ 0  0 -I       -I       ] [ x[n:] ]   [ bx[n:] ]
        # [ P -I -D1^{-1}  0       ] [ z[:m] ] = [ bz[:m] ]
        # [-P -I  0       -D2^{-1} ] [ z[m:] ]   [ bz[m:] ]
        #
        # where D1 = diag(di[:m])^2, D2 = diag(di[m:])^2 and di = W['di'].
        #
        # On entry bx, bz are stored in x, z.
        # On exit x, z contain the solution, with z scaled (di .* z is
        # returned instead of z).

        # Factor A = 4*P'*D*P where D = d1.*d2 ./(d1+d2) and
        # d1 = d[:m].^2, d2 = d[m:].^2.

        di = W['di']
        d1, d2 = di[:m]**2, di[m:]**2
        D = cvxopt.div( cvxopt.mul(d1,d2), d1+d2 )
        Ds = cvxopt.spdiag(2 * cvxopt.sqrt(D))
        cvxopt.base.gemm(Ds, P, Ps)
        cvxopt.blas.syrk(Ps, A, trans = 'T')
        cvxopt.lapack.potrf(A)

        def f(x, y, z):

            # Solve for x[:n]:
            #
            #    A*x[:n] = bx[:n] + P' * ( ((D1-D2)*(D1+D2)^{-1})*bx[n:]
            #        + (2*D1*D2*(D1+D2)^{-1}) * (bz[:m] - bz[m:]) ).

            cvxopt.blas.copy(( cvxopt.mul( cvxopt.div(d1-d2, d1+d2), x[n:]) +
                cvxopt.mul( 2*D, z[:m]-z[m:] ) ), u)
            cvxopt.blas.gemv(P, u, x, beta = 1.0, trans = 'T')
            cvxopt.lapack.potrs(A, x)
            cvxopt.base.gemv(P, x, u)
            x[n:] =  cvxopt.div( x[n:] - cvxopt.mul(d1, z[:m]) - cvxopt.mul(d2, z[m:]) + cvxopt.mul(d1-d2, u), d1+d2 )
            z[:m] = cvxopt.mul(di[:m],  u-x[n:]-z[:m])
            z[m:] = cvxopt.mul(di[m:], -u-x[n:]-z[m:])
        return f


    # Don't explicitly provide initial primal and dual points.
    dims = {'l': 2*m, 'q': [], 's': []}
    sol = cvxopt.solvers.conelp(c, Fi, h, dims, kktsolver = Fkkt)
    return sol['x'][:n]
