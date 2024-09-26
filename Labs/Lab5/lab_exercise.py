# import libraries
import numpy as np

def driver():

# use routines    
    # f = lambda x: x**3+x-4
    # fp = lambda x: 3*x**3+1
    # fpp = lambda x: 9*x**2
    # a = 1
    # b = 4

    f = lambda x: np.exp(x**2+7*x-30)-1
    fp = lambda x: (2*x+7)*np.exp(x**2+7*x-30)
    fpp = lambda x: (2*x+7)**2*np.exp(x**2+7*x-30) + 2*np.exp(x**2+7*x-30)
    a=2
    b=4.5
    p0 = 4.5




#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-8

    [astar,ier,it1] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('Number of iterations:', '%d' % it1)

    (p,pstar,info,it2) = newton(f,fp,p0,tol, 100)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it2)

    [astar,ier,ita] = hybrid(f,fp,fpp,a,b,tol)
    if ier == 2:
       [a,astar,info,itb] = newton(f,fp,p0,tol,100)
       ier = 0
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('Number of iterations:', '%d' % ita, "+", itb, '=', ita+itb)


def bisection(f,a,b,tol):


    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier,0]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier,0]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier,0]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier,count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier,count]

# define routines
def hybrid(f,fp,fpp,a,b,tol):
    g= lambda x: x-f(x)/fp(x)
    gp = lambda x: f(x)*fpp(x)/fp(x)**2

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier,0]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier,0]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier,0]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier,count]
  
      gm = gp(d)
      if abs(gm)<1:
        ier = 2
        return [gm,ier,count+1]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]

def newton(f,fp,p0,tol,Nmax):

  p = np.zeros(Nmax+1);
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]
      
driver()               

