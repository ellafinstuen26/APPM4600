import numpy as np

def driver():
    h = 0.01*2.0**(-np.arange(1, 10))
    f = lambda x: np.cos(x)
    x = np.pi/2
    correct = -1

    approx_forward = forward(f,x,h)
    approx_center = center(f,x,h)


    print("Approximated Derivative with forward difference with varying h values: ", approx_forward)
    convergence_order(approx_forward,correct)
    print("Approximated Derivative with centered difference with varying h values: ", approx_center)
    convergence_order(approx_center,correct)


    

def forward(f,x,h):
    return (f(x+h)-f(x))/h


def center(f,x,h):
    return (f(x+h)-f(x-h))/(2*h)

def convergence_order(x,xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)
    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
    print('the order equation is')
    print('log(|p_{n+1}-p|) = log(lambda) + alpha*log(|p_n-p|) where')
    print('lambda = ', str(np.exp(fit[1])))
    print('alpha = ', str(fit[0]))

    return [fit,diff1,diff2]


driver()

