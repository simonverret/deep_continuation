#%% 
import numpy as np
from scipy.special import erf, expit  # expit = sigmoid
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).parent
PLOT = HERE/"plots"


def piecelin_unit(x, n, soft=0):
    flx_pts = np.sort(np.random.uniform(0, 1, size=n))
    slopes = np.random.uniform(0, 10e5, size=n+1)
    slopes[0] = 0.001
    slopes[n] = 0.999 
    slp_chg = slopes[1:] - slopes[:-1]
    nomalize = ((flx_pts[1:]-flx_pts[:-1])*slopes[1:-1]).sum()

    x = x[np.newaxis, :]
    slp_chg = slp_chg[:, np.newaxis]
    flx_pts = flx_pts[:, np.newaxis]
    if soft:  # using softplus
        y = (slp_chg * np.logaddexp(0,(x - flx_pts)/soft)*soft).sum(axis=0)
    else:  # using relu
        y = (slp_chg * (x - flx_pts) * (x > flx_pts)).sum(axis=0)
    
    return y/nomalize


def piecewise_lin(x, xlims=[0,1], ylims=[0,1], **kwargs):
    (l,r),(b,t) = xlims,ylims
    return b+(t-b)*piecelin_unit((x-l)/(r-l), **kwargs)


def piecetan_unit(x, n, soft=0):
    jmps = np.random.uniform(0, 1, size=n+1)
    amps = np.random.uniform(0, 1, size=n+1)
    amps /= amps.sum()

    x = x[np.newaxis, :]
    jmps = jmps[:, np.newaxis]
    amps = amps[:, np.newaxis]
    if soft:  # using sigmoid
        y = (amps * expit((x-jmps)/(soft))).sum(axis=0)
    else:  # using heavyside
        y = (amps * (x > jmps)).sum(axis=0)
    return y


def piecewise_tan(x, xlims=[0,1], ylims=[0,1], **kwargs):
    (l,r),(b,t) = xlims,ylims
    return b+(t-b)*piecetan_unit((x-l)/(r-l), **kwargs)


def test_plot_piecewise():
    x = np.linspace(-3,3,1000)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,3))
    seed = np.random.randint(1000)
    for soft in np.linspace(0,0.03,3):
        np.random.seed(seed)
        ax1.plot(x, piecewise_lin(x, n=4, soft=soft, xlims=[-2,2], ylims=[-1,1]))
        ax2.plot(x, piecewise_tan(x, n=4, soft=soft, xlims=[-2,2], ylims=[-1,1]))
        ax2.set_xlim(-2,2)
        ax2.set_ylim(-1,1)
    ax1.set_title("relu and softplus")
    ax2.set_title("heavysides and sigmoid")
    plt.savefig(PLOT/"monofunc__my_piecewise.pdf")
    plt.clf()

# test_plot_piecewise()


#%%

def piecewise_gap_unit(x, n, soft=0):
    N = n
    n = np.random.randint(1,n)
    N = N-n
    flx = np.sort(np.random.uniform(0, 1, size=n+1))
    flx[0] = 0
    flx[n] = 1
    yflex = np.sort(np.random.uniform(0, 1, size=n+1))
    yflex[0] = 0
    yflex[n] = 1    
    slopes = (yflex[1:]-yflex[:n])/(flx[1:]-flx[:n])
    chg = slopes[1:] - slopes[:-1]
    y = slopes[0]*x
    
    x = x[np.newaxis, :]
    chg = chg[ : , np.newaxis]
    flx = flx[1:n, np.newaxis]
    y += (chg * np.logaddexp(0,(x - flx)/soft)*soft).sum(axis=0)

    n = np.random.randint(0,N)
    N = N-n
    jmps = np.random.uniform(0, 1, size=n+1)
    amps = np.random.uniform(0, 1, size=n+1)
    jmps = jmps[:, np.newaxis]
    amps = amps[:, np.newaxis]
    y += (amps * expit((x-jmps)/(soft))).sum(axis=0)

    n = np.random.randint(0,N)
    N = N-n
    jmps = np.random.uniform(0, 1, size=n+1)
    amps = np.random.uniform(0, 1, size=n+1)
    jmps = jmps[:, np.newaxis]
    amps = amps[:, np.newaxis]
    y += (amps * (x > jmps)).sum(axis=0)
    return y/y.max()
    
def piecewise_gap(x, xlims=[0,1], ylims=[0,1], **kwargs):
    (l,r),(b,t) = xlims,ylims
    return b+(t-b)*piecewise_gap_unit((x-l)/(r-l), **kwargs)

def test_plot_piecewise2():
    x = np.linspace(-2,2,1000)
    seed = np.random.randint(1000)
    plt.plot(x, piecewise_gap(x, n=18, soft=0.01, xlims=[-2,2], ylims=[-1,1]))
    plt.show()

# test_plot_piecewise2()


#%%


def piecelin(v, N_seg): # A randomizable monotonically increasing piecewise linear function
    v_max = v[-1]
    angles = np.random.uniform(0, np.pi/2, size=N_seg) # Generates a list of random angles of length N_seg in the interval [0,pi/2).
    sines = np.sin(angles)
    cosines = np.cos(angles) # Two lists, one of the sines of the angles, one of the cosines.
    x_list = np.cumsum(cosines) # The x-coordinates of the connection points between line segments. Each is the previous x-value plus the cosine of the current angle.
    x_list *= 1.1*v_max/x_list[-1] # Resize the x-coordinates so all the peaks fit inside
    y_list = np.cumsum(sines) # The y-coordinates of the connection points between line segments
    x_list = np.insert(x_list,0,0)
    y_list = np.insert(y_list,0,0) # We add (0,0) to the beginning of the list of endpoints
    x_lower = np.zeros(len(v))
    y_lower = np.zeros(len(v))
    x_upper = np.zeros(len(v))
    y_upper = np.zeros(len(v)) # Initialized vectors that will store the endpoints of the line segment each input point is on
    for i in range(len(v)):
        # For each value in the vector v, this for-loop will find the x and y coordinates of the endpoints to either side of it.
        # In other words, these lists determine which line segment each point is located on.
        x_lower[i] = np.max(x_list[x_list <= v[i]])
        y_lower[i] = np.max(y_list[x_list <= v[i]])
        x_upper[i] = np.min(x_list[x_list > v[i]])
        y_upper[i] = np.min(y_list[x_list > v[i]])
    x_diffs = x_upper - x_lower
    y_diffs = y_upper - y_lower
    slopes = y_diffs/x_diffs # These three lines find the slope of the line segment each point exists on
    return slopes*v + y_lower - slopes*x_lower # This outputs the heights of the piecewise function at each point. 
                                                # This is derived from expressing the line segment in point-slope form.


def softp(x, N_seg): # Each term has the form A*log(1+exp(x-c)), plus a linear term 
    x_max = x[-1]
    c = np.random.uniform(0, 1, N_seg) # This will store all the values of c
    c = np.sort(c) # Put the values in order
    c *= np.random.uniform(0.8, 1.2)*x_max/c[-1] # Rescale the list of c-values so too much of the tail doesn't get included
    A = np.zeros(N_seg + 1) # This will store all the values of A, including the linear coefficient (hence the extra element)
    A[0] = 10 * np.random.rand() # This is the linear coefficient
    for i in range(N_seg):
        A_sum = np.sum(A) # Add up all the coefficients so far
        A[i+1]= np.random.uniform(-A_sum, 10) # We want each subsequent coefficient to be greater than the negative sum of all the
                                                # previous coefficients, or the function will not be monotonically increasing
    mat = np.tile(x,(N_seg,1))
    mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
    expmat = np.exp(mat)
    logmat = np.log(1+expmat)
    unsized = np.vstack((x,logmat)) # Add one more copy of x to the matrix to represent the linear term
    unsummed = unsized * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by a coefficient before summing
    out = unsummed.sum(axis=0)
    return (out-out.min())/(out.max()-out.min())


def arctsum(x, N_seg): # Each term has the form A*arctan(B*(x+c))
    x_max = x[-1]
    c = np.random.uniform(-100, 0, size=N_seg)
    B = np.random.uniform(0, 10, size=N_seg)
    A = np.random.uniform(0, 30, size=N_seg)
    c = np.sort(c) # Put the random values of c in order
    c *= np.random.uniform(0.8, 1.2)*x_max/c[-1] # Rescale so too much of the tail doesn't get included
    mat = np.tile(x,(N_seg,1))
    mat = mat + np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
    mat = mat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
    arcmat = np.arctan(mat)
    unsummed = arcmat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
    out = unsummed.sum(axis=0)
    return (out-out.min())/(out.max()-out.min())


def erfsum(x, N_seg): # Each term has the form A*erf(B*(x-c)), plus a linear term
    x_max = x[-1]
    c = np.random.uniform(-10, 0, size=N_seg)
    B = np.random.uniform(0, 100, size=N_seg)
    A = np.random.uniform(0, 10, size=N_seg+1)
    c = np.sort(c) # Put the random values of c in order
    c *= np.random.uniform(0.8, 1.2)*x_max/c[-1] # Rescale so too much of the tail doesn't get included
    mat = np.tile(x,(N_seg,1))
    mat = mat + np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
    mat = mat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
    erfmat = erf(mat)
    unsized = np.vstack((x,erfmat)) # Add one more copy of x to the matrix to represent the linear term
    unsummed = unsized * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
    out = unsummed.sum(axis=0)
    return (out-out.min())/(out.max()-out.min())


def arssum(x, N_seg): # Each term has the form A*arsinh(B*(x+c))
    x_max = x[-1]
    c = np.random.uniform(-10, 0, size=N_seg)
    B = np.random.uniform(0, 10, size=N_seg)
    A = np.random.uniform(0, 10, size=N_seg)
    c = np.sort(c) # Put the random values of c in order
    c *= np.random.uniform(0.8, 1.2)*x_max/c[-1] # Rescale so too much of the tail doesn't get included
    mat = np.tile(x,(N_seg,1))
    mat = mat + np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
    mat = mat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
    arsmat = np.arcsinh(mat)
    unsummed = arsmat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
    out = unsummed.sum(axis=0)
    return (out-out.min())/(out.max()-out.min())


def rootsum(x, N_seg): # Each term has the form A*sign(x+c)*(|x+c|)^(1/n)
    x_max = x[-1]
    c = np.random.uniform(-10, 0, size=N_seg)
    A = np.random.uniform(0, 10, size=N_seg)
    n = np.random.choice([2, 3, 4, 5], N_seg, p=[0.5, 0.25, 0.2, 0.05])
    n = 1/n
    # Generate the order of the root for each term. We will use second, third, fourth, and fifth roots with decreasing probability.
    c = np.sort(c) # Put the random values of c in order
    c *= np.random.uniform(0.8, 1.2)*x_max/c[-1] # Rescale so too much of the tail doesn't get included
    mat = np.tile(x,(N_seg,1))
    mat = mat + np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
    signmat = np.sign(mat) # Store the signs of the elements
    powermat = np.power(np.abs(mat), np.transpose(np.tile(n,(len(x),1)))) # Take the relevant root for the absolute value of each element, row by row
    powermat = signmat * powermat # Restore the original signs after taking the root, so all negative inputs yield negative outputs
    unsummed = powermat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
    out = unsummed.sum(axis=0)
    return (out-out.min())/(out.max()-out.min())


def exparsinh(x, N_seg): # Each term has the form A*exp(B*arsinh(x+c))
    x_max = x[-1]
    c = np.random.uniform(0, 100, size=N_seg)
    B = np.random.uniform(0, 0.9999, size=N_seg) # As long as B < 1, the function's slope will decrease for large x
    A = np.random.uniform(0, 10, size=N_seg)
    c = np.sort(c) # Put the random values of c in order
    c *= np.random.uniform(0.8, 1.2)*x_max/c[-1] # Rescale so too much of the tail doesn't get included
    mat = np.tile(x,(N_seg,1))
    mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
    arsmat = np.arcsinh(mat)
    arsmat = arsmat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
    expmat = np.exp(arsmat)
    unsummed = expmat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
    out = unsummed.sum(axis=0)
    return (out-out.min())/(out.max()-out.min())


def exparctan(x, N_seg): # Each term has the form A*exp(arctan(B(x+c)))
    x_max = x[-1]
    c = np.random.uniform(0, 100, size=N_seg)
    B = np.random.uniform(0, 2, size=N_seg) # As long as B < 1, the function's slope will decrease for large x
    A = np.random.uniform(0, 10, size=N_seg)
    c = np.sort(c) # Put the random values of c in order
    c *= np.random.uniform(0.8, 1.2)*x_max/c[-1] # Rescale so too much of the tail doesn't get included
    mat = np.tile(x,(N_seg,1))
    mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
    mat = mat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
    arcmat = np.arctan(mat)
    expmat = np.exp(arcmat)
    unsummed = expmat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
    out = unsummed.sum(axis=0)
    return (out-out.min())/(out.max()-out.min())


def arssoft(x, N_seg): # Each term has the form A*arsinh(ln(1+exp(x+c)))
    x_max = x[-1]
    c = np.random.uniform(0, 100, size=N_seg)
    A = np.random.uniform(0, 10, size=N_seg)
    c = np.sort(c) # Put the random values of c in order
    c *= np.random.uniform(0.8, 1.2)*x_max/c[-1] # Rescale so too much of the tail doesn't get included
    mat = np.tile(x,(N_seg,1))
    mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
    expmat = np.exp(mat)
    logmat = np.log(1 + expmat)
    arsmat = np.arcsinh(logmat)
    unsummed = arsmat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
    out = unsummed.sum(axis=0)
    return (out-out.min())/(out.max()-out.min())


def tanerf(x, N_seg): # Each term has the form A*tan(B*erf(C*(x+c)))
    x_max = x[-1]
    c = np.random.uniform(0, 10, size=N_seg)
    A = np.random.uniform(0, 10, size=N_seg)
    B = np.random.uniform(0, np.pi/2, size=N_seg) # If B were greater than pi/2, multiple cycles of tan would activate and the function would be discontinuous
    C = np.random.uniform(0, 2, size=N_seg)
    c = np.sort(c) # Put the random values of c in order
    c *= np.random.uniform(0.8, 1.2)*x_max/c[-1] # Rescale so too much of the tail doesn't get included
    mat = np.tile(x,(N_seg,1))
    mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
    mat = mat * np.transpose(np.tile(C,(len(x),1))) # Multiply all the arguments by their coefficients
    erfmat = erf(mat)
    erfmat = erfmat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
    tanmat = np.tan(erfmat)
    unsummed = tanmat * np.transpose(np.tile(A,(len(x),1))) # Multiply all the arguments by their coefficients
    out = unsummed.sum(axis=0)
    return (out-out.min())/(out.max()-out.min())


def logarc(x, N_seg): # Each term has the form A*log(np.pi/2 + a + arctan(x+c))
    x_max = x[-1]
    c = np.random.uniform(0, 10, size=N_seg)
    A = np.random.uniform(0, 10, size=N_seg)
    a = np.random.uniform(0, 10, size=N_seg) # If you add something less than pi/2 before taking the log, you will get a vertical asymptote, which we do not want
    c = np.sort(c) # Put the random values of c in order
    c *= np.random.uniform(0.8, 1.2)*x_max/c[-1] # Rescale so too much of the tail doesn't get included
    mat = np.tile(x,(N_seg,1))
    mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
    arcmat = np.arctan(mat)
    arcmat = arcmat + np.pi/2 + np.transpose(np.tile(c,(len(x),1)))
    logmat = np.log(arcmat)
    unsummed = logmat * np.transpose(np.tile(A,(len(x),1))) # Multiply all the arguments by their coefficients
    out = unsummed.sum(axis=0)
    return (out-out.min())/(out.max()-out.min())


def debug(x, N_seg): # A simple, non-randomizable function that is used to test whether the code is working. Should not be called normally
    x_max = x[-1]
    return x**2
# More center distribution functions to be added.



def plot_base_functions(c=0, A=1, B=1, C=1, r=2, a=0):
    x = np.linspace(-3,3,200)
    # kink-type
    plt.plot(x, A*np.log(1+np.exp(x-c)))  #softp
    plt.plot(x, A*np.exp(B*np.arcsinh(x+c)))  #exparsinh
    ## s-type
    plt.plot(x, A*np.arctan(B*(x-c)))  #arct
    plt.plot(x, A*erf(B*(x-c)))  #erf
    plt.plot(x, A*np.tan(B*erf(C*(x+c))))  #tanerf
    ## slope-s-type
    plt.plot(x, A*np.sign(x)*np.power(np.abs(x),1/r))  #root
    plt.plot(x, A*np.arcsinh(B*(x+c)))  #arsinh
    ## kink-s-type
    plt.plot(x, A*np.exp(np.arctan(B*(x+c))))  #expartan
    plt.plot(x, A*np.arcsinh(np.log(1+np.exp(x+c))))  #arssoft
    plt.plot(x, A*np.log(np.pi/2 + a + np.arctan(x+c)))  #logarc  (looks like a flipped arssoft)
    plt.savefig(PLOT/"monofunc__base_funcs.pdf")
    plt.clf()



def main():
    k = np.linspace(0,20,1000)
    n = 100
    for i in range(3):
        plt.plot(k, piecelin(k,n))
    plt.title("piecelin")
    plt.savefig(PLOT/"monofunc_piecelin.pdf")
    plt.clf()
    for i in range(3):
        plt.plot(k, softp(k,n))
    plt.title("softp")
    plt.savefig(PLOT/"monofunc_softp.pdf")
    plt.clf()
    for i in range(3):
        plt.plot(k, arctsum(k,n))
    plt.title("arctsum")
    plt.savefig(PLOT/"monofunc_arctsum.pdf")
    plt.clf()
    for i in range(3):
        plt.plot(k, erfsum(k,n))
    plt.title("erfsum")
    plt.savefig(PLOT/"monofunc_erfsum.pdf")
    plt.clf()
    for i in range(3):
        plt.plot(k, arssum(k,n))
    plt.title("arssum")
    plt.savefig(PLOT/"monofunc_arssum.pdf")
    plt.clf()
    for i in range(3):
        plt.plot(k, rootsum(k,n))
    plt.title("rootsum")
    plt.savefig(PLOT/"monofunc_rootsum.pdf")
    plt.clf()
    for i in range(3):
        plt.plot(k, exparsinh(k,n))
    plt.title("exparsinh")
    plt.savefig(PLOT/"monofunc_exparsinh.pdf")
    plt.clf()
    for i in range(3):
        plt.plot(k, exparctan(k,n))
    plt.title("exparctan")
    plt.savefig(PLOT/"monofunc_exparctan.pdf")
    plt.clf()
    for i in range(3):
        plt.plot(k, arssoft(k,n))
    plt.title("arssoft")
    plt.savefig(PLOT/"monofunc_arssoft.pdf")
    plt.clf()
    for i in range(3):
        plt.plot(k, tanerf(k,n))
    plt.title("tanerf")
    plt.savefig(PLOT/"monofunc_tanerf.pdf")
    plt.clf()
    for i in range(3):
        plt.plot(k, logarc(k,n))
    plt.title("logarc")
    plt.savefig(PLOT/"monofunc_logarc.pdf")
    plt.clf()


def test_plot_piecewise():
    x = np.linspace(-3,3,1000)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,3))
    for soft in np.linspace(0,0.1,5):
        np.random.seed(111)
        ax1.plot(x, piecewise_lin(x, n=4, soft=soft, xlims=[-2,2], ylims=[-1,1]))
        ax2.plot(x, piecewise_tan(x, n=4, soft=soft, xlims=[-2,2], ylims=[-1,1]))
    ax1.set_title("relu and softplus")
    ax2.set_title("heavysides and sigmoid")
    plt.savefig(PLOT/"monofunc__my_piecewise.pdf")
    plt.clf()



if __name__ == "__main__":
    plot_base_functions()
    main()
    test_plot_piecewise()
