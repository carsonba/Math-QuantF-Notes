from numpy import array, diag, degrees, exp, ones, tile, sqrt
from numpy.linalg import eig
from scipy.stats import multivariate_normal
from math import atan
from seaborn import scatterplot
from matplotlib.patches import Ellipse
from matplotlib.pyplot import gca






######## inputs (you can change them) ########
j_bar = 1000  # number of scenarios
mu = array([0.1, 0.08])  # location
sigma2 = array([[0.123, -0.063],
                [-0.063, 0.09]])  # dispersion
# parameters of affine transformation
a = array([-0.5, 0.5])
b = array([[-1, -0.1],
           [0.01, 0.8]])
##############################################

x = (exp(multivariate_normal.rvs(mu, sigma2, j_bar)) - 1).T  # shifted lognormal scenarios
y = tile(a.reshape(2, 1), (1, j_bar)) + b@x  # affine transformation of lognormal scenarios







# parameters of shifted lognormal
e_x = exp(mu + 0.5*diag(sigma2)) - 1  # mean
cv_x = diag(exp(mu + 0.5*diag(sigma2)))@(exp(sigma2) - ones((mu.shape[0], mu.shape[0])))@\
       diag(exp(mu + 0.5*diag(sigma2)))  # covariance

# parameters of affine transformation of lognormal

e_y = a + b@e_x  # mean
cv_y = b@cv_x@b.T  # covariance





# parameters for ellipses based on covariance and radius (width, height, angle)
ellipse_params = lambda sigma2, r: (2*r*sqrt(eig(sigma2)[0][0]), 2*r*sqrt(eig(sigma2)[0][1]),
                                    degrees(atan(eig(sigma2)[1][1, 0]/eig(sigma2)[1][1, 1])))

# ellipse of lognormal
width, height, angle = ellipse_params(cv_x, 1)
ellipse_x = Ellipse(e_x, width, height, angle=angle, fill=False)  # ellipse

# ellipse of affine transformation
width, height, angle = ellipse_params(cv_y, 1)
ellipse_y = Ellipse(e_y, width, height, angle=angle, fill=False)  # ellipse

######################## plot #########################
# affine transformation of a bivariate lognormal
scatterplot(x=x[0, :], y=x[1, :], label='X')
scatterplot(x=y[0, :], y=y[1, :], label='Y')
gca().add_patch(ellipse_x); gca().add_patch(ellipse_y);



