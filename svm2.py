from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math


##x and y are 2 dim vectors
def kernel(x, y):
    return numpy.dot(x, y) + 1

def kernel_poly(x, y):
    dotPart = numpy.dot(x, y) + 1
    return numpy.power(dotPart, 3)


def buildMatrix(A, B):
    matrix = numpy.empty([20, 20])
    for i in range(len(A)):
        for j in range(len(B)):
            m = A[i][2] * B[j][2] * kernel_poly(A[i][:2], B[j][:2])
            matrix[i][j] = m
    return matrix


def buildExtras():
    q = numpy.empty([20])
    for ndex1 in range(20):
        q[ndex1] = -1

    h = numpy.zeros([20])

    G = numpy.zeros((20, 20))
    numpy.fill_diagonal(G, -1)
    return q, h, G


def indicator(x, y):
    sumi = 0
    for (k, v) in solutions.items():
        sumi = sumi + (k * v[2] * kernel_poly(v[:2], (x, y)))
    return sumi


## ------------------ Main

numpy.random.seed(1)

classA = [(
              random.normalvariate(-2.5, 1),
              random.normalvariate(0.5, 1),
              1.0) for i in range(5)] + \
         [(
              random.normalvariate(-1.5, 1),
              random.normalvariate(0.5, 1),
              1.0) for i in range(5)]

classB = [(random.normalvariate(0.0, 0.5),
           random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]

data = classA + classB

random.shuffle(data)

_q, _h, _G = buildExtras()

built_matrix = buildMatrix(data, data)
numpy.set_printoptions(precision=3)
r = qp(matrix(built_matrix), matrix(_q), matrix(_G), matrix(_h))

solutions = {}
r_x = list(r['x'])
for i, ai in enumerate(r_x):
    if abs(ai) > 1.0e-05:
        solutions[ai] = data[i]
print(solutions)

pylab.hold(True)

xrange = numpy.arange(-4, 4, 0.3)
yrange = numpy.arange(-4, 4, 0.3)

grid = matrix([[indicator(x, y) for y in yrange] for x in xrange])

pylab.plot([p[0] for p in classA],
           [p[1] for p in classA],
           'bo')
pylab.plot([p[0] for p in classB],
           [p[1] for p in classB],
           'ro')

pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors=('blue', 'black', 'red'), linewidths=(1, 3, 1))
pylab.show()
