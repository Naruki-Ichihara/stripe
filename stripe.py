import numpy as np
from dolfin import *
from dolfin_adjoint import *
from mpi4py import MPI
import fenics_optimize as op
from fecr import from_numpy
comm = MPI.COMM_WORLD

eps = 0.25

recorder = op.Recorder('./results', 'field')

class VectorField(UserExpression):
    def eval(self, val, x):
        val[0] = 1/np.sqrt(2)
        val[1] = 1/np.sqrt(2)
    def value_shape(self):
        return (2,)

mesh = UnitSquareMesh(50, 50)
X = FunctionSpace(mesh, 'CG', 1)
x = Function(X)
V = VectorFunctionSpace(mesh, 'CG', 1)
v = Function(V)
vectorfield = VectorField()
v.interpolate(vectorfield)

@op.with_derivative([X])
def forward(xs):
    vec = grad(xs[0])-v
    e = sqrt(vec[0]**2 + vec[1]**2)**2*dx
    cost = assemble(e)
    return cost

problemSize = op.catch_problemSize([X])
x0 = np.zeros(problemSize)
x_min = -np.ones(problemSize)*100
x_max = np.ones(problemSize)*100

solved_numpy = op.HSLoptimize(problemSize=problemSize, initial=x0, forward=forward, bounds=[x_min, x_max], maxeval=1000, solver_type='ma97')
solved = from_numpy(solved_numpy, x)
p = 2*np.pi/eps
wave = project(cos(p*solved), X)

file = File('results/field.pvd')
file << wave