# Stokes Flow
# Polynomial Shear Thickening/Thinening
# Thickening N < 1 
# Thinning N > 1
from __future__ import division
from dolfin import * 
import numpy as np   
import copy as cp
import pylab
import mshr # For Meshes
import time
import pdb

# Plotting
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
cmap = plt.cm.Set1
cseq = [plt.cm.Set1(i) for i in np.linspace(0, 1, 9, endpoint=True)]
plt.rcParams['axes.color_cycle'] = cseq
import nb

#list_linear_solver_methods()
#list_krylov_solver_preconditioners()
##############################################################################
# -1 Issues

##############################################################################
# 0 Setup of problem
set_log_active(False) # No output from FEniCS
verify = False # Verify first and second Variation
LS = True # Use Line Search
PRE = True # Use user-defined Preconditioner else PETSc_amg
DIR = True # Use Directsolver, else krylovsolver
savemesh = False #Save a plot of the Mesh

# Direct Solver used, preconditioner is irrelevant
if DIR:
    PRE = False

# mesh setup
radius = .2 # 0<r<0.5 # For mesh

# Solver Setup
max_iter            = 50
maxbackit           = 30
linmaxit            = 10000
c_armijo            = 1e-4
linsol_rtol         = 1e-6
rtol                = 1e-10
atol                = 1e-8
gdvq_tol            = 1e-14

# Variational setup
eps = 1e-8
N = 2.
n = Constant(float(N)) 
mu = Constant(1.0)
inflowFactor = 1.
# Forcing as expression
class ForceExpr(Expression):
    def eval(self, value, x):
        value[0] = 0.0
        value[1] = 0.0
    def value_shape(self):
        return (2,)
forcing = ForceExpr(degree=2)

# Output setup
prefix = "figures/stokesPoly"
outname = prefix+str(N)
#outname = "testN"+str(N) # For testing


##############################################################################
# 1 Bulding Domain and marking subregions

#domain = mshr.Rectangle(Point(0,0), Point(3,1))
# Define rectangular domain with circular "hole" in the middle
domain = mshr.Rectangle(Point(0,0), Point(3,1))-mshr.Circle(Point(1.5,0.5), radius)
# Two holes
#domain = mshr.Rectangle(Point(0,0), Point(3,1)) -\
#mshr.Circle(Point(1.5,0.3), radius/2) -\
#mshr.Circle(Point(1.5,0.7), radius/2)


#mesh = mshr.generate_mesh(domain, 250) # For Testing
mesh = mshr.generate_mesh(domain, 150)
if savemesh:
    nb.plot(mesh)
    plt.savefig("figures/mesh.pdf")

# Subdomain for no-slip 
# marks whole boundary. Inflow and outflow will overwrite
class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Subdomain for inflow left
class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS and on_boundary

# Subdomain for outflow right
class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 3.0 -  DOLFIN_EPS and on_boundary

def boundary(x,on_boundary):
    return on_boundary

# Create mesh functions over the cell facets
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
# Marking all cells
sub_domains.set_all(3)

# Marking noslip boundary cells i.e. top, bottom and circle boundaries
noslip = Noslip()
noslip.mark(sub_domains, 0) 
# Marking inflow boundary i.e. left boundary
inflow = Inflow()
inflow.mark(sub_domains, 1)
# Marking outflow boundary i.e. right boundary
outflow = Outflow()
outflow.mark(sub_domains, 2)

##############################################################################
# 2 Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
VQ = MixedFunctionSpace([V, Q])

linmaxit = max(linmaxit,VQ.dim())

#pdb.set_trace()
##############################################################################
# 3 Boundary Conditions

# No slip Boundaries
noslip = Constant((0.0, 0.0))
bc0 = DirichletBC(VQ.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity, at x0 = 1
inStr = str(inflowFactor)+"*(-pow(x[1],2)+x[1])"
inflow = Expression((inStr, "0.0"), degree=2)
bc1 = DirichletBC(VQ.sub(0), inflow, sub_domains, 1)

#pressure outflow
pzero = Constant(0.)
bc2 = DirichletBC(VQ.sub(1), pzero, outflow)

# Neumann boundary on outflow, hence not specified.

# Collect boundary conditions
#bcs = [bc0, bc1] #ignoring Pressure and outflow
bcs = [bc0, bc1, bc2] #ignoring outflow

# Homogeneous Dirichlet for step
bcz0 = DirichletBC(VQ.sub(0), noslip, sub_domains, 0)
bcz1 = DirichletBC(VQ.sub(0), noslip, sub_domains, 1)
#bczero = [bcz0, bcz1]
bczero = [bcz0, bcz1, bc2]

# Homogeneous Dirichlet for verifiction
bcveri = DirichletBC(VQ.sub(0), noslip, boundary)

# Normal Direction at Inflow Boundary
nDir = FacetNormal(mesh)


##############################################################################
# 4 Linear Problem for initial guess
if N == 1.:
    print "Solving Linear Problem using FEniCS variational solver"
else:
    print "Solving Linear Problem for Initial Guess"

# Define variational problem
(uh, ph) = TrialFunctions(VQ)
(vh, qh) = TestFunctions(VQ)
f = forcing

Euh = Constant(0.5)*(nabla_grad(uh)+nabla_grad(uh).T)
Evh = Constant(0.5)*(nabla_grad(vh)+nabla_grad(vh).T)

a = inner(Euh, Evh)*dx + div(vh)*ph*dx + qh*div(uh)*dx
L = inner(f, vh)*dx 

# Compute solution
vq = Function(VQ)
solve(a == L, vq, bcs)
(uh, ph) = vq.split(deepcopy=True)

#plot(vq.sub(0))
if N == 1.: 
    ufile_pvd = File(outname+".pvd")
    ufile_pvd << uh

    nosample = 100
    xpos = [.0, .15, .3]
    ys = np.linspace(0,1,nosample)
    zs = np.zeros([nosample,7])
    
    #pdb.set_trace()
    for i in xrange(nosample):
        zs[i,:] = np.hstack([ys[i],uh(xpos[0],ys[i]),uh(xpos[1],ys[i]),uh(xpos[2],ys[i])])
    
    #Plotting flow profile
    plt.figure()

    plt.plot(zs[:,1],ys,'-',label = "$x = "+str(xpos[0])+"$",lw = 3)
    plt.plot(zs[:,3],ys,'--',label = "$x = "+str(xpos[1])+"$",lw = 3)
    plt.plot(zs[:,5],ys,':',label = "$x = "+str(xpos[2])+"$",lw = 3)
    plt.grid(True,alpha = .3)
    plt.xlim([0,1.1*np.max(zs[:,1:])])

    plt.title("Flowprofile x-direction",style = 'italic')
    plt.legend(framealpha=1)
    plt.xlabel("$u(x,y)$, Velocity in $x$ direction",style = 'italic')
    plt.ylabel("$y$")
    plt.savefig(outname+"FlowX.pdf")
    
    print "Problem Solved and saved." 
    exit()
#plh = plot(uh)
#plh.write_png(prefix+"ULin")
#plot(vq.sub(1))
#interactive()

##############################################################################
# 5 Define Energy functional, 1st and 2nd variation

# Define variational problem
(u, p) = vq.split()
(v, q) = TestFunctions(VQ)
(w, r) = TrialFunctions(VQ)

# Forcing
f = forcing 

# Energy Functional
def Ju(u,p):
    Eu = Constant(.5)*(nabla_grad(u) + nabla_grad(u).T)
    normEu = (inner(Eu,Eu)+eps)**(1./2.)
    
    #J = mu*(n/(1.+n))*pow(normEu,(1. + n)/n)*dx - inner(f,u)*dx
    J = mu*(n/(1.+n))*pow(normEu,(1. + n)/n)*dx - inner(f,u)*dx #+ p*nabla_div(u)*dx
    return J
    
# First Variation
def grad(u,v,p,q):
    Eu = Constant(.5)*(nabla_grad(u) + nabla_grad(u).T)
    Ev = Constant(.5)*(nabla_grad(v) + nabla_grad(v).T)
    normEu = (inner(Eu,Eu)+eps)**(1./2.)
    
    G = mu*(pow(normEu,(1. - n)/n)*inner(Eu,Ev))*dx -\
          inner(f,v)*dx +\
          p*nabla_div(v)*dx +\
          q*nabla_div(u)*dx 

    return G

def hess(u,v,w,p,q,r):
    Eu = Constant(.5)*(nabla_grad(u) + nabla_grad(u).T)
    Ev = Constant(.5)*(nabla_grad(v) + nabla_grad(v).T)
    Ew = Constant(.5)*(nabla_grad(w) + nabla_grad(w).T)
    normEu = (inner(Eu,Eu)+eps)**(1./2.)

    H = mu*((1.-n)/(n))*(pow(normEu, (1.-3.*n)/n)*inner(Eu,Ew)*inner(Eu,Ev))*dx +\
        mu*(pow(normEu, (1.-n)/n)*inner(Ew,Ev))*dx +\
        r*nabla_div(v)*dx +\
        q*nabla_div(w)*dx 
    
    return H

def precond(u,v,w,p,q,r):
    Eu = Constant(.5)*(nabla_grad(u) + nabla_grad(u).T)
    Ev = Constant(.5)*(nabla_grad(v) + nabla_grad(v).T)
    Ew = Constant(.5)*(nabla_grad(w) + nabla_grad(w).T)
    normEu = (inner(Eu,Eu)+eps)**(1./2.)

    P = mu*((1.-n)/(n))*(pow(normEu, (1.-3.*n)/n)*inner(Eu,Ew)*inner(Eu,Ev))*dx +\
        mu*(pow(normEu, (1.-n)/n)*inner(Ew,Ev))*dx +\
        q*r*dx
    
    return P

##############################################################################
# 5.5 Verification
if verify:
    print "verifying variations"
    u0 = interpolate(Expression(("x[1]","x[0]","0")), VQ) # div u = 0
    uh = Function(VQ)
    uh.assign(u0) 
    (u,p) = uh.split()

    J = Ju(u,p)
    G = grad(u,v,p,q)
    
    Ju0 = assemble(J)
    grad0 = assemble(G)
    
    n_eps = 32
    epsarr = 1e-2*np.power(2., -np.arange(n_eps))
    err_grad = np.zeros(n_eps)
        
    direc = Function(VQ)
    direc.vector().set_local(np.random.randn(VQ.dim()))
    #bcveri.apply(direc.vector())
    dir_grad0 = grad0.inner(direc.vector())
        
    for i in range(n_eps):
        uh.assign(u0)
        uh.vector().axpy(epsarr[i], direc.vector()) #uh = uh + eps[i]*direc
        (u,p) = uh.split()
        J = Ju(u,p)
        Juplus = assemble(J)
        err_grad[i] = abs( (Juplus - Ju0)/epsarr[i] - dir_grad0 )
    
    plt.figure()
    plt.loglog(epsarr, err_grad, "-ob")
    plt.loglog(epsarr, (.5*err_grad[0]/epsarr[0])*epsarr, "-.k")
    plt.title("Finite difference check of the first variation (gradient)",style = 'italic')
    plt.xlabel("$\epsilon$",style = 'italic')
    plt.ylabel("Error grad",style = 'italic')
    plt.legend(["Error Grad", "First Order"], "upper left")
    plt.grid(True,alpha = .3)
    plt.savefig(prefix+"VeriGrad.pdf")

    uh.assign(u0)
    
    H = hess(u,v,w,p,q,r)
    H_0 = assemble(H)
    
    err_H = np.zeros(n_eps)
    
    for i in range(n_eps):
        uh.assign(u0)
        uh.vector().axpy(epsarr[i], direc.vector())
        (u,p) = uh.split()
        G = grad(u,v,p,q)
        grad_plus = assemble(G)
    
        diff_grad = (grad_plus - grad0)
        diff_grad *= 1/epsarr[i]
        H_0dir = H_0 * direc.vector()
        err_H[i] = (diff_grad - H_0dir).norm("l2")
    
    plt.figure()
    plt.loglog(epsarr, err_H, "-ob")
    plt.loglog(epsarr, (.5*err_H[0]/epsarr[0])*epsarr, "-.k")
    plt.title("Finite difference check of the second variation (Hessian)",style = 'italic')
    plt.xlabel("$\epsilon$",style = 'italic')
    plt.ylabel("Error Hessian",style = 'italic')
    plt.legend(["Error Hessian", "First Order"], "upper left")
    plt.grid(True,alpha = .3)
    plt.savefig(prefix+"VeriHess.pdf")
    plt.close()

##############################################################################
# 6 Infinite-Dimensional Newton Descent
start = time.time()
print "Solving for Nonlinear Problem"
if DIR: 
    print "Using Direct Solver"
else:
    print "Using Krylov Solver Method"
    if PRE:
        print "with predefined preconditioner"
termination_reasons = ["Maximum number of Iteration reached",      #0
                       "Norm of the gradient less than tolerance", #1
                       "Norm of (g, dvq) less than tolerance",     #2
                       "Maximum number of backtracking reached"    #3
                       ]
print "Number of finite elements: {:d}".format(mesh.num_cells())
print "Degrees of freedom: {:d}".format(np.max(VQ.dofmap().dofs()))

converged = False

dvq = Function(VQ)
dvq.assign(vq)

J = Ju(u,p)
G = grad(u,v,p,q)
H = hess(u,v,w,p,q,r)

Ju0 = assemble(J)
Jn = cp.deepcopy(Ju0) # For line search
g0 = assemble(G)

# Setting variables
lin_it = 0
g0_norm = g0.norm("l2")
tol = max(g0_norm*rtol, atol)

print "-"*80
print   "{0:3}  {1:4}   {2:15} {3:15} {4:15} {5:15}".format(\
    "It", "lin_it", "Energy", "(g,dvq)", "||g||l2", "alpha")    
print   "{0:3d}  {1:4}  {2:15e} {3:15} {4:15e}    {5:15}".format(\
            0, "NA", Jn,"   NA", g0_norm,"NA")

# Standard Termination reason is max iteration
tr = termination_reasons[0]

for i in xrange(1,max_iter+1):
    [Hn, gn] = assemble_system(H, G, bczero)
    Hn.init_vector(dvq.vector(),1)

    # Preconditioner
    if PRE:
        P = precond(u,v,w,p,q,r)
        [Pn, gtemp] = assemble_system(P, G, bczero)
        if DIR:
            solver = LinearSolver("umfpack")
        else:
            solver = PETScKrylovSolver("minres")
            solver.set_operators(Hn,Pn)
    else:
        if DIR: 
            solver = LinearSolver("umfpack")
        else:
            solver = PETScKrylovSolver("minres","petsc_amg")
            solver.set_operator(Hn)

    if not DIR:
        solver.parameters["nonzero_initial_guess"] = True
        solver.parameters["maximum_iterations"] = linmaxit
        solver.parameters["relative_tolerance"] = linsol_rtol
    
    if DIR: 
        solver.solve(Hn,dvq.vector(),gn)
        myit = 0
    else:
        myit = solver.solve(dvq.vector(),gn)

    gn_norm = gn.norm("l2")
    dvq_gn = -dvq.vector().inner(gn)
    lin_it +=  myit  
    
    # At saddle point
    if gn_norm < tol:
        converged = True
        tr = termination_reasons[1]
        break 

    # ....
    if np.abs(dvq_gn) < gdvq_tol:
        converged=True
        tr = termination_reasons[2]
        break

    # Line Search
    alpha = 1.
    if LS:
        vq_back = vq.copy(deepcopy=True)
        bk_converged = False
        
        for j in xrange(maxbackit):
            vq.vector().axpy(-alpha,dvq.vector())
            (u,p) = vq.split()

            J = Ju(u,p)
            Jnext = assemble(J)

            if Jnext < Jn + np.abs(alpha*c_armijo*dvq_gn):
                Jn = Jnext
                bk_converged = True
                break
            
            alpha = alpha/2.
            vq.assign(vq_back)    
    
        if not bk_converged:
            vq.assign(vq_back)
            tr = termination_reasons[3]
            break
    else:
        vq.vector().axpy(-alpha,dvq.vector())
        (u,p) = vq.split()
        J = Ju(u,p)
        Jn = assemble(J)

    G = grad(u,v,p,q)
    H = hess(u,v,w,p,q,r)

    print   "{0:3d}  {1:4d}  {2:15e} {3:15e} {4:15e} {5:15e}".format(\
            i, myit, Jn, -gn.inner(dvq.vector()), gn_norm,alpha)
    

print   "{0:3d}  {1:4d}  {2:15e} {3:15e} {4:15e} {5:15e}".format(\
            i, myit, Jn, -gn.inner(dvq.vector()), gn_norm,alpha)

print "-"*80
print tr

if converged:
    print "Newton converged in ", i, "iterations"
    if not DIR:
        print "nonlinear iterations and ", lin_it, "linear iterations."
else:
    print "Newton did NOT converge after ", i, "iterations"
    if not DIR:
          "nonlinear iterations and ", lin_it, "linear iterations."

print "Final norm of the gradient", gn_norm
print "Value of the cost functional", Jn

elapsed = time.time() - start
print "Time elapsed: {0:10f} seconds".format(elapsed)

(u, p) = vq.split(deepcopy=True)
ufile_pvd = File(outname+".pvd")
ufile_pvd << u

#pdb.set_trace()
#pl = plot(u)
#plh.write_png(prefix+"U"+str(N))

##############################################################################
# 7 Plotting Flow Profile

nosample = 100
xpos = [.0, .15, .3]
ys = np.linspace(0,1,nosample)
zs = np.zeros([nosample,7])

#pdb.set_trace()
for i in xrange(nosample):
    zs[i,:] = np.hstack([ys[i],u(xpos[0],ys[i]),u(xpos[1],ys[i]),u(xpos[2],ys[i])])

#Plotting flow profile
plt.figure()
plt.plot(zs[:,1],ys,'-',label = "$x = "+str(xpos[0])+"$",lw = 3)
plt.plot(zs[:,3],ys,'--',label = "$x = "+str(xpos[1])+"$",lw = 3)
plt.plot(zs[:,5],ys,':',label = "$x = "+str(xpos[2])+"$",lw = 3)
plt.grid(True,alpha = .3)
plt.xlim([0,1.1*np.max(zs[:,1:])])
plt.title("Flowprofile x-direction",style = 'italic')
plt.legend(framealpha=1)
plt.xlabel("$u(x,y)$, Velocity in $x$ direction",style = 'italic')
plt.ylabel("$y$")
plt.savefig(outname+"FlowX.pdf")

#plt.figure()
#plt.plot(zs[:,1],ys)
#plt.title("Flowprofile y-direction")
#plt.xlabel("u in y direction")
#plt.ylabel("y")
#plt.savefig(outname+"FlowY.pdf")
#plt.close()
#plot(u)
#interactive()

##############################################################################
# 8 Free Boundary

Eu = Constant(.5)*(nabla_grad(u) + nabla_grad(u).T)
U = pow(inner(Eu,Eu),(1.-n)/(2.*n))

W = FunctionSpace(mesh, "Lagrange", 2)
Uproj = project(U,W)

ufile_pvd2 = File(outname+"Thick.pvd")
ufile_pvd2 << Uproj


#EuFreeB = inner(Eu,Eu)
#ufile_pvd = File(outname+"EU.pvd")
#ufile_pvd << EuFreeB
#EuhFreeB = inner(Euh,Euh)
#ufile_pvd = File(outname+"EULIN.pvd")
#ufile_pvd << EuhFreeB


#pdb.set_trace()










