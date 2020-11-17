#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import sys, petsc4py
#petsc4py.init(sys.argv)
from dolfin import *
from mshr import *
from ufl import RestrictedElement

import fem
from fem.functionspace import *
from fem.utils import inner_e


import argparse
import math
import os
import shutil
import sympy
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../")
from fem.util import save_timings


# In[2]:


#Import displacement from experience data
table = pd.read_excel(r'/home/xinyuan/Fenics/35_1_2.xlsx')
displacement = table.Disp
displacement

# In[3]:


# -----------------------------------------------------------------------------
# Parameters for DOLFIN and SOLVER
# -----------------------------------------------------------------------------
set_log_level(LogLevel.INFO)  # log level
# set some dolfin specific parameters
parameters["use_petsc_signal_handler"] = True
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
#parameters["mesh partitioner"] = "scotch";
# -----------------------------------------------------------------------------
# parameters of the solvers
solver_u_parameters = {"nonlinear_solver": "newton",
                       "newton_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 100,
                                          "absolute_tolerance": 1e-8,
                                          "relative_tolerance": 1e-8,
                                          "report": True,
                                          "error_on_nonconvergence": True}} 
                                         
#PETScOptions.set("help")
PETScOptions.set("snes_type","vinewtonssls")
PETScOptions.set("snes_converged_reason")
PETScOptions.set("snes_linesearch_type","basic") #shell basic l2 bt nleqerr cp
PETScOptions.set("ksp_type","preonly")
PETScOptions.set("pc_type","lu")
PETScOptions.set("pc_factor_mat_solver_type","mumps")
PETScOptions.set("snes_report")
PETScOptions.set("snes_monitor")
PETScOptions.set("snes_vi_zero_tolerance",1.e-6)
PETScOptions.set("snes_stol",1.e-6)
PETScOptions.set("snes_atol",1.e-8)
PETScOptions.set("snes_rtol",1.e-6)
PETScOptions.set("snes_max_it",150)
PETScOptions.set("snes_error_if_not_converged",1)
PETScOptions.set("snes_force_iteration",1)



# In[4]:
userpar = Parameters("user")
userpar.add("project", True)
parameters.add(userpar)

if userpar["project"] == True:
    userpar.add("MITC","project") 
    print("Solving the damage sub-problem in the PROJECTED SPACE with static condensation of the local variable")
else:
    userpar.add("MITC","full") 
    print("Solving the damage sub- problem in the FULL SPACE")
    
modelname = "CT_25°"
meshname  = modelname+"-mesh.xdmf"

savedir   = "CT/"+modelname+"/"

if MPI.rank(MPI.comm_world) == 0:
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)



#Geometry
l0= 50.; h0=48.; rayon=5.2; x_centre=60.; y_centre=57.5; c0=24.; a0=25; hc=0.01
s0=20.; x_centre1=10.;  x_centre2=10.;y_centre1=9.; y_centre2=39.;rayon_f=5.2;
x2=0;

m0 = 1000.;

# Meters
h0=h0/m0; 
rayon=rayon/m0;
x_centre=x_centre/m0
x_centre1=x_centre1/m0
x_centre2=x_centre2/m0
y_centre1=y_centre1/m0
y_centre2=y_centre2/m0
y_centre=y_centre/m0 
rayon_f=rayon_f/m0
c0=c0/m0;
a0=a0/m0; 
hc=hc/m0; 
s0=s0/m0;
#y1=y1/l0;
x2=x2/m0;
#y2=y2/l0;
#kk=kk/l0;
l0=l0/m0;

cell_size = 0.0001;
nel = l0/cell_size;

#mesh = Mesh()
geom=Rectangle(Point(0, 0), Point(l0, h0))
cutout1 = Circle(Point(x_centre1, y_centre1), rayon_f)
cutout2 = Circle(Point(x_centre2, y_centre2), rayon_f)



domain_vertices = [ Point(x2, (c0+hc)),Point(-0.001+x2, c0), Point(a0+x2, c0) ] 
crack = Polygon(domain_vertices)
plaque=geom-crack
plaq=plaque-cutout1
plaquue=plaq-cutout2
mesh = generate_mesh(plaquue, nel)

ndim = mesh.topology().dim()
# In[8]:


C11 = Constant(1);
C12 = Constant(0.5);
C44 = Constant(8.25) ;

#kink angle

theta0 = -25.;
theta0 = theta0*np.pi/180.0


# Constitutive matrix Cmat for the fourth order phase-field and its rotated matrix Cmatr
Cmat = [[C11, C12, 0], [C12, C11 , 0], [0, 0, C44]]
K = [[np.cos(theta0)**2, np.sin(theta0)**2 ,  2.0*np.cos(theta0)*np.sin(theta0)],             [np.sin(theta0)**2, np.cos(theta0)**2 , -2.0*np.cos(theta0)*np.sin(theta0)],             [-np.cos(theta0)*np.sin(theta0), np.cos(theta0)*np.sin(theta0) , np.cos(theta0)**2-np.sin(theta0)**2]]
Cmatr = np.matmul(np.matmul(K,Cmat), np.transpose(K))




#Rotated constitutive matrix
Cr11 = Cmatr[0,0]
Cr12 = Cmatr[0,1]
Cr14 = Cmatr[0,2]
Cr44 = Cmatr[2,2]



#Material constant
# Material constant
E = Constant(1980.);
nu = Constant(0.3)
Gc = Constant(6.5)
k_ell = Constant(1.e-6)  # residual stiffness
ell = 0.0006 #longueur interne
ut = 1.;



def down_boundary(x):
    return ((x[0]>(x_centre2-rayon_f)-DOLFIN_EPS and x[0]<(x_centre2+rayon_f)-DOLFIN_EPS )and (x[1]<(y_centre1+rayon_f) and x[1]>(y_centre1-rayon_f)))
def up_boundary(x):
    return((x[0]>(x_centre1-rayon_f)-DOLFIN_EPS and x[0]<(x_centre1+rayon_f)-DOLFIN_EPS )and (x[1]<(y_centre2+rayon_f) and x[1]>(y_centre2-rayon_f)))


# In[9]:


# dissipation potential
def w(alpha):
    return 9.0*alpha



# stiffness modulation with respect to damage
def aa(alpha):
    return E*(1-alpha)**2



#Strain and Stress

def eps(u):
    return sym(grad(u))

def sigma_0(u):
    mu = E/(2.*(1.+nu))
    lmbda = E*nu/(1.-nu**2.)
    return 2.*mu*eps(u) + lmbda*tr(eps(u))*Identity(ndim)



#   stress of damaged material
def sigma(u, alpha):
    return (aa(alpha)+k_ell)*sigma_0(u)


# In[10]:


# -----------------------------------------------------------------------------
# Variational formulation
# -----------------------------------------------------------------------------
# Create function space for 2D elasticity
V_u = VectorFunctionSpace(mesh, "Lagrange", 1)
# Create the function space for the damage using mixed formulation
# see fenics-shells for further details
element_alpha = FiniteElement("Lagrange", triangle, 1)
element_a = VectorElement("Lagrange", triangle, 2)
element_s = FiniteElement("N1curl", triangle, 1)
element_p = RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")
element = MixedElement([element_alpha,element_a,element_s,element_p])
V_alpha = FunctionSpace(mesh,element_alpha)
V_a = FunctionSpace(mesh,element_a)
V_s = FunctionSpace(mesh,element_s)
V_p = FunctionSpace(mesh,element_p)
V_damage = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
V_damage_F = V_damage.full_space
V_damage_P = V_damage.projected_space
assigner_F = FunctionAssigner(V_damage_F,[V_alpha,V_a,V_s,V_p])
assigner_P = FunctionAssigner(V_damage_P,[V_alpha,V_a])

# Define the function, test and trial fields
u  = Function(V_u, name="Displacement")
du = TrialFunction(V_u)
v  = TestFunction(V_u)

damage = Function(V_damage_F, name="Damage")
damage_trial = TrialFunction(V_damage_F)
damage_test = TestFunction(V_damage_F),
alpha, a, s, p = split(damage)
damage_p = Function(V_damage_P, name="Damage")


# In[11]:


# Define the bounds for the damage field
alpha_ub = Function(V_alpha)
alpha_lb = Function(V_alpha)
a_lb  = Function(V_a)
a_ub = Function(V_a)
s_lb = Function(V_s)
s_ub = Function(V_s)  
p_lb  = Function(V_p)
p_ub = Function(V_p)
alpha_ub.vector()[:] = 1.
alpha_lb.vector()[:] = 0.
for ub in [a_ub,s_ub,p_ub]:
    ub.vector()[:] = np.infty
for lb in [a_lb,s_lb,p_lb]:
    lb.vector()[:] = -np.infty

if userpar["MITC"] == "project":
    damage_lb = Function(V_damage_P); 
    damage_ub = Function(V_damage_P)
    assigner_P.assign(damage_ub,[alpha_ub, a_ub])
    assigner_P.assign(damage_lb,[alpha_lb, a_lb])
else:
    damage_lb = Function(V_damage_F); 
    damage_ub = Function(V_damage_F)
    assigner_F.assign(damage_ub,[alpha_ub,a_ub,s_ub,p_ub])
    assigner_F.assign(damage_lb,[alpha_lb,a_lb,s_lb,p_lb])


# In[12]:


# Boundary conditions
g1 = Expression((0, "t"), t=0.00051, degree=0)
bc_u1 = DirichletBC(V_u, g1, up_boundary )
#bc_u2 = DirichletBC(V_u, g2, down_boundary )
bc_u2 = DirichletBC(V_u, Constant((0.,0.)), down_boundary )
bc_u = [bc_u1, bc_u2]


# In[13]:


# Energies
elastic_energy = 1./2.*inner(sigma(u, alpha), eps(u))*dx
# Weak form of elasticity problem
E_u  = derivative(elastic_energy,u,v)
# Writing tangent problems in term of test and trial functions for matrix assembly
E_du = derivative(E_u, u, du)

# Variational problem for the displacement
problem_u = NonlinearVariationalProblem(E_u, u, bc_u, J=E_du)
# Set up the solvers                                        
solver_u  = NonlinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)
solver_u.solve()
#plt.figure(figsize=(15, 15))
#plot(mesh)
#plot(u, mode = 'displacement')


# In[14]:


bc_a1 = DirichletBC(V_damage_P.sub(0), Constant(0.), up_boundary)
bc_a2 = DirichletBC(V_damage_P.sub(0), Constant(0.), down_boundary)
bc_damage = [bc_a1, bc_a2]


# In[15]:


#--------------------------------------------------------------------
# Damage subproblem
#--------------------------------------------------------------------
kappa_tensor = sym(grad(a)) # Hessian matrix of damage field
kappa = as_vector([kappa_tensor[0,0], kappa_tensor[1,1], kappa_tensor[0,1]])
# Voigt notation for fourth-order tensor Cr
Crv = as_matrix([[Cr11, Cr12, 2.0*Cr14],                  [Cr12, Cr11, -2.0*Cr14],                  [2.0*Cr14, -2.0*Cr14, 4.0*Cr44]])

dissipated_energy = Constant(5.0/96.0)*Gc*(w(alpha)/ell+pow(ell,3)*dot(kappa, Crv*kappa))*dx

penalty_energy = Constant(1.0e+3)*inner(s, s)*dx
# Here we show another way to apply the Duran-Liberman reduction operator,
# through constructing a Lagrangian term L_R.
# -----------------------------------------------------------------------------
# Impose the constraint that s=(grad(w)-theta) in a weak form
constraint = inner_e(grad(alpha)-a-s, p, False)
damage_functional = elastic_energy + dissipated_energy + penalty_energy + constraint
# Compute directional derivative about alpha in the test direction (Gradient)
F = derivative(damage_functional, damage, damage_test)
# Compute directional derivative about alpha in the trial direction (Hessian)
J = derivative(F, damage, damage_trial)
# =============================================================================


# In[16]:


file_u = XDMFFile(MPI.comm_world, savedir+"/u.xdmf")
file_alpha = XDMFFile(MPI.comm_world, savedir+"/alpha.xdmf")
file_a = XDMFFile(MPI.comm_world, savedir+"/a.xdmf")
for file in [file_u,file_alpha,file_a]:
    file.parameters["flush_output"]=True
    file.parameters["rewrite_function_mesh"]=False


# In[17]:


load_multipliers = displacement.to_numpy()
energies = np.zeros((len(load_multipliers),4))
iterations = np.zeros((len(load_multipliers),2))

# Numerical parameters of the alternate minimization
maxiteration = 2000
AM_tolerance = 1e-4

# In[18]:


# Define the damage problem
(alpha_0, a_0, s_0, p_0) = damage.split(deepcopy=True)
if userpar["MITC"] == "project":
    problem_damage = fem.ProjectedNonlinearProblem(V_damage_P, F, damage, damage_p, bcs=bc_damage, J=J)
    damage_ = damage_p
else:
    problem_damage = fem.FullNonlinearProblem(V_damage_F, F, damage, bcs=bc_damage, J=J)
    damage_ = damage


# In[19]:


# Initialize the damage snes solver and set the bounds
solver_damage = PETScSNESSolver()
solver_damage.set_from_options()
snes = solver_damage.snes()
snes.setType("vinewtonssls")
as_vec = lambda field:  as_backend_type(field.vector()).vec() 
snes.setSolution(as_vec(damage_))
snes.setVariableBounds(as_vec(damage_lb),as_vec(damage_ub))


# In[ ]:


# Iterate on the time steps and solve
for (i_t, t) in enumerate(load_multipliers):
    g1.t = ut*t
    if MPI.rank(MPI.comm_world) == 0:
        print("\033[1;32m--- Starting of Time step {0:2d}: t = {1:4f} ---\033[1;m".format(i_t, t))
    # Alternate Mininimization
    # Initialization
    iteration = 1
    err_alpha = 1.0
    # Iterations
    while err_alpha > AM_tolerance and iteration < maxiteration:
        # solve elastic problem
        solver_u.solve()
        # solve damage problem
        solver_damage.solve(problem_damage,damage_.vector())
        problem_damage.counter = 0
        # check error
        (alpha_1, a_1, s_1, p_1) = damage.split(deepcopy=True)
        alpha_error = alpha_1.vector() - alpha_0.vector()
        err_alpha   = alpha_error.norm('linf')
        # monitor the results
        if MPI.rank(MPI.comm_world) == 0:
            print ("AM Iteration: {0:3d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha))
        # update iterations
        alpha_0.assign(alpha_1)
        iteration = iteration+1

    # updating the lower bound to account for the irreversibility
    if userpar["MITC"] == "project":
        assigner_P.assign(damage_lb,[project(alpha_1,V_alpha),a_lb])
    else:
        assigner_F.assign(damage_lb,[project(alpha_1,V_alpha),a_lb,s_lb,p_lb])

    # ----------------------------------------
    # Some post-processing
    # ----------------------------------------
    # Dump solution to file
    damage.split()[1].rename("Gradient_Damamge", "a")
    file_a.write(damage.split()[1],t)
    damage.split()[0].rename("Damage", "alpha")
    file_alpha.write(damage.split()[0],t)
    u.rename("Displacement", "u")
    file_u.write(u,t)
    iterations[i_t] = np.array([t,iteration])
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, elastic_energy_value+surface_energy_value])    
    if MPI.rank(MPI.comm_world) == 0:
        print("\nEnd of timestep {0:3d} with load multiplier {1:4f}".format(i_t, t))
        print("\nElastic and Surface Energies: [{0:6f},{1:6f}]".format(elastic_energy_value, surface_energy_value))
        print("\nElastic and Surface Energies: [%s,%s]"%(elastic_energy_value, surface_energy_value))
        print("-----------------------------------------")
        # Save some global quantities as a function of the time
        np.savetxt(savedir+'/energies.txt', energies)
        np.savetxt(savedir+'/iterations.txt', iterations)
        print("Results saved in ", savedir)
    save_timings(savedir)





