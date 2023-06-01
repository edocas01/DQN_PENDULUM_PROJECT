from Classes.pendulum import Pendulum
import numpy as np
from numpy import pi
import time

class DPendulum:
    ''' Discrete Pendulum environment. Torques at the joints are discretized
        with the specified steps. Joint velocity and torque are saturated. 
        Guassian noise can be added in the dynamics. 
    '''
    def __init__(self, nbJoint = 1, dnu=11, vMax=5, uMax=5, dt=0.2, ndt=1, noise_stddev=0):
        self.pendulum = Pendulum(nbJoint,noise_stddev)
        self.pendulum.DT  = dt
        self.pendulum.NDT = ndt
        self.pendulum.vmax = vMax
        self.pendulum.umax = uMax
        self.nx = self.pendulum.nx # state dimension
        # self.nv = self.pendulum.nv -> NOT USED??
        self.vMax = vMax     # Max velocity (v in [-vmax,vmax])
        self.dnu = dnu       # Number of discretization steps for joint torque
        self.uMax = uMax     # Max torque (u in [-umax,umax])
        self.dt = dt         # time step
        self.DU = 2*uMax/dnu # discretization resolution for joint torque (for uniform discretization)
        self.u_values = self.define_u() # all possible values taken by the joint torque

    # Continuous to discrete joint torque (not really used, only in initialization for double pendulum)
    def c2du(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        return int(np.floor((u+self.uMax)/self.DU))
    
    # Discrete to continuous joint torque
    def d2cu(self, iu):
        iu = np.clip(iu,0,self.dnu-1) - (self.dnu-1)/2
        DU_fine = self.DU*2*np.abs(iu)/self.dnu
        return iu*DU_fine

    def reset(self,x=None):
        # reset using the reset of pendulum (since it is not discretized)
        self.x = self.pendulum.reset(x)
        return self.x

    def step(self,iu):
        # iu is a discrete torque index so we need to convert it to continuous to use it in pendulum
        u   = self.d2cu(iu)
        self.x, cost   = self.pendulum.step(u)
        return self.x, cost

    def render(self):
        q = self.x[0]
        self.pendulum.display(np.array([q,]))
        time.sleep(self.pendulum.DT)

    # define all possible values taken by the joint torque
    def define_u(self):
        u_values = np.zeros(self.dnu)
        for i in range(self.dnu):
            u_values[i] = self.d2cu(i)
        return u_values
    
    # TO BE REDEFINED
    def plot_V_table(self, V, q, dq, string):
        ''' Plot the given Value table V '''
        import matplotlib.pyplot as plt

        plt.figure()
        plt.pcolormesh(q, dq, V, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        title = 'V table'
        plt.title(title)
        plt.xlabel("Joint angle [rad]")
        plt.ylabel("Joint velocity [rad/s]")
        plt.savefig(string + title + '.png')
        
    def plot_policy(self, pi, q, dq, string):
        ''' Plot the given policy table pi '''
        import matplotlib.pyplot as plt

        plt.figure()
        plt.pcolormesh(q, dq, pi, cmap=plt.cm.get_cmap('RdBu'))
        plt.colorbar()
        title = 'Policy'
        plt.title(title)
        plt.xlabel("Joint angle [rad]")
        plt.ylabel("Joint velocity [rad/s]")
        plt.savefig(string + title + '.png')
        

    