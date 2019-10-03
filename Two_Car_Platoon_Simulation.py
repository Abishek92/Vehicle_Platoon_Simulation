##Simulating the Platoon using Random Variables
import control
import numpy as np
import matplotlib.pyplot as plot
from scipy.integrate import odeint
from random import randint

rd = 200                #nominal distance
delx = randint(350,450) #actual distance
e = delx - rd           #error distance

#####################################################################################
#Kalman Filter Section 

# intial parameters
n_iter = 10
sz = (n_iter,) # size of array

z = np.random.normal(e,1,size=sz) # observations (normal about e, sigma=0.1)

Q = 1e-4 # process variance

# allocate space for arrays
ehat=np.zeros(sz)      # a posteri estimate of e
P=np.zeros(sz)         # a posteri error estimate
ehatminus=np.zeros(sz) # a priori estimate of e
Pminus=np.zeros(sz)    # a priori error estimate
K=np.zeros(sz)         # gain 

R = 1e-1 # estimate of measurement variance

# intial estimate
ehat[0] = 200.0
P[0] = 1.0

for k in range(1,n_iter):
    # time update
    ehatminus[k] = ehat[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    ehat[k] = ehatminus[k]+K[k]*(z[k]-ehatminus[k])
    P[k] = (1-K[k])*Pminus[k]

#Estimate of error = ehat


########################################################################################
#Adaptive Control Section 
def adaptive1(x1,t):
    #state vector
    e1       = x1[0]
    v_est1   = x1[1]

    #Constants
    k_v     = 4                       #feedback gain
    gamma_v = 2                       #adaptive gain


    # Distance error change and velocity estimate error change
    edot1       = -k_v*e1 + v_est1
    v_est1dot   = -gamma_v*e1

    # State derivatives
    return [edot1, v_est1dot]

x10 = [ehat[7], 50.0]
t_range = 10
t = np.arange(0.0, t_range, 0.01)

#Initializing and defining velocity and position data
nd          = 200                       #nominal distance to be maintained in mm
v1          = 300                       #velocity of vehicle 1 in mm/sec 
del_x1      = np.arange(0.0,t_range)    #Initializing actual distance in mm
vcap1       = np.arange(0.0,t_range)    #Initializing velocity estimate vcap in mm/sec
v2          = np.arange(0.0,t_range)    #Initializing input velocity v2 in mm/sec
k_v         = 1                         #feedback gain

#Solving the state space model using ODEint
x1 = odeint(adaptive1, x10, t)


pos_error1   = x1[:,0]
v_est_error1 = x1[:,1]

del_x1 = pos_error1 + nd              #Change in actual distance between the cars in mm
vcap1  = v1 - v_est_error1            #Change in velocity estimate in mm/sec
v2     = vcap1 + k_v*pos_error1       #Change in input to vehicle 2, V2 in mm/sec  or v2 = v1 - x1[:,1] + k_v*x1[:,0]

plot.figure()
plot.plot(z,'k+',label='noisy measurements')
plot.plot(ehat,'b-',label='a posteri estimate')
plot.axhline(e,color='g',label='truth value')
plot.plot(ehat[5],'r*',label= 'chosen estimate value')
plot.grid(color='k', linestyle='-', linewidth=0.3)
plot.legend()
plot.title('Estimate vs. iteration step', fontweight='bold')
plot.xlabel('Iteration')
plot.ylabel('Distance')

plot.figure()
plot.plot(x1[:,1],'r-', label = 'Velocity estimate error for car 2')
plot.plot(x1[:,0],'b-', label = 'Distance error')
plot.title('Velocity estimate error and Distance error')
plot.grid(color='k', linestyle='-', linewidth=0.3)
plot.legend()
plot.xlabel('time in milli seconds')
plot.ylabel('Velocity estimate error in mm/sec')

plot.figure()
plot.plot(x1[:,0],'b-', label = 'Distance error between cars 1 and 2')
plot.plot(del_x1,'g-',label = 'Actual distance between cars 1 and 2' )
plot.title('Distance error & Actual distance 1&2')
plot.grid(color='k', linestyle='-', linewidth=0.3)
plot.legend()
plot.xlabel('time in milli seconds')
plot.ylabel('Distance in mm')

plot.figure()
plot.plot(vcap1, 'b-', label = 'Velocity estimate')
plot.plot(v2, 'r-', label = 'Input to vehicle 2')
plot.title('Velocity estimate')
plot.grid(color='k', linestyle='-', linewidth=0.3)
plot.legend()
plot.xlabel('time in milli seconds')
plot.ylabel('Velocity estimate in mm/sec')
plot.show()
