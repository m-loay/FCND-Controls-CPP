import numpy as np 
import math
from math import sin, cos, tan, sqrt
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.mplot3d import Axes3D
# import jdc
import random

from solution import UDACITYDroneIn3D, UDACITYController
import testing

pylab.rcParams['figure.figsize'] = 10, 10

total_time = 20.0
dt = 0.01

t=np.linspace(0.0,total_time,int(total_time/dt))

omega_x = 0.8
omega_y = 0.4
omega_z = 0.4

a_x = 1.0 
a_y = 1.0
a_z = 1.0

x_path= a_x * np.sin(omega_x * t) 
x_dot_path= a_x * omega_x * np.cos(omega_x * t)
x_dot_dot_path= -a_x * omega_x**2 * np.sin(omega_x * t)

y_path= a_y * np.cos(omega_y * t)
y_dot_path= -a_y * omega_y * np.sin(omega_y * t)
y_dot_dot_path= -a_y * omega_y**2 * np.cos(omega_y * t)

z_path= a_z * np.cos(omega_z * t)
z_dot_path= -a_z * omega_z * np.sin(omega_z * t)
z_dot_dot_path= - a_z * omega_z**2 * np.cos(omega_z * t)

psi_path= np.arctan2(y_dot_path,x_dot_path)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x_path, y_path, z_path)
plt.title('Flight path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Planned path'],fontsize = 14)
plt.figure(figsize=(10,10))
plt.show()

ax = fig.gca(projection='3d')

u = np.cos(psi_path)
v = np.sin(psi_path)
w = np.zeros(psi_path.shape)
for i in range(0,z_path.shape[0],20):
    ax.quiver(x_path[i], y_path[i], z_path[i], u[i], v[i], w[i], length=0.2, normalize=True,color='green')

plt.title('Flight path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Planned yaw',],fontsize = 14)

plt.show()

# how fast the inner loop (Attitude controller) performs calculations 
# relative to the outer loops (altitude and position controllers).
inner_loop_relative_to_outer_loop = 10

# creating the drone object
drone = UDACITYDroneIn3D()
 
# creating the control system object 

control_system = UDACITYController(z_k_p=2.0, 
                            z_k_d=1.0, 
                            x_k_p=6.0,
                            x_k_d=4.0,
                            y_k_p=6.0,
                            y_k_d=4.0,
                            k_p_roll=8.0,
                            k_p_pitch=8.0,
                            k_p_yaw=8.0,
                            k_p_p=20.0,
                            k_p_q=20.0,
                            k_p_r=20.0)



# declaring the initial state of the drone with zero
# height and zero velocity 

drone.X = np.array([x_path[0],
                               y_path[0],
                               z_path[0],
                               0.0,
                               0.0,
                               psi_path[0],
                               x_dot_path[0],
                               y_dot_path[0],
                               z_dot_path[0],
                               0.0,
                               0.0,
                               0.0])

# arrays for recording the state history, 
# propeller angular velocities and linear accelerations
drone_state_history = drone.X
omega_history = drone.omega
accelerations = drone.linear_acceleration()
accelerations_history= accelerations
angular_vel_history = drone.get_euler_derivatives()

# executing the flight
for i in range(0,z_path.shape[0]):
    
    rot_mat = drone.R()

    c = control_system.altitude_controller(z_path[i],
                                           z_dot_path[i],
                                           z_dot_dot_path[i],
                                           drone.X[2],
                                           drone.X[8],
                                           rot_mat)
    
    b_x_c, b_y_c = control_system.lateral_controller(x_path[i],
                                                     x_dot_path[i],
                                                     x_dot_dot_path[i],
                                                     drone.X[0],
                                                     drone.X[6],
                                                     y_path[i],
                                                     y_dot_path[i],
                                                     y_dot_dot_path[i],
                                                     drone.X[1],
                                                     drone.X[7],
                                                     c) 
    
    for j in range(inner_loop_relative_to_outer_loop):
        rot_mat = drone.R()
        
        angular_vel = drone.get_euler_derivatives()
        
        u_bar_p, u_bar_q, u_bar_r = control_system.attitude_controller(
            b_x_c,
            b_y_c,
            psi_path[i],
            drone.psi,
            drone.X[9],
            drone.X[10],
            drone.X[11],
            rot_mat)
        
        drone.set_propeller_angular_velocities(c, u_bar_p, u_bar_q, u_bar_r)
        
        drone_state = drone.advance_state(dt/inner_loop_relative_to_outer_loop)
        
    # generating a history of the state history, propeller angular velocities and linear accelerations
    drone_state_history = np.vstack((drone_state_history, drone_state))
    
    omega_history=np.vstack((omega_history,drone.omega))
    accelerations = drone.linear_acceleration()
    accelerations_history= np.vstack((accelerations_history,accelerations))
    angular_vel_history = np.vstack((angular_vel_history,drone.get_euler_derivatives()))
    


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x_path, y_path, z_path,linestyle='-',marker='.',color='red')
ax.plot(drone_state_history[:,0],
         drone_state_history[:,1],
         drone_state_history[:,2],
         linestyle='-',color='blue')

plt.title('Flight path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Planned path','Executed path'],fontsize = 14)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

u = np.cos(psi_path)
v = np.sin(psi_path)
w = np.zeros(psi_path.shape)

drone_u = np.cos(drone_state_history[:,5])
drone_v = np.sin(drone_state_history[:,5])
drone_w = np.zeros(psi_path.shape)

for i in range(0,z_path.shape[0],20):
    ax.quiver(x_path[i], y_path[i], z_path[i], u[i], v[i], w[i], length=0.2, normalize=True,color='red')
    ax.quiver(drone_state_history[i,0], 
              drone_state_history[i,1], 
              drone_state_history[i,2], 
              drone_u[i], drone_v[i], drone_w[i], 
              length=0.2, normalize=True,color='blue')
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.title('Flight path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Planned yaw','Executed yaw'],fontsize = 14)

plt.show()

plt.plot(t,-omega_history[:-1,0],color='blue')
plt.plot(t,omega_history[:-1,1],color='red')
plt.plot(t,-omega_history[:-1,2],color='green')
plt.plot(t,omega_history[:-1,3],color='black')

plt.title('Angular velocities').set_fontsize(20)
plt.xlabel('$t$ [$s$]').set_fontsize(20)
plt.ylabel('$\omega$ [$rad/s$]').set_fontsize(20)
plt.legend(['P 1','P 2','P 3','P 4' ],fontsize = 14)

plt.grid()
plt.show()

err= np.sqrt((x_path-drone_state_history[:-1,0])**2 
             +(y_path-drone_state_history[:-1,1])**2 
             +(y_path-drone_state_history[:-1,2])**2)


plt.plot(t,err)
plt.title('Error in flight position').set_fontsize(20)
plt.xlabel('$t$ [$s$]').set_fontsize(20)
plt.ylabel('$e$ [$m$]').set_fontsize(20)
plt.show()

plt.plot(t,psi_path,marker='.')
plt.plot(t,drone_state_history[:-1,5])
plt.title('Yaw angle').set_fontsize(20)
plt.xlabel('$t$ [$s$]').set_fontsize(20)
plt.ylabel('$\psi$ [$rad$]').set_fontsize(20)
plt.legend(['Planned yaw','Executed yaw'],fontsize = 14)
plt.show()
