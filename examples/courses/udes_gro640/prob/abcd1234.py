#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:19:16 2020

@author: alex
------------------------------------


Fichier d'amorce pour les livrables de la problématique GRO640'


"""

import numpy as np

from pyro.control  import robotcontrollers
from pyro.control.robotcontrollers import EndEffectorPD
from pyro.control.robotcontrollers import EndEffectorKinematicController


###################
# Part 1
###################

def dh2T( r , d , theta, alpha ):
    """

    Parameters
    ----------
    r     : float 1x1
    d     : float 1x1
    theta : float 1x1
    alpha : float 1x1
    
    4 paramètres de DH

    Returns
    -------
    T     : float 4x4 (numpy array)
            Matrice de transformation

    """
    
    T = np.zeros((4,4))
    
    T[0, 0] = np.cos(theta)
    T[0, 1] = -np.sin(theta) * np.cos(alpha)
    T[0, 2] = np.sin(theta) * np.sin(alpha)
    T[0, 3] = r * np.cos(theta)
    
    T[1, 0] = np.sin(theta)
    T[1, 1] = np.cos(theta) * np.cos(alpha)
    T[1, 2] = -np.cos(theta) * np.sin(alpha)
    T[1, 3] = r * np.sin(theta)
    
    T[2, 0] = 0
    T[2, 1] = np.sin(alpha)
    T[2, 2] = np.cos(alpha)
    T[2, 3] = d
    
    T[3, 0] = 0
    T[3, 1] = 0
    T[3, 2] = 0
    T[3, 3] = 1
    

    return T



def dhs2T( r , d , theta, alpha ):
    """

    Parameters
    ----------
    r     : float nx1
    d     : float nx1
    theta : float nx1
    alpha : float nx1
    
    Colonnes de paramètre de DH

    Returns
    -------
    WTT     : float 4x4 (numpy array)
              Matrice de transformation totale de l'outil

    """
    
    WTT = np.zeros((4,4))
    
    ###################
    # Votre code ici
    ###################

    for i in range(len(r)):
        T = dh2T(r[i], d[i], theta[i], alpha[i])
        if i == 0:
            WTT = T
        else:
            WTT = WTT @ T
    
    return WTT


def f(q):
    """
    

    Parameters
    ----------
    q : float 6x1
        Joint space coordinates

    Returns
    -------
    r : float 3x1 
        Effector (x,y,z) position

    """
    r = np.zeros((3,1))
    
    ###################
    # Votre code ici
    ###################
    d =     [0.147,   0,     0,     0,                 0.165,    0.009 + q[5]]
    theta = [q[0],    q[1],  q[2],  q[3] + np.pi/2,    q[4],     -np.pi/2    ]           
    r_dh =  [0.039,   0.155, 0.136, 0,                 0,        0.053       ]
    alpha = [np.pi/2, 0,     0,     np.pi/2,           -np.pi/2, 0           ]

    T = dhs2T( r_dh , d , theta, alpha )
    r = T[0:3,3]
    
    return r


########################
# Part 2    |   By RG  #
########################
class CustomPositionController( EndEffectorKinematicController ) :
    """ 
    Kinematic effector coordinates controller using the Jacobian of the system
    ------------------------------------------
    r = r_d : reference signal vector  e   x 1
    y = q   : sensor signal vector     dof x 1
    u = dq  : control inputs vector    dof x 1
    t       : time                     1   x 1
    -------------------------------------------
    u = c( y , r , t ) = J(q)^T *  [ (r - r_robot(q)) * k ]

    """
    
    ############################
    def __init__(self, manipulator, k = 1 ):
        """ """
        
        # Using functions from robot model
        self.fwd_kin = manipulator.forward_kinematic_effector
        self.J       = manipulator.J
        self.e       = manipulator.e # nb of effector dof
        
        # Dimensions
        self.dof = manipulator.dof
        self.k   = self.e 
        self.m   = self.dof
        self.p   = self.dof
        EndEffectorKinematicController.__init__( self, manipulator, 1)

        # Label
        self.name = 'End Effector Kinematic Controller'
        
        # Gains
        self.gains = np.ones( self.e  ) * k
        
        # Damping factor for least square solution
        self.lambda_ = 0.1
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUTPUTS
        u  : control inputs vector    m x 1
        
        """
        
        #u = np.zeros(self.m) 
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        # Error=
        e  = r_desired - r_actual
        # Effector space speed
        dr_r = e * self.gains
        

        ### Least square solution by RG ###
        dq = (J.T @ J + self.lambda_**2 * np.eye(self.dof)) @ J.T @ dr_r
            
        
        return dq
    
###################
# Part 3
###################
        

class CustomDrillingController( robotcontrollers.RobotController) :
    """ 

    """
    
    ############################
    def __init__(self, robot_model, k = 1 ):
        """ """
        
        super().__init__( dof = 3 )
        
        self.robot_model = robot_model
        
        # Label
        self.name = 'Custom Drilling Controller'
        # Gains
        self.e       = robot_model.e # nb of effector dof
        self.gains = np.ones( self.e  ) * k
        self.pos_gains = np.diag([50, 50, 25])
        self.r_d = np.array([0.25,0.25,0.405])
        self.Kp = np.diag([200.0, 200.0, 100.0])
        self.Kd = np.diag([30.0, 30.0, 20.0])
        self.drilling_pos_reached = False
        self.control_mode = "IMPEDANCE"
        print('Controller initialized')
        print('Gains: ' , self.gains)
        print('Position gains: ' , self.pos_gains)
        print('Desired position: ' , self.r_d)
        

        
        
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        # Ref
        f_e = np.array([0,0,-200]) # Force de forage souhaitée en N
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Robot model
        r = self.robot_model.forward_kinematic_effector( q ) # End-effector actual position
        J = self.robot_model.J( q )      # Jacobian matrix
        g = self.robot_model.g( q )      # Gravity vector
        H = self.robot_model.H( q )      # Inertia matrix
        C = self.robot_model.C( q , dq ) # Coriolis matrix
            
        # Jacobian computation
        J = self.robot_model.J( q )

        # Effector space position error
        e = self.r_d - r
        e_max = 0.01
        #Loi de commande:
        #Approche du robot vers la position de forage
        if not self.drilling_pos_reached:
            if (all(np.abs(x) < e_max for x in e)):
                self.drilling_pos_reached = True
                print('Drilling position reached, switching to force control')
                #print('Force control')

            if(self.control_mode == "POSITION"):
                # From effector target to joint torque
                cartesian_force = self.pos_gains @ e
                u = J.T @ cartesian_force + g
                #print('Position control')
            else:
                #impedance control
                cartesian_force = self.pos_gains @ e - self.Kd @ (J @ dq)
                u = J.T @ cartesian_force + g
                #print('Force control')
        # Forage
        else:
            # Force control
            if (r[2] > 0.21):
                #Forage en controle de force
                #u = J.T @ (np.eye(3) @ f_e + g)

                #Forage hybride impédance-force
                u= J.T @ (self.Kp @ e + self.Kd @ (-J @ dq) + f_e) + g
            else:
                u = g
                print('Drilling complete, stopping robot')
                self.drilling_pos_reached = False

        
        return u
        
    
###################
# Part 4
###################
        
    
def goal2r( r_0 , r_f , t_f ):
    """
    
    Parameters
    ----------
    r_0 : numpy array float 3 x 1
        effector initial position
    r_f : numpy array float 3 x 1
        effector final position
    t_f : float
        time 

    Returns
    -------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l

    """
    # Time discretization
    l = 1000 # nb of time steps
    
    # Number of DoF for the effector only
    m = 3
    
    r = np.zeros((m,l))
    dr = np.zeros((m,l))
    ddr = np.zeros((m,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    t_array = np.linspace(0, t_f, l)

    t_a = t_f / 4.0
    v_max = 1.0 / (t_f - t_a)
    a_c = v_max / t_a
    
    s = np.zeros(l)
    s_dot = np.zeros(l)
    s_ddot = np.zeros(l)
    
    for i, t in enumerate(t_array):
        if t < t_a: #Accélération
            s[i] = 0.5 * a_c * t**2
            s_dot[i] = a_c * t
            s_ddot[i] = a_c
        elif t < (t_f - t_a): #Vitesse constante
            s[i] = 0.5 * a_c * t_a**2 + v_max * (t - t_a)
            s_dot[i] = v_max
            s_ddot[i] = 0.0
        else: #Décélération
            t_d = t - (t_f - t_a)
            s[i] = (0.5 * a_c * t_a**2 + v_max * (t_f - 2*t_a) 
                    + v_max * t_d - 0.5 * a_c * t_d**2)
            s_dot[i] = v_max - a_c * t_d
            s_ddot[i] = -a_c

    m = len(r_0)
    
    D = (r_f - r_0).reshape(m, 1)
    
    r = r_0.reshape(m, 1) + D * s
    dr = D * s_dot
    ddr = D * s_ddot
    
    return r, dr, ddr


def r2q( r, dr, ddr , manipulator ):
    """

    Parameters
    ----------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l

    """
    # Time discretization
    l = r.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    q = np.zeros((n,l))
    dq = np.zeros((n,l))
    ddq = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    l1 = manipulator.l1
    l2 = manipulator.l2
    l3 = manipulator.l3
    
    #Position
    r1 = np.sqrt(r[0,:]**2 + (r[1,:])**2)
    r2 = (r[2, :] - l1)
    r3 = np.sqrt(r1**2 + r2**2)

    phi1 = np.arccos(np.clip((l3**2 - l2**2 - r3**2) / (-2 * l2 * r3), -1.0, 1.0))
    phi2 = np.arctan2(r2, r1)
    phi3 = np.arccos(np.clip((r3**2 - l2**2 - l3**2) / (-2 * l2 * l3), -1.0, 1.0))
    
    q[0, :] = np.arctan2(r[1,:], r[0,:])
    q[1, :] = phi1 + phi2
    q[2, :] = phi3 - np.pi
   
    #Vitesse
    for i in range(l):
        J = manipulator.J(q[:, i])
        dq[:, i] = np.linalg.pinv(J) @ dr[:, i]

    #Accélération
    mid = max(1, l // 2)
    v_norm = np.linalg.norm(dr[:, mid])
    if v_norm > 1e-8:
        dt_estime = np.linalg.norm(r[:, mid] - r[:, mid - 1]) / v_norm
    else:
        dt_estime = 0.01

    ddq = np.gradient(dq, dt_estime, axis=1)
    
    return q, dq, ddq



def q2torque( q, dq, ddq , manipulator ):
    """

    Parameters
    ----------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    tau   : numpy array float 3 x l

    """
    # Time discretization
    l = q.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    tau = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return tau