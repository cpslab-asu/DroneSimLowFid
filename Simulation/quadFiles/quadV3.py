import pathlib

import numpy as np
from numpy import sin, cos, tan, pi, sign
from scipy.integrate import ode

from quadFiles.initQuad import sys_params, init_cmd, init_state
from utils.EKF import EKF_IMU
from utils import stateConversions as sC
from utils import rotationConversion as rC
import pandas as pd
import utils
import config
import pdb

deg2rad = pi/180.0
# Global variables for Kalman Filter
P = 1  # Initial covariance
K = 0  # Kalman gain
Q = 8.9  # Process noise covariance
R = 500 ** 2  # Measurement noise covariance
estimated_pressure = None  # Initial estimated pressure
excel_path = pathlib.Path(__file__).parent / "output.xlsx"  # Change this to your actual file path
df = pd.read_excel(excel_path, header=None)  # Assuming no headers
numerical_data = df.to_numpy()
upsampled_data = np.repeat(numerical_data, 60, axis=0)



class Quadcopter:

    def __init__(self, Ti):
        
        # Quad Params
        # ---------------------------
        self.params = sys_params()
        
        # Command for initial stable hover
        # ---------------------------
        ini_hover = init_cmd(self.params)
        self.params["FF"] = ini_hover[0]         # Feed-Forward Command for Hover
        self.params["w_hover"] = ini_hover[1]    # Motor Speed for Hover
        self.params["thr_hover"] = ini_hover[2]  # Motor Thrust for Hover  
        self.thr = np.ones(4)*ini_hover[2]
        self.tor = np.ones(4)*ini_hover[3]

        # Initial State
        # ---------------------------
        self.state = init_state(self.params)

        self.pos   = self.state[0:3]
        self.quat  = self.state[3:7]
        self.vel   = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = np.array([self.state[13], self.state[15], self.state[17], self.state[19]])
        self.vel_dot = np.zeros(3)
        self.omega_dot = np.zeros(3)
        self.acc = np.zeros(3)

        self.extended_state()
        self.forces()

        # Set Integrator
        # ---------------------------
        self.integrator = ode(self.state_dot).set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.state, Ti)
        self.ekf = EKF_IMU(dt=0.005, process_noise=10, measurement_noise=5)


    def extended_state(self):

        # Rotation Matrix of current state (Direct Cosine Matrix)
        self.dcm = utils.quat2Dcm(self.quat)

        # Euler angles of current state
        YPR = utils.quatToYPR_ZYX(self.quat)
        self.euler = YPR[::-1] # flip YPR so that euler state = phi, theta, psi
        self.psi   = YPR[0]
        self.theta = YPR[1]
        self.phi   = YPR[2]

    
    def forces(self):
        
        # Rotor thrusts and torques
        self.thr = self.params["kTh"]*self.wMotor*self.wMotor
        self.tor = self.params["kTo"]*self.wMotor*self.wMotor

    def state_dot(self, t, state, cmd, wind):

        # Import Params
        # ---------------------------    
        mB   = self.params["mB"]
        g    = self.params["g"]
        dxm  = self.params["dxm"]
        dym  = self.params["dym"]
        IB   = self.params["IB"]
        IBxx = IB[0,0]
        IByy = IB[1,1]
        IBzz = IB[2,2]
        Cd   = self.params["Cd"]
        
        kTh  = self.params["kTh"]
        kTo  = self.params["kTo"]
        tau  = self.params["tau"]
        kp   = self.params["kp"]
        damp = self.params["damp"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]

        IRzz = self.params["IRzz"]
        if (config.usePrecession):
            uP = 1
        else:
            uP = 0
    
        # Import State Vector
        # ---------------------------  
        x      = state[0]
        y      = state[1]
        z      = state[2]
        q0     = state[3]
        q1     = state[4]
        q2     = state[5]
        q3     = state[6]
        xdot   = state[7]
        ydot   = state[8]
        zdot   = state[9]
        p      = state[10]
        q      = state[11]
        r      = state[12]
        wM1    = state[13]
        wdotM1 = state[14]
        wM2    = state[15]
        wdotM2 = state[16]
        wM3    = state[17]
        wdotM3 = state[18]
        wM4    = state[19]
        wdotM4 = state[20]

        # Motor Dynamics and Rotor forces (Second Order System: https://apmonitor.com/pdc/index.php/Main/SecondOrderSystems)
        # ---------------------------
        
        uMotor = cmd
        wddotM1 = (-2.0*damp*tau*wdotM1 - wM1 + kp*uMotor[0])/(tau**2)
        wddotM2 = (-2.0*damp*tau*wdotM2 - wM2 + kp*uMotor[1])/(tau**2)
        wddotM3 = (-2.0*damp*tau*wdotM3 - wM3 + kp*uMotor[2])/(tau**2)
        wddotM4 = (-2.0*damp*tau*wdotM4 - wM4 + kp*uMotor[3])/(tau**2)
    
        wMotor = np.array([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, minWmotor, maxWmotor)
        thrust = kTh*wMotor*wMotor
        torque = kTo*wMotor*wMotor
    
        ThrM1 = thrust[0]
        ThrM2 = thrust[1]
        ThrM3 = thrust[2]
        ThrM4 = thrust[3]
        TorM1 = torque[0]
        TorM2 = torque[1]
        TorM3 = torque[2]
        TorM4 = torque[3]

        # Wind Model
        # ---------------------------
        [velW, qW1, qW2] = wind.randomWind(t)
        # velW = 0

        # velW = 5          # m/s
        # qW1 = 0*deg2rad    # Wind heading
        # qW2 = 60*deg2rad     # Wind elevation (positive = upwards wind in NED, positive = downwards wind in ENU)
    
        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        if (config.orient == "NED"):
            DynamicsDot = np.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 - 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 + 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 - (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) + g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + ( ThrM1 + ThrM2 - ThrM3 - ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IByy)*p*q - TorM1 + TorM2 - TorM3 + TorM4)/IBzz]])
        elif (config.orient == "ENU"):
            DynamicsDot = np.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 + 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 - 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 + (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) - g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + (-ThrM1 - ThrM2 + ThrM3 + ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IBzz)*p*q + TorM1 - TorM2 + TorM3 - TorM4)/IBzz]])
    
    
        # State Derivative Vector
        # ---------------------------
        sdot     = np.zeros([21])
        sdot[0]  = DynamicsDot[0]
        sdot[1]  = DynamicsDot[1]
        sdot[2]  = DynamicsDot[2]
        sdot[3]  = DynamicsDot[3]
        sdot[4]  = DynamicsDot[4]
        sdot[5]  = DynamicsDot[5]
        sdot[6]  = DynamicsDot[6]
        sdot[7]  = DynamicsDot[7]
        sdot[8]  = DynamicsDot[8]
        sdot[9]  = DynamicsDot[9]
        sdot[10] = DynamicsDot[10]
        sdot[11] = DynamicsDot[11]
        sdot[12] = DynamicsDot[12]
        sdot[13] = wdotM1
        sdot[14] = wddotM1
        sdot[15] = wdotM2
        sdot[16] = wddotM2
        sdot[17] = wdotM3
        sdot[18] = wddotM3
        sdot[19] = wdotM4
        sdot[20] = wddotM4

        self.acc = sdot[7:10]

        return sdot
        
    def add_noise(p, snr_db):
        """ Add Gaussian noise to p with given SNR in dB. """
        signal_power = np.var(p)  # Estimate signal power
        noise_power = signal_power / (10**(snr_db / 10))  # Compute noise power
        noise = np.random.normal(0, np.sqrt(noise_power), size=np.shape(p))  # Generate noise
        p_noisy = p + noise  # Add noise to signal
        return p_noisy    
        
    def barometer(self,x):
        """
    	Computes the corrected altitude based on measured pressure at a given altitude.
    
    	Parameters:
        	x (array-like): Altitude values in meters.
    
    	Returns:
        	numpy.ndarray: Corrected altitude values.
    	"""
    	# Constants
        P0 = 101325  # Sea level standard atmospheric pressure (Pa)
        T0 = 288.15  # Sea level standard temperature (K)
        L = 0.0065   # Temperature lapse rate (K/m)
        R = 287.05   # Specific gas constant for dry air (J/kg*K)
        g0 = 9.80665 # Gravity (m/s^2)
    
    	# True pressure based on the ISA model
        true_pressure = P0 * (1 - L * np.array(x) / T0) ** (g0 / (R * L))
    
    
        noise_std = 0.5  # Standard deviation of sensor noise (Pa)
        measured_pressure = true_pressure + noise_std * np.random.randn(*np.shape(x))
    
    	# Temperature compensation
        T_actual = T0 - L * np.array(x)  # Actual temperature at altitude
        altitude_corrected = (T_actual / L) * (1 - ((measured_pressure) / P0) ** (R * L / g0))
    
        return altitude_corrected
        
        
        
    def barometerV2(self,x,t):
    	
        global P, K, estimated_pressure
       
        P0 = 101325  # Sea level standard atmospheric pressure (Pa)
        T0 = 288.15  # Sea level standard temperature (K)
        L = 0.0065   # Temperature lapse rate (K/m)
        R = 287.05   # Specific gas constant for dry air (J/kg*K)
        g0 = 9.80665 # Gravity (m/s^2)
        true_pressure = P0 * (1 - L * np.array(x) / T0) ** (g0 / (R * L))
    	
        

        noise_std = 0.5  # Standard deviation of sensor noise (Pa)
        measured_pressure = true_pressure
        val = 0
        if t > 35 and t < 50 :
            #measured_pressure = true_pressure
            val = 8
        
        
        print(measured_pressure)
        # Initialize estimated pressure if it's the first measurement
        if estimated_pressure is None:
            estimated_pressure = measured_pressure
    
        
        
    
        P = P + Q
    
    
        K = P / (P + R)  
        estimated_pressure = estimated_pressure + K * (measured_pressure - estimated_pressure)
        P = (1 - K) * P
    
    
        T_actual = T0  # Assume measurement at sea level for simplicity
        altitude_corrected = (T_actual / L) * (1 - ((estimated_pressure + val)/ P0) ** (R * L / g0))
    
        return altitude_corrected

    def update(self, t, Ts, cmd, wind):

        prev_vel   = self.vel
        prev_omega = self.omega
        scale = 1
        cmd = np.array([scale*upsampled_data[np.floor(t/Ts).astype(int),0], scale*upsampled_data[np.floor(t/Ts).astype(int),1], scale*upsampled_data[np.floor(t/Ts).astype(int),2], scale*upsampled_data[np.floor(t/Ts).astype(int),3]])
        self.integrator.set_f_params(cmd, wind)
        self.state = self.integrator.integrate(t, t+Ts)
        if self.state[2] < 0 :
            self.state[2] = 0
            
        print(self.state[2])
        ### Rotor speed from an external source
        

        # Convert the DataFrame to a NumPy array
        
        
        # Assuming the file has two columns and you want to extract numerical data
        #numerical_data = df.select_dtypes(include=['number'])
        
        
        ### Sensing model
        
        # Gyroscope
        p = self.state[10]
        q = self.state[11]
        r = self.state[12]
        
                
        #p = (p+50*np.sin(t))
        #q = (q+50*np.sin(t))
        #r = (r+50*np.sin(t))
        #pdb.set_trace()
        #print("First Values:" )
        #print([p, q, r])
        self.ekf.predict([p, q, r])
        
        accel = sC.accelerometer_readings(self.acc[0], self.acc[1], self.acc[2], self.state[3], self.state[4], self.state[5], self.state[6] )
        
        self.ekf.update(accel,[0, 0, 0])
        
        roll, pitch, yaw = self.ekf.get_orientation()
        q1, q2, q3, q4 = rC.YPRToQuat(yaw,pitch,roll)
        #pdb.set_trace()
        #print([p,q,r])
        self.state[3] = q1
        self.state[4] = q2
        self.state[5] = q3
        self.state[6] = q4
        
        # Accelerometer
        
        
        
        
        # Magnetometer
        
        ###

        self.pos   = self.state[0:3]
        # Barometer
        #self.pos[2] = self.barometerV2(self.state[2],t)
        self.quat  = self.state[3:7]
        self.vel   = self.state[7:10]
        self.omega = self.state[10:13]
        scale = 1
        #self.wMotor = np.array([self.state[13], self.state[15], self.state[17], self.state[19]])
        #if len(numerical_data) < 1000 : 
        #self.wMotor = np.array([scale*upsampled_data[np.floor(t/Ts).astype(int),0], scale*upsampled_data[np.floor(t/Ts).astype(int),1], scale*upsampled_data[np.floor(t/Ts).astype(int),2], scale*upsampled_data[np.floor(t/Ts).astype(int),3]])
        #else :
        
                
            

        self.vel_dot = (self.vel - prev_vel)/Ts
        self.omega_dot = (self.omega - prev_omega)/Ts

        self.extended_state()
        self.forces()
