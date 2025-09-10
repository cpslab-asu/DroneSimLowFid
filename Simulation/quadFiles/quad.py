import logging
import pathlib

import cv2
import numpy as np
import pandas as pd
from numpy import sin, cos, tan, pi, sign
from scipy.integrate import ode
from scipy.spatial.transform import Rotation

from .. import utils, config
from ..opticalflow.optical_flow_emulation import CalculateOpticalFlow, SlidingBarAnimation
from ..utils.EKF import EKF_IMU
from ..utils import stateConversions as sC
from ..utils import rotationConversion as rC
from .initQuad import sys_params, init_cmd, init_state

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

_logger = logging.getLogger("dronesim2.quad")
_logger.addHandler(logging.NullHandler())

"""
optical_flow_to_velocity.py
 
Pipeline:
  pixel flow (px/s) -> image angular rates (rad/s) -> remove rotational component
  -> scale by distance -> rotate to navigation frame (NED or ENU)
 
Conventions (defaults used here):
  - Body frame (b): x forward, y right, z down (aerospace NED convention).
  - Camera frame (c): x right, y down, z forward (pinhole camera).
  - Navigation frame (n): choose NED or ENU via NAV_FRAME constant.
  - R_cb maps body->camera: v_c = R_cb @ v_b
  - R_nb maps body->nav:    v_n = R_nb @ v_b  (from your AHRS/EKF)
  - R_nc = R_nb @ R_bc, with R_bc = R_cb.T
 
Notes:
  - The small-angle “pure rotation flow” sign mapping used here matches common PX4/ArduPilot conventions:
      omega_rot_x ~ +omega_cam_y
      omega_rot_y ~ -omega_cam_x
    If your sign looks off, flip these two lines.
"""
 
from dataclasses import dataclass
import numpy as np
 
# ------------------------
# Config / Conventions
# ------------------------
NAV_FRAME = "NED"  # or "ENU". Only affects how you interpret R_nb from your estimator.
 
# ------------------------
# Data structures
# ------------------------
@dataclass
class CameraIntrinsics:
    fx: float  # focal length in pixels along u-axis
    fy: float  # focal length in pixels along v-axis
 
@dataclass
class CameraFOV:
    width_px: int
    height_px: int
    fov_x_rad: float  # horizontal FOV in radians
    fov_y_rad: float  # vertical FOV in radians
 
@dataclass
class MountAngles:
    # Camera mount angles relative to BODY axes (roll, pitch, yaw), radians.
    # These define R_cb = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (body->camera).
    roll: float
    pitch: float
    yaw: float
 
@dataclass
class FlowMeasurement:
    du_px_s: float   # optical flow in u (pixels/sec)
    dv_px_s: float   # optical flow in v (pixels/sec)
    quality: float = 1.0  # optional quality metric [0..1]
 
@dataclass
class GyroRates:
    p: float  # body rate about x_b [rad/s]
    q: float  # body rate about y_b [rad/s]
    r: float  # body rate about z_b [rad/s]
 
 
# ------------------------
# Intrinsics helpers
# ------------------------
def intrinsics_from_fov(fov: CameraFOV) -> CameraIntrinsics:
    """
    Approximate fx,fy from FOV and image size.
    fx ≈ (W/2) / tan(FOVx/2), fy ≈ (H/2) / tan(FOVy/2)
    """
    fx = (fov.width_px / 2.0) / np.tan(fov.fov_x_rad / 2.0)
    fy = (fov.height_px / 2.0) / np.tan(fov.fov_y_rad / 2.0)
    return CameraIntrinsics(fx=fx, fy=fy)
 
 
# ------------------------
# Rotation utilities
# ------------------------
def R_x(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]])
 
def R_y(b: float) -> np.ndarray:
    cb, sb = np.cos(b), np.sin(b)
    return np.array([[ cb, 0, sb],
                     [  0, 1,  0],
                     [-sb, 0, cb]])
 
def R_z(g: float) -> np.ndarray:
    cg, sg = np.cos(g), np.sin(g)
    return np.array([[ cg, -sg, 0],
                     [ sg,  cg, 0],
                     [  0,   0, 1]])
 
def R_cb_from_mount(angles: MountAngles) -> np.ndarray:
    """
    Build camera<-body rotation R_cb using body-fixed roll->pitch->yaw.
    v_c = R_cb @ v_b
    """
    return R_z(angles.yaw) @ R_y(angles.pitch) @ R_x(angles.roll)
 
def is_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> bool:
    should_be_I = R @ R.T
    I = np.eye(3)
    return np.allclose(should_be_I, I, atol=tol) and np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
 
 
# ------------------------
# Quaternion utilities (body->nav attitude)
# ------------------------
def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])
 
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
 
def dcm_from_quat(q):
    # expects unit quaternion [w, x, y, z]
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),         2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),         1 - 2*(x*x + y*y)]
    ])
 
 
# ------------------------
# Core conversions
# ------------------------
def pixel_flow_to_img_rates(flow: FlowMeasurement, K: CameraIntrinsics) -> np.ndarray:
    """
    Convert pixel flow (px/s) to angular rates on the image plane (rad/s).
    Returns [omega_x, omega_y] where:
      omega_x ~ rotation of sightline about camera y-axis (horizontal sweep)
      omega_y ~ rotation of sightline about camera x-axis (vertical sweep)
    """
    omega_x = flow.du_px_s / K.fx
    omega_y = flow.dv_px_s / K.fy
    return np.array([omega_x, omega_y])  # rad/s
 
def body_rates_to_camera(gyro_b: GyroRates, R_cb: np.ndarray) -> np.ndarray:
    """
    Map body rates (p,q,r) to camera rates via R_cb (v_c = R_cb v_b).
    Returns [omega_cam_x, omega_cam_y, omega_cam_z].
    """
    w_b = np.array([gyro_b.p, gyro_b.q, gyro_b.r])
    return R_cb @ w_b
 
def remove_rotational_component(img_rates_xy: np.ndarray, omega_cam_xyz: np.ndarray) -> np.ndarray:
    """
    Subtract the rotational image motion predicted from gyro (small-angle model).
    Using convention:
      omega_rot_x ≈ +omega_cam_y
      omega_rot_y ≈ -omega_cam_x
    Adjust signs if your axes differ.
    """
    omega_rot_x = +omega_cam_xyz[1]
    omega_rot_y = -omega_cam_xyz[0]
    omega_trans_x = img_rates_xy[0] - omega_rot_x
    omega_trans_y = img_rates_xy[1] - omega_rot_y
    return np.array([omega_trans_x, omega_trans_y])
 
def angular_to_camera_velocity(omega_trans_xy: np.ndarray, Z_m: float) -> np.ndarray:
    """
    Scale by distance to ground along optical axis (meters) to get linear velocity in camera frame.
    For nadir-looking small-tilt:
      V_cam_x ≈ -Z * omega_trans_x
      V_cam_y ≈ -Z * omega_trans_y
      V_cam_z ≈ 0 (no info from planar flow)
    """
    Vx = -Z_m * omega_trans_xy[0]
    Vy = -Z_m * omega_trans_xy[1]
    Vz = 0.0
    return np.array([Vx, Vy, Vz])  # camera frame
 
def R_nc_from_Rnb_and_Rcb(R_nb: np.ndarray, R_cb: np.ndarray) -> np.ndarray:
    """
    R_nc = R_nb @ R_bc, with R_bc = R_cb.T
    """
    R_bc = R_cb.T
    return R_nb @ R_bc
 
def rotate_cam_to_nav(V_cam_xyz: np.ndarray, R_nc: np.ndarray) -> np.ndarray:
    """
    V_nav = R_nc @ V_cam
    """
    return R_nc @ V_cam_xyz
 
 
# ------------------------
# Convenience: full pipeline
# ------------------------
def flow_to_nav_velocity(
    flow: FlowMeasurement,
    K: CameraIntrinsics,
    gyro_b: GyroRates,
    altitude_m: float,
    R_nb: np.ndarray,
    R_cb: np.ndarray
) -> dict:
    """
    Full pipeline from (px/s, gyro, altitude, attitude, mount) to nav velocity.
 
    Returns dict with:
      {
        'omega_img_xy_rad_s': [..,..],
        'omega_cam_xyz_rad_s': [..,..,..],
        'omega_trans_xy_rad_s': [..,..],
        'V_cam_m_s': [..,..,..],
        'V_nav_m_s': [..,..,..]
      }
    """
    assert is_rotation_matrix(R_nb), "R_nb must be a valid rotation matrix"
    assert is_rotation_matrix(R_cb), "R_cb must be a valid rotation matrix"
 
    omega_img_xy = pixel_flow_to_img_rates(flow, K)                      # rad/s
    omega_cam_xyz = body_rates_to_camera(gyro_b, R_cb)                   # rad/s
    omega_trans_xy = remove_rotational_component(omega_img_xy, omega_cam_xyz)
    V_cam = angular_to_camera_velocity(omega_trans_xy, altitude_m)       # m/s
    R_nc = R_nc_from_Rnb_and_Rcb(R_nb, R_cb)
    V_nav = rotate_cam_to_nav(V_cam, R_nc)
 
    return {
        'omega_img_xy_rad_s': omega_img_xy,
        'omega_cam_xyz_rad_s': omega_cam_xyz,
        'omega_trans_xy_rad_s': omega_trans_xy,
        'V_cam_m_s': V_cam,
        'V_nav_m_s': V_nav
    }
 
 
# ------------------------
# Example usage / sanity test
# ------------------------
"""
if __name__ == "__main__":
    # Example camera: 640x480, 90°x 60° FOV
    fov = CameraFOV(width_px=640, height_px=480,
                    fov_x_rad=np.deg2rad(90.0),
                    fov_y_rad=np.deg2rad(60.0))
    K = intrinsics_from_fov(fov)
 
    # Flow sample: features moving right 100 px/s and down 40 px/s
    flow = FlowMeasurement(du_px_s=100.0, dv_px_s=40.0, quality=0.8)
 
    # Gyro: small body rotation (rad/s)
    gyro = GyroRates(p=np.deg2rad(1.0), q=np.deg2rad(-0.5), r=np.deg2rad(0.0))
 
    # Camera mount: slightly canted; example: roll=0°, pitch=90° (nadir), yaw=180°
    # (Common for a downward-looking camera aligned with body x forward.)
    mount = MountAngles(roll=0.0, pitch=np.deg2rad(90.0), yaw=np.deg2rad(180.0))
    R_cb = R_cb_from_mount(mount)
    assert is_rotation_matrix(R_cb)
 
    # Attitude: vehicle level, heading north (R_nb ≈ Identity in NED)
    R_nb = np.eye(3)
 
    # Altitude / range to ground along optical axis (meters)
    Z = 2.0
 
    out = flow_to_nav_velocity(flow, K, gyro, Z, R_nb, R_cb)
 
    print("fx, fy (px):", K.fx, K.fy)
    print("omega_img_xy [rad/s]:", out['omega_img_xy_rad_s'])
    print("omega_cam_xyz [rad/s]:", out['omega_cam_xyz_rad_s'])
    print("omega_trans_xy [rad/s]:", out['omega_trans_xy_rad_s'])
    print("V_cam [m/s]:", out['V_cam_m_s'])
    print("V_nav [m/s]:", out['V_nav_m_s'])
 
    # Quick check: if you spin in place (no translation), flow should be ~pure rotation and
    # removing rotational component should push V towards ~0.
    # You can test by setting flow from gyro only and altitude arbitrary; V_nav should be ~0.
"""


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

        # Optical Flow
        # ---------------------------
        self.optical_flow_prev_vel = np.zeros((3,))
        self.optical_flow = CalculateOpticalFlow(
            SlidingBarAnimation(num_rgb_sets=3, frequency=126),
            lk_params={
                "winSize": (15, 15),
                "maxLevel": 2,
                "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            },
            feature_params={
                "maxCorners": 50,
                "qualityLevel": 0.3,
                "minDistance": 5,
                "blockSize": 5,
            },
            visualize=False,
        )
        mount = MountAngles(roll=0.0, pitch=np.deg2rad(90.0), yaw=np.deg2rad(180.0))
        self.R_cb = R_cb_from_mount(mount)

        # Attitude: vehicle level, heading north (R_nb ≈ Identity in NED)
        # self.R_nb = np.eye(3)
        # self.R_nb = np.array([
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 1],
        # ])

        fov = CameraFOV(
            width_px=25,
            height_px=25,
            fov_x_rad=np.deg2rad(42.0),
            fov_y_rad=np.deg2rad(42.0),
        )

        self.K = intrinsics_from_fov(fov)



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
        
        
        _logger.debug("Measured Pressure: %.4f", measured_pressure)
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
        if t > 35 and t <= 38 :
            cmd = cmd - 100
        if t > 38 :
            cmd = np.array([scale*upsampled_data[np.floor((t-60)/Ts).astype(int),0], scale*upsampled_data[np.floor((t-60)/Ts).astype(int),1], scale*upsampled_data[np.floor((t-60)/Ts).astype(int),2], scale*upsampled_data[np.floor((t-60)/Ts).astype(int),3]])
        self.integrator.set_f_params(cmd, wind)
        self.state = self.integrator.integrate(t, t+Ts)
        #if self.state[2] < 0 :
        #    self.state[2] = 0
            
            
        _logger.debug("Simulation time: %.2f, Altitude: %.4f", t, self.state[2])
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
        self.optical_flow.step(Ts)
     
        # Altitude / range to ground along optical axis (meters)
        Z = self.state[2]

        flow = FlowMeasurement(
            self.optical_flow.x_pixel_flow,
            self.optical_flow.y_pixel_flow,
            quality=1.0,
        )

        _logger.debug("Optical flow pixel flow reported: < %.6f, %.6f >", flow.du_px_s, flow.dv_px_s)

        # Create rotation matrix from current drone orientation
        rot = Rotation.from_quat(self.state[3:7])
        R_nb = rot.as_matrix()
     
        out = flow_to_nav_velocity(flow, self.K, GyroRates(p, q, r), Z, R_nb, self.R_cb)
        opticalflow_vel = out['V_nav_m_s']
        opticalflow_acc = (opticalflow_vel - self.optical_flow_prev_vel) / Ts  

        _logger.debug("Optical flow velocities reported: < %.6f, %.6f >", opticalflow_vel[0], opticalflow_vel[1])

        accel = sC.accelerometer_readings(self.acc[0], self.acc[1], self.acc[2], self.state[3], self.state[4], self.state[5], self.state[6])
        accel[0:2] = opticalflow_acc[0:2]  # Overwrite accelerometer acceleration values with acceleration computed from optical flow
        
        self.optical_flow_prev_vel = opticalflow_vel
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
        self.wMotor = np.array([self.state[13], self.state[15], self.state[17], self.state[19]])
        #if len(numerical_data) < 1000 : 
        #self.wMotor = np.array([scale*upsampled_data[np.floor(t/Ts).astype(int),0], scale*upsampled_data[np.floor(t/Ts).astype(int),1], scale*upsampled_data[np.floor(t/Ts).astype(int),2], scale*upsampled_data[np.floor(t/Ts).astype(int),3]])
        #else :
        
                
            

        self.vel_dot = (self.vel - prev_vel)/Ts
        self.omega_dot = (self.omega - prev_omega)/Ts

        self.extended_state()
        self.forces()
