import math
import logging
from typing import List, Type, Optional, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.planner.project2.bfs_router import BFSRouter
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from nuplan.planning.simulation.planner.project2.simple_predictor import SimplePredictor
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor
from nuplan.planning.simulation.planner.project2.merge_path_speed import transform_path_planning, cal_dynamic_state, cal_pose
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.planning.simulation.planner.project2.frame_transform import cartesian2frenet, frenet2cartesian, local2global_vector
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

logger = logging.getLogger(__name__)

class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.x = []
        self.y = []
        self.yaw = []
        self.cost = 0.0

class MyPlanner(AbstractPlanner):
    """
    Planner going straight.
    """

    def __init__(
            self,
            horizon_seconds: float,
            sampling_time: float,
            max_velocity: float = 5.0,

    ):
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param max_velocity: [m/s] ego max velocity.
        """
        self.horizon_time = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.max_velocity = max_velocity

        self._router: Optional[BFSRouter] = None
        self._predictor: AbstractPredictor = None
        self._reference_path_provider: Optional[ReferenceLineProvider] = None
        self._routing_complete = False

        self._last_lead_detection_time = 0.0
        self._last_lead_speed = 0.0

        self.min_safe_distance = 10.0
        self.time_gap = 1.8
        self.max_decel = 3.0
        self.max_accel = 2.0

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._router = BFSRouter(initialization.map_api)
        self._router._initialize_route_plan(initialization.route_roadblock_ids)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """
        # 1. Routing
        ego_state, observations = current_input.history.current_state
        if not self._routing_complete:
            self._router._initialize_ego_path(ego_state, self.max_velocity)
            self._routing_complete = True

        # 2. Generate reference line
        self._reference_path_provider = ReferenceLineProvider(self._router)
        self._reference_path_provider._reference_line_generate(ego_state)

        # 3. Objects prediction
        self._predictor = SimplePredictor(ego_state, observations, self.horizon_time.time_s, self.sampling_time.time_s)
        objects = self._predictor.predict()

        # 4. Planning
        trajectory: List[EgoState] = self.planning(ego_state, self._reference_path_provider, objects,
                                                  self.horizon_time, self.sampling_time, self.max_velocity)

        return InterpolatedTrajectory(trajectory)

    def planning(self,
             ego_state: EgoState,
             reference_path_provider: ReferenceLineProvider,
             objects: List[TrackedObjects],
             horizon_time: TimePoint,
             sampling_time: TimePoint,
             max_velocity: float) -> List[EgoState]:
    
        x_set = [ego_state.car_footprint.center.x]
        y_set = [ego_state.car_footprint.center.y]
    
        global_vx, global_vy = local2global_vector(ego_state.dynamic_car_state.rear_axle_velocity_2d.x, ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
                                                   ego_state.car_footprint.center.heading)
        vx_set = [global_vx]
        vy_set = [global_vy]

        global_ax, global_ay = local2global_vector(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
                                                   ego_state.car_footprint.center.heading)
        ax_set = [global_ax]
        ay_set = [global_ay]

        frenet_path_x = reference_path_provider._x_of_reference_line
        frenet_path_y = reference_path_provider._y_of_reference_line
        frenet_path_heading = reference_path_provider._heading_of_reference_line
        frenet_path_kappa = reference_path_provider._kappa_of_reference_line
        frenet_path_s = reference_path_provider._s_of_reference_line


        if not frenet_path_x or len(frenet_path_x) < 2: 
            return [ego_state]

        try:
            s_set, l_set, s_dot_set, l_dot_set, dl_set, l_dot2_set, s_dot2_set, ddl_set = cartesian2frenet(x_set, y_set, vx_set, vy_set, ax_set, ay_set,
                                                                                                           frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, frenet_path_s)
            current_s = s_set[0]
            current_speed = s_dot_set[0]
            current_accel = s_dot2_set[0]

        except Exception as e:
            logger.error(f"Frenet conversion failed: {e}. Returning current state.")
            return [ego_state]

         
        closest_lead_car = None
        min_lead_distance = float('inf')
        current_time = ego_state.time_point.time_s
 
        for obj in objects:
            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                obj_s, obj_l, obj_s_dot, _, _, _, _, _ = cartesian2frenet([obj.box.center.x], [obj.box.center.y], [obj.velocity.x], [obj.velocity.y], [0.0], [0.0],
                                                                  frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, frenet_path_s) 
                if obj_s[0] > current_s and abs(obj_l[0]) <= 3 and obj_s[0] - current_s < 50:
    
                    distance = obj_s[0] - current_s
                    speed = obj_s_dot[0]
                     
                    if distance < min_lead_distance:
                        min_lead_distance = distance
                        closest_lead_car = (obj_s[0], speed)
                        self._last_lead_detection_time = current_time
                        self._last_lead_speed = speed

        lead_s, lead_speed = None, None
        if closest_lead_car:
            lead_s, lead_speed = closest_lead_car
            
             
        num_of_time_samples = int(horizon_time.time_s / sampling_time.time_s) + 1
        time_samples = [i * sampling_time.time_s for i in range(num_of_time_samples)] 
        max_s = max(frenet_path_s) if frenet_path_s else s_set[0] + max_velocity * horizon_time.time_s
        end_conditions = []

        for time in time_samples[1:]:
            if time <= 0:
                continue 
            
            if lead_s is not None and lead_speed is not None:
                safety_distance = self.calculate_safety_distance(current_speed, lead_speed)
                target_distance = max(self.min_safe_distance, safety_distance * 0.9)

                predicted_lead_s = lead_s + lead_speed * time

                s_target = predicted_lead_s - target_distance
                s_target = max(current_s, min(s_target, predicted_lead_s - 1.0))

                v_target = max(0.1, min(lead_speed * 0.95, max_velocity))
                speed_diff = v_target - current_speed

                time_to_target = max(0.1, min(5.0, abs(speed_diff)/self.max_accel))  # Adaptive time constant
                a_target = speed_diff / time_to_target
                a_target = max(-self.max_decel, min(a_target, self.max_accel))
                
                end_conditions.append((s_target, v_target, a_target, time, "lead_car"))
            else:
                
                if current_time - self._last_lead_detection_time < 2.0 and self._last_lead_speed > 0:

                    v_target = min(max_velocity, self._last_lead_speed)
                else:
                    v_target = min(max_velocity, current_speed + 0.5)
                
                s_target = current_s + v_target * time
                a_target = 0.0
                end_conditions.append((s_target, v_target, a_target, time, "no_lead"))
                
 
        fp_list = [] 
        for s_target, v_target, a_target, time, type in end_conditions:
            try:
                
                if s_target < current_s or time <= 0:
                    continue
                 
                coeffs = self.solve_quintic_coefficients(
                    current_s, current_speed, current_accel,
                    s_target, v_target, a_target,
                    time
                )
                

                s = coeffs[0]*time**5 + coeffs[1]*time**4 + coeffs[2]*time**3 + coeffs[3]*time**2 + coeffs[4]*time + coeffs[5]
                v = 5*coeffs[0]*time**4 + 4*coeffs[1]*time**3 + 3*coeffs[2]*time**2 + 2*coeffs[3]*time + coeffs[4]
                a = 20*coeffs[0]*time**3 + 12*coeffs[1]*time**2 + 6*coeffs[2]*time + 2*coeffs[3]       
                
 
                
                if v < -0.5 or v > max_velocity + 0.5 or a < -self.max_decel - 0.5 or a > self.max_accel + 0.5 or s < current_s or s > max_s:  # Small buffer
                    continue
                
                

                try:
                    x, y, heading, _ = frenet2cartesian(
                    [s], [0.0], [0.0], [0.0],
                    frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, frenet_path_s
                ) 
                except Exception as e:
                    logger.warning(f"Cartesian conversion failed: {e}")
                    continue
                
                fp = FrenetPath()
                fp.s.append(s)
                fp.s_d.append(v)
                fp.s_dd.append(a)
                fp.t.append(time)
                fp.x.append(x[0])
                fp.y.append(y[0])
                fp.yaw.append(heading[0])
             
                jerk = abs(a - current_accel) / time if time > 0 else 0
                s_deviation = abs(s - s_target)
                v_deviation = abs(v - v_target)

                w_jerk = 0.5
                w_accel = 1.0
                w_safety = 2.0 if type == "lead_car" else 0.5
                w_comfort = 2.0
                
                fp.cost = (w_accel * abs(a) + w_jerk * jerk + w_safety * s_deviation + w_comfort * v_deviation)

                fp_list.append(fp)

            except Exception as e:
                logger.warning(f"Path generation failed: {e}")
                continue 
        
        if not fp_list:
            logger.warning("No valid paths found")
            trajectory = []
            vehicle_params = get_pacifica_parameters()
            base_time_us = ego_state.time_point.time_us
            dt_us = int(sampling_time.time_s * 1e6)
            
            for i in range(num_of_time_samples):
                time_us = base_time_us + i * dt_us
                trajectory.append(
                    EgoState.build_from_rear_axle(
                        rear_axle_pose=ego_state.car_footprint.rear_axle,
                        rear_axle_velocity_2d=ego_state.dynamic_car_state.rear_axle_velocity_2d,
                        rear_axle_acceleration_2d=ego_state.dynamic_car_state.rear_axle_acceleration_2d,
                        tire_steering_angle=ego_state.tire_steering_angle,
                        time_point=TimePoint(time_us),
                        vehicle_parameters=vehicle_params,
                        is_in_auto_mode=True,
                        angular_vel=ego_state.dynamic_car_state.angular_velocity,
                        angular_accel=ego_state.dynamic_car_state.angular_acceleration
                    )
                )
            return trajectory
        optimal_fp = min(fp_list, key=lambda fp: fp.cost) 
        logger.info(f"type:{type}, len:{len(fp_list)}, v:{optimal_fp.s_d[0]}")
    
        trajectory = []
        vehicle_params = get_pacifica_parameters()
        base_time_us = ego_state.time_point.time_us
        dt_us = int(sampling_time.time_s * 1e6)
    
        for i in range(num_of_time_samples):
            t = i * sampling_time.time_s
            time_us = base_time_us + i * dt_us

            if t == 0:
                trajectory.append(ego_state)
                continue
            
            optimal_coeffs = self.solve_quintic_coefficients(
                    current_s, current_speed, current_accel,
                    optimal_fp.s[0], optimal_fp.s_d[0], optimal_fp.s_dd[0],
                    time
                )
            s = optimal_coeffs[0]*t**5 + optimal_coeffs[1]*t**4 + optimal_coeffs[2]*t**3 + optimal_coeffs[3]*t**2 + optimal_coeffs[4]*t + optimal_coeffs[5]
            v = 5*optimal_coeffs[0]*t**4 + 4*optimal_coeffs[1]*t**3 + 3*optimal_coeffs[2]*t**2 + 2*optimal_coeffs[3]*t + optimal_coeffs[4]
            a = 20*optimal_coeffs[0]*t**3 + 12*optimal_coeffs[1]*t**2 + 6*optimal_coeffs[2]*t + 2*optimal_coeffs[3]
        
            v = max(0.1, min(v, max_velocity))
            s = max(current_s, min(s, max_s))

            if lead_s is not None:
                predicted_lead_s = lead_s + lead_speed * t
                if s > predicted_lead_s - self.min_safe_distance:
                    s = predicted_lead_s - self.min_safe_distance
                    v = max(0.1, min(lead_speed, v))

            try:
                x, y, heading, _ = frenet2cartesian(
                [s], [0.0], [0.0], [0.0],
                frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, frenet_path_s
            )
             
                ego_state_new = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x[0], y[0], heading[0]),
                rear_axle_velocity_2d=StateVector2D(v, 0.0),
                rear_axle_acceleration_2d=StateVector2D(a, 0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(time_us),
                vehicle_parameters=vehicle_params,
                is_in_auto_mode=True,
                angular_vel=0.0,
                angular_accel=0.0
            )
                trajectory.append(ego_state_new)
            except Exception as e:
                logger.warning(f"Trajectory point generation failed: {e}")
                continue
            
        return trajectory if len(trajectory) > 1 else [ego_state] * num_of_time_samples

        
    def calculate_safety_distance(self, ego_speed: float, lead_speed: float) -> float:
        time_gap = self.time_gap
        if ego_speed > lead_speed + 1:
            time_gap = max(1.0, self.time_gap * (1 + (ego_speed - lead_speed)/5.0))
        
        braking_distance = (ego_speed**2 - lead_speed**2)/(2*self.max_decel) if ego_speed > lead_speed else 0

        return max(self.min_safe_distance, ego_speed * time_gap, braking_distance)
    
     
    def solve_quintic_coefficients(self, s0, v0, a0, s1, v1, a1, t1):
        """
        求解五阶多项式 s(t) = c1*t^5 + c2*t^4 + c3*t^3 + c4*t^2 + c5*t + c6 的系数
        输入参数：
            t1: 终止时间
            s0, v0, a0: 初始位置、速度、加速度
            s1, v1, a1: 终止位置、速度、加速度
        返回：
            系数列表 [c1, c2, c3, c4, c5, c6]
        """
        
        c6 = s0
        c5 = v0
        c4 = a0 / 2

        
        A = np.array([
            [t1**5,      t1**4,      t1**3],
            [5*t1**4,    4*t1**3,    3*t1**2],
            [20*t1**3,   12*t1**2,   6*t1]
        ])

        b = np.array([
            s1 - c4*t1**2 - c5*t1 - c6,
            v1 - 2*c4*t1 - c5,
            a1 - 2*c4
        ])
 
        c1, c2, c3 = np.linalg.solve(A, b)

        return [c1, c2, c3, c4, c5, c6]


    def solve_quartic_coefficients(s0, v0, a0, s1, v1, a1, t1):
        """
        Solve for coefficients of a quartic polynomial for s(t) given initial and final conditions.
        """
        c5 = s0
        c4 = v0
        c3 = a0 / 2
        A = np.array([[t1**4, t1**3], [4*t1**3, 3*t1**2]])
        b = np.array([s1 - c3*t1**2 - c4*t1 - c5, v1 - c4 - 2*c3*t1])
        c1, c2 = np.linalg.solve(A, b)
        return [c1, c2, c3, c4, c5]











