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

class MyPlanner01(AbstractPlanner):
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
    
        # Initial state
        x_set = [ego_state.car_footprint.center.x]
        y_set = [ego_state.car_footprint.center.y]
    
        # Convert velocity to global frame
        global_vx, global_vy = local2global_vector(
        ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
        ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
        ego_state.car_footprint.center.heading
    )
        vx_set = [global_vx]
        vy_set = [global_vy]
        ax_set = [ego_state.dynamic_car_state.rear_axle_acceleration_2d.x]
        ay_set = [ego_state.dynamic_car_state.rear_axle_acceleration_2d.y]

        # Get reference path
        frenet_path_x = reference_path_provider._x_of_reference_line
        frenet_path_y = reference_path_provider._y_of_reference_line
        frenet_path_heading = reference_path_provider._heading_of_reference_line
        frenet_path_kappa = reference_path_provider._kappa_of_reference_line
        frenet_path_s = reference_path_provider._s_of_reference_line

        # Check if reference path is valid
        if not frenet_path_x or len(frenet_path_x) < 2:
            logger.error("Invalid reference path! Returning current state.")
            return [ego_state]

        # Convert to Frenet frame
        try:
            s_set, l_set, s_dot_set, l_dot_set, dl_set, l_dot2_set, s_dot2_set, ddl_set = cartesian2frenet(
            x_set, y_set, vx_set, vy_set, ax_set, ay_set,
            frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, frenet_path_s
        )
        except Exception as e:
            logger.error(f"Frenet conversion failed: {e}. Returning current state.")
            return [ego_state]

        # Ensure positive velocity
        # 在 planning() 函数的 "# Ensure positive velocity" 后添加：
 
        # s_dot_set[0] = max(0.1, s_dot_set[0])  # Minimum velocity to prevent stalling

        # Time parameters
        num_of_time_samples = int(horizon_time.time_s / sampling_time.time_s) + 1
        time_samples = [i * sampling_time.time_s for i in range(num_of_time_samples)]
        # time_samples = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        max_s = max(frenet_path_s) if frenet_path_s else s_set[0] + max_velocity * horizon_time.time_s

        #  Generate multiple candidate end conditions
        end_conditions = []

        closest_lead_car = None
        min_lead_distance = float('inf')
        # logger.info(f"objects size:{len(objects)}")
        for obj in objects:
            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                obj_s, obj_l, _, _, _, _, _, _ = cartesian2frenet([obj.box.center.x], [obj.box.center.y], [obj.velocity.x], [obj.velocity.y], [0.0], [0.0],
                frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, frenet_path_s)
                # logger.info(f"obj_s = {obj_s}, obj_l = {obj_l}")
                if obj_s[0] >= 0 and s_set[0] >= 0 and abs(obj_l[0]) <= 3:
                    # logger.info(f"condition1 obj s = {obj_s[0]}, s_set[0] = {s_set[0]}")
                    distance = obj_s[0] - s_set[0]
                    speed = np.hypot(obj.velocity.x, obj.velocity.y)
                    
                    # logger.info(f"distance = {distance}, min_lead_distance = {min_lead_distance}")
                    if 0 < distance <= 20 and distance < min_lead_distance:
                        min_lead_distance = distance
                        closest_lead_car = (obj_s[0], speed)
                        # logger.info(f"there lead car")
                     
        for time in time_samples[1:]:  # Skip t=0
            # logger.info(f"Generating candidate end condition at t={time}")
            if time <= 0:
                continue 
            if closest_lead_car:
                # logger.info(f"closest_lead_car:{closest_lead_car}")
                # 跟车逻辑：前车的s和速度作为目标
                lead_s, lead_speed = closest_lead_car 
                logger.info(f"lead_speed:{lead_speed}")
                v_target = max(0.1, lead_speed * 0.9)
                time_gap = 2.0
                safety_distance = max(3.0, v_target * time_gap)
                s_target = lead_s - safety_distance  # 保持安全距离
                end_conditions.append(([s_target, v_target * 0.1, 0.0], time, "lead_car"))
            else:
                #Accelerate case
                # logger.info(f"no closest_lead_car:{closest_lead_car}")
                s_acc = s_set[0] + s_dot_set[0] * time + 0.5 * 2.0 * time**2
                v_acc = min(s_dot_set[0] + 2.0 * time, max_velocity)
                end_conditions.append(([s_acc, v_acc, 0.0], time, "no_lead_car"))
            # Base case: maintain current speed
            # s_maintain = s_set[0] + s_dot_set[0] * time
            # end_conditions.append(([s_maintain, s_dot_set[0], 0.0], time))
     
        
                # Decelerate case
                s_dec = s_set[0] + s_dot_set[0] * time - 0.5 * 2.0 * time**2
                v_dec = max(0.1, s_dot_set[0] - 2.0 * time)  # Don't stop completely
                end_conditions.append(([s_dec, v_dec, 0.0], time, "no_lead_car"))

        # Generate and evaluate Frenet paths
        fp_list = []
        logger.info(f"end_conditions:{len(end_conditions)}")
        for end_condition, time, type in end_conditions:
            try:
                # Skip invalid conditions
                # logger.info(f"type:{type}")
                logger.info(f"end_condition[0]:{end_condition[0]}, s_set[0]:{s_set[0]}")
                if end_condition[0] < s_set[0] or time <= 0:
                    continue
                
                # Calculate polynomial coefficients
                logger.info(f"end_condition[0]:{end_condition[0]}, end_condition[1]:{end_condition[1]}, time:{time}")
                c1, c2, c3, c4, c5 = solve_quartic_coefficients(
                s_set[0], s_dot_set[0], s_dot2_set[0],
                end_condition[0], end_condition[1], end_condition[2], time
            )
            
                # Evaluate at final time
                s = c1 * time**4 + c2 * time**3 + c3 * time**2 + c4 * time + c5
                v = 4 * c1 * time**3 + 3 * c2 * time**2 + 2 * c3 * time + c4
                a = 12 * c1 * time**2 + 6 * c2 * time + 2 * c3
                logger.info(f"solve_s:{s}, v:{v}")
            
                # Skip invalid paths
 
                logger.info(f"s:{s}, s_set[0]:{s_set[0]}")
                if s < s_set[0] - 1.0 or v < 0 or s > max_s + 10:  # Small buffer
                    continue
                
                # Create Frenet path
                fp = FrenetPath()
                fp.s.append(s)
                fp.s_d.append(v)
                fp.s_dd.append(a)
                fp.t.append(time)
            
                # Convert to Cartesian (lateral offset = 0 for simplicity)
                try:
                    x, y, heading, _ = frenet2cartesian(
                    [s], [0.0], [0.0], [0.0],
                    frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, frenet_path_s
                )
                    fp.x.append(x[0])
                    fp.y.append(y[0])
                    fp.yaw.append(heading[0])
                except Exception as e:
                    logger.warning(f"Cartesian conversion failed: {e}")
                    continue
                
                # Cost function (prioritize smooth, progressive motion)
                fp.cost = abs(a) + 0.1 * abs(v - max_velocity) + 0.01 * abs(s - end_condition[0])
                if (type == "lead_car"):
                    logger.info(f"lead_car fp_cost:{fp.cost}, type:{type}")
                else:
                    logger.info(f"no_lead_car fp_cost:{fp.cost}, type:{type}")
                fp_list.append(fp)
            except Exception as e:
                logger.warning(f"Path generation failed: {e}")
                continue 
        
        if not fp_list:
            logger.warning("No valid paths found")
        # 创建具有正确时间戳的轨迹
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
    # Select best path 
        optimal_fp = min(fp_list, key=lambda fp: fp.cost) 
        logger.info(f"type:{type}, len:{len(fp_list)}, v:{optimal_fp.s_d[0]}")
    
    # Generate full trajectory
        trajectory = []
        vehicle_params = get_pacifica_parameters()
        base_time_us = ego_state.time_point.time_us
        dt_us = int(sampling_time.time_s * 1e6)
    
        for i in range(num_of_time_samples):
            t = i * sampling_time.time_s
            if t == 0:
                trajectory.append(ego_state)
                continue
            
        # Interpolate along optimal path
            c1, c2, c3, c4, c5 = solve_quartic_coefficients(
                s_set[0], s_dot_set[0], s_dot2_set[0],
                optimal_fp.s[0], optimal_fp.s_d[0], optimal_fp.s_dd[0], optimal_fp.t[0]
        )
        
            s = c1 * t**4 + c2 * t**3 + c3 * t**2 + c4 * t + c5
            v = 4 * c1 * t**3 + 3 * c2 * t**2 + 2 * c3 * t + c4
            a = 12 * c1 * t**2 + 6 * c2 * t + 2 * c3
        
        # Convert to Cartesian
            try:
                x, y, heading, _ = frenet2cartesian(
                [s], [0.0], [0.0], [0.0],
                frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, frenet_path_s
            )
            
                time_us = base_time_us + i * dt_us
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









