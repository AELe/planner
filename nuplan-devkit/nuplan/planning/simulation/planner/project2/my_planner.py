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
    Enhanced car-following planner with improved stability and safety
    """

    def __init__(
            self,
            horizon_seconds: float,
            sampling_time: float,
            max_velocity: float = 5.0,
    ):
        """
        Constructor for EnhancedPlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param max_velocity: [m/s] ego max velocity.
        """
        self.horizon_time = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.max_velocity = max_velocity
        
        # Controller parameters
        self.min_safe_distance = 10.0  # Absolute minimum distance (m)
        self.time_gap = 1.8  # Desired time gap (s)
        self.max_accel = 2.0  # m/s^2
        self.max_decel = 3.0  # m/s^2
        self.smoothing_alpha = 0.5  # Smoothing factor for lead car state
        
        # State variables
        self._router: Optional[BFSRouter] = None
        self._predictor: AbstractPredictor = None
        self._reference_path_provider: Optional[ReferenceLineProvider] = None
        self._routing_complete = False
        self._prev_lead_s = None
        self._prev_lead_speed = None
        self._prev_s = None
        self._last_lead_detection_time = 0
        self._last_lead_speed = 0 

        logger.info(f"Planner initialized with max_velocity={max_velocity}")

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._router = BFSRouter(initialization.map_api)
        self._router._initialize_route_plan(initialization.route_roadblock_ids)

    def name(self) -> str:
        """Inherited, see superclass."""
        return "EnhancedCarFollowingPlanner"

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Main planning function
        """
        ego_state, observations = current_input.history.current_state
        
        # 1. Routing
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
        trajectory = self.planning(ego_state, self._reference_path_provider, objects,
                                 self.horizon_time, self.sampling_time, self.max_velocity)

        return InterpolatedTrajectory(trajectory)

    def planning(self,
                 ego_state: EgoState,
                 reference_path_provider: ReferenceLineProvider,
                 objects: List[TrackedObjects],
                 horizon_time: TimePoint,
                 sampling_time: TimePoint,
                 max_velocity: float) -> List[EgoState]:
        
        x = ego_state.car_footprint.center.x
        y = ego_state.car_footprint.center.y
        vx = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        vy = ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        global_vx, global_vy = local2global_vector(
        ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
        ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
        ego_state.car_footprint.center.heading
    )
        ax = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        ay = ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
        global_ax, global_ay = local2global_vector(
        ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
        ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
        ego_state.car_footprint.center.heading
    )

        frenet_path_x = reference_path_provider._x_of_reference_line
        frenet_path_y = reference_path_provider._y_of_reference_line
        frenet_path_heading = reference_path_provider._heading_of_reference_line
        frenet_path_kappa = reference_path_provider._kappa_of_reference_line
        frenet_path_s = reference_path_provider._s_of_reference_line

        if not frenet_path_x or len(frenet_path_x) < 2:
            return [ego_state]

        try:
            s_set, l_set, s_dot_set, l_dot_set, dl_set, l_dot2_set, s_dot2_set, ddl_set = cartesian2frenet(
                [x], [y], [global_vx], [global_vy], [global_ax], [global_ay],
                frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, frenet_path_s
            )
            current_s = s_set[0]
            current_speed = s_dot_set[0]
            current_accel = s_dot2_set[0]
            logger.info(f"current_speed:{current_speed}, current_accel:{current_accel}")

        except Exception as e:
            logger.error(f"Frenet conversion failed: {e}")
            return [ego_state]

        closest_lead_car = None
        min_lead_distance = float('inf')
        current_time = ego_state.time_point.time_s
        
        for obj in objects:
            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                try:
                    obj_s, obj_l, obj_s_dot, _, _, _, _, _ = cartesian2frenet(
                        [obj.box.center.x], [obj.box.center.y], 
                        [obj.velocity.x], [obj.velocity.y], 
                        [0.0], [0.0],
                        frenet_path_x, frenet_path_y, 
                        frenet_path_heading, frenet_path_kappa, 
                        frenet_path_s
                    )
                    

                    if (obj_s[0] > current_s and
                        abs(obj_l[0]) <= 3.0 and
                        obj_s[0] - current_s < 50.0):
                        
                        distance = obj_s[0] - current_s
                        speed = obj_s_dot[0]
                        
                        if distance < min_lead_distance:
                            min_lead_distance = distance
                            closest_lead_car = (obj_s[0], speed)
                            self._last_lead_detection_time = current_time
                            self._last_lead_speed = speed
                            
                except Exception as e:
                    logger.warning(f"Frenet conversion for object failed: {e}")
                    continue

        lead_s, lead_speed = None, None
        if closest_lead_car:
            if not isinstance(closest_lead_car[0], (int, float)) or not isinstance(closest_lead_car[1], (int, float)):
                logger.warning(f"Invalid lead car data: {closest_lead_car}")
            else:
                # rel_speed_diff = abs(closest_lead_car[1] - current_speed)
                # alpha = max(0.3, min(0.7, 0.5 - rel_speed_diff/10.0))
                
                # if self._prev_lead_s is not None and self._prev_lead_speed is not None:
                #     lead_s = alpha * self._prev_lead_s + (1 - alpha) * closest_lead_car[0]
                #     lead_speed = alpha * self._prev_lead_speed + (1 - alpha) * closest_lead_car[1]
                # else:
                #     lead_s, lead_speed = closest_lead_car
                lead_s, lead_speed = closest_lead_car
                self._prev_lead_s = lead_s
                self._prev_lead_speed = lead_speed 
        
        num_of_time_samples = int(horizon_time.time_s / sampling_time.time_s) + 1
        time_samples = [i * sampling_time.time_s for i in range(num_of_time_samples)]
        max_s = max(frenet_path_s) 

        target_states = []
        for time in time_samples[1:]:
            if time <= 0:
                continue
                
            if lead_s is not None and lead_speed is not None:

                safety_distance = self.calculate_safety_distance(current_speed, lead_speed)
                target_distance = max(self.min_safe_distance, safety_distance * 0.9)  # Slightly more aggressive
                
                predicted_lead_s = lead_s + lead_speed * time
                
            
                s_target = predicted_lead_s - target_distance
                s_target = max(current_s, min(s_target, predicted_lead_s - 1.0))  # Never exceed lead car
                
                
                v_target = max(0.1, min(lead_speed * 0.95, max_velocity))
                
                speed_diff = v_target - current_speed
                time_to_target = max(0.1, min(5.0, abs(speed_diff)/self.max_accel))  # Adaptive time constant
                a_target = speed_diff / time_to_target
                a_target = max(-self.max_decel, min(a_target, self.max_accel))
                
                target_states.append((s_target, v_target, a_target, time, "lead_car"))
            else:
                
                if current_time - self._last_lead_detection_time < 2.0 and self._last_lead_speed > 0:

                    v_target = min(max_velocity, self._last_lead_speed)
                else:
                    v_target = min(max_velocity, current_speed + 0.5)
                
                s_target = current_s + v_target * time
                a_target = 0.0
                target_states.append((s_target, v_target, a_target, time, "no_lead"))

        # Generate and evaluate Frenet paths
        fp_list = []
        for s_target, v_target, a_target, time, condition_type in target_states:
            if s_target < current_s or time <= 0:
                continue

            try:
                # Use quintic polynomial for smoother trajectories
                coeffs = self.solve_quintic_coefficients(
                    current_s, current_speed, current_accel,
                    s_target, v_target, a_target,
                    time
                )
                
                # Evaluate trajectory at this time point
                s = coeffs[0]*time**5 + coeffs[1]*time**4 + coeffs[2]*time**3 + coeffs[3]*time**2 + coeffs[4]*time + coeffs[5]
                v = 5*coeffs[0]*time**4 + 4*coeffs[1]*time**3 + 3*coeffs[2]*time**2 + 2*coeffs[3]*time + coeffs[4]
                a = 20*coeffs[0]*time**3 + 12*coeffs[1]*time**2 + 6*coeffs[2]*time + 2*coeffs[3]

                # Validate trajectory
                if (v < -0.5 or v > max_velocity + 0.5 or 
                    a < -self.max_decel - 0.5 or a > self.max_accel + 0.5 or
                    s < current_s or s > max_s):
                    continue

                # Convert to Cartesian coordinates
                try:
                    x, y, heading, _ = frenet2cartesian(
                        [s], [0.0], [0.0], [0.0],
                        frenet_path_x, frenet_path_y, 
                        frenet_path_heading, frenet_path_kappa, 
                        frenet_path_s
                    )
                except Exception as e:
                    logger.warning(f"Cartesian conversion failed: {e}")
                    continue

                # Create Frenet path point
                fp = FrenetPath()
                fp.s.append(s)
                fp.s_d.append(v)
                fp.s_dd.append(a)
                fp.t.append(time)
                fp.x.append(x[0])
                fp.y.append(y[0])
                fp.yaw.append(heading[0])

                # Cost calculation with multiple factors
                jerk = abs(a - current_accel) / time if time > 0 else 0.0
                s_deviation = abs(s - s_target)
                v_deviation = abs(v - v_target)
                
                # Cost weights
                w_jerk = 0.5
                w_accel = 1.0
                w_safety = 2.0 if condition_type == "lead_car" else 0.5
                w_comfort = 0.2
                
                fp.cost = (w_accel * abs(a) + 
                          w_jerk * jerk + 
                          w_safety * s_deviation + 
                          w_comfort * v_deviation)
                
                fp_list.append(fp)
                
            except Exception as e:
                logger.warning(f"Path generation failed: {e}")
                continue

        # Select best trajectory
        if fp_list:
            optimal_fp = min(fp_list, key=lambda fp: fp.cost)
            self._prev_s = optimal_fp.s[0]
        else:
            # Fallback trajectory
            logger.warning("No valid paths found, using fallback trajectory")
            return self.generate_fallback_trajectory(
                ego_state, current_s, current_speed, 
                lead_s, lead_speed, frenet_path_x, frenet_path_y,
                frenet_path_heading, frenet_path_kappa, frenet_path_s,
                num_of_time_samples, sampling_time
            )

        # Generate final trajectory
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

            # Calculate state at time t
            s = coeffs[0]*t**5 + coeffs[1]*t**4 + coeffs[2]*t**3 + coeffs[3]*t**2 + coeffs[4]*t + coeffs[5]
            v = 5*coeffs[0]*t**4 + 4*coeffs[1]*t**3 + 3*coeffs[2]*t**2 + 2*coeffs[3]*t + coeffs[4]
            a = 20*coeffs[0]*t**3 + 12*coeffs[1]*t**2 + 6*coeffs[2]*t + 2*coeffs[3]

            # Apply limits
            v = max(0.1, min(v, max_velocity))
            s = max(current_s, min(s, max_s))
            
            # Additional safety check for lead car
            if lead_s is not None:
                predicted_lead_s = lead_s + lead_speed * t
                if s > predicted_lead_s - self.min_safe_distance:
                    s = predicted_lead_s - self.min_safe_distance
                    v = max(0.1, min(lead_speed, v))

            # Convert to Cartesian
            try:
                x, y, heading, _ = frenet2cartesian(
                    [s], [0.0], [0.0], [0.0],
                    frenet_path_x, frenet_path_y, 
                    frenet_path_heading, frenet_path_kappa, 
                    frenet_path_s
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
                trajectory.append(ego_state)  # Fallback to current state

        return trajectory if len(trajectory) > 1 else [ego_state] * num_of_time_samples 

    def generate_fallback_trajectory(self,
                                   ego_state: EgoState,
                                   current_s: float,
                                   current_speed: float,
                                   lead_s: Optional[float],
                                   lead_speed: Optional[float],
                                   frenet_path_x: List[float],
                                   frenet_path_y: List[float],
                                   frenet_path_heading: List[float],
                                   frenet_path_kappa: List[float],
                                   frenet_path_s: List[float],
                                   num_of_time_samples: int,
                                   sampling_time: TimePoint) -> List[EgoState]:
        """
        Generate a safe fallback trajectory when no valid paths are found
        """
        vehicle_params = get_pacifica_parameters()
        base_time_us = ego_state.time_point.time_us
        dt_us = int(sampling_time.time_s * 1e6)
        trajectory = []
        
        # Determine target speed
        if lead_s is not None and lead_speed is not None:
            # If we have lead car info but no valid path, be conservative
            safety_distance = self.calculate_safety_distance(current_speed, lead_speed)
            target_speed = max(0.1, min(lead_speed * 0.8, self.max_velocity))
            max_s = lead_s - safety_distance
        else:
            # No lead car - gently accelerate to cruising speed
            target_speed = min(self.max_velocity, current_speed + 0.2)
            max_s = current_s + self.max_velocity * self.horizon_time.time_s
        
        for i in range(num_of_time_samples):
            t = i * sampling_time.time_s
            time_us = base_time_us + i * dt_us
            
            # Simple linear interpolation to target speed
            alpha = min(t / 1.0, 1.0)  # Ramp over 1 second
            v = (1 - alpha) * current_speed + alpha * target_speed
            s = current_s + v * t
            s = max(current_s, min(s, max_s))
            a = 0.0  # Zero acceleration in fallback
            
            try:
                x, y, heading, _ = frenet2cartesian(
                    [s], [0.0], [0.0], [0.0],
                    frenet_path_x, frenet_path_y, 
                    frenet_path_heading, frenet_path_kappa, 
                    frenet_path_s
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
                logger.warning(f"Fallback trajectory point failed: {e}")
                trajectory.append(ego_state)
        
        return trajectory

    def solve_quintic_coefficients(self,
                                  s0: float,
                                  v0: float,
                                  a0: float,
                                  s1: float,
                                  v1: float,
                                  a1: float,
                                  t1: float) -> List[float]:
        """
        Solve for coefficients of a quintic polynomial for s(t) given initial and final conditions.
        Returns coefficients [a, b, c, d, e, f] for s(t) = a*t^5 + b*t^4 + c*t^3 + d*t^2 + e*t + f
        """
        # Set up the system of equations
        A = np.array([
            [t1**5, t1**4, t1**3],
            [5*t1**4, 4*t1**3, 3*t1**2],
            [20*t1**3, 12*t1**2, 6*t1]
        ])
        
        b = np.array([
            s1 - (a0/2)*t1**2 - v0*t1 - s0,
            v1 - a0*t1 - v0,
            a1 - a0
        ])
        
        # Solve for the highest order coefficients
        try:
            x = np.linalg.solve(A, b)
            a, b, c = x[0], x[1], x[2]
        except np.linalg.LinAlgError:
            # Fallback to cubic if quintic fails
            logger.warning("Quintic solve failed, using cubic fallback")
            return self.solve_cubic_coefficients(s0, v0, a0, s1, v1, a1, t1)
        
        return [a, b, c, a0/2, v0, s0]

    def solve_cubic_coefficients(self,
                                s0: float,
                                v0: float,
                                a0: float,
                                s1: float,
                                v1: float,
                                a1: float,
                                t1: float) -> List[float]:
        """
        Fallback cubic polynomial solver
        Returns coefficients [a, b, c, d] for s(t) = a*t^3 + b*t^2 + c*t + d
        """
        # We can't satisfy all boundary conditions with cubic, so we prioritize position and velocity
        A = np.array([
            [t1**3, t1**2],
            [3*t1**2, 2*t1]
        ])
        
        b = np.array([
            s1 - (a0/2)*t1**2 - v0*t1 - s0,
            v1 - a0*t1 - v0
        ])
        
        try:
            x = np.linalg.solve(A, b)
            a, b = x[0], x[1]
            return [a, b, a0/2, v0, s0]  # Still return 5 coefficients for consistency
        except np.linalg.LinAlgError:
            # Ultimate fallback - linear interpolation
            logger.warning("Cubic solve failed, using linear fallback")
            avg_speed = (s1 - s0) / t1
            return [0, 0, 0, 0, avg_speed, s0]

    def calculate_safety_distance(self, ego_speed: float, lead_speed: float) -> float:
        """
        Calculate dynamic safety distance based on current speeds
        """
        time_gap = self.time_gap
        if ego_speed > lead_speed + 1.0:
            time_gap = max(1.0, self.time_gap * (1 + (ego_speed - lead_speed)/5.0))
        
        braking_distance = (ego_speed**2 - lead_speed**2)/(2*self.max_decel) if ego_speed > lead_speed else 0
        return max(self.min_safe_distance, ego_speed * time_gap, braking_distance)






