import numpy as np
import random
from enum import Enum
import math

class Strut:
    # Class for strut to track endpoints/connections and for agents to be able to see 
    def __init__(self, start, end, thickness, strut_id=None, occupied=False):
        self.start = np.array(start)
        self.end = np.array(end)
        self.thickness = thickness
        self.id = strut_id if strut_id is not None else id(self)
        self.occupied = occupied  # need tell agents not to join a strut that is actively being built
        
        # Track connected struts at each endpoint
        self.start_connections = set()  #for now this is an empty tuple
        self.end_connections = set()    
    
    def get_length(self):
        return np.linalg.norm(self.end - self.start)
    
    def get_angle(self):
        direction = self.end - self.start
        length = np.linalg.norm(direction)
        tan = direction / length if length > 0 else np.array([1, 0])
        return math.atan2(tan[1], tan[0])
    
    #for adding connections to ensure agents dont build off of nodes that already have 2 connections
    def add_connection(self, other_strut, connection_point):
        # Determine which endpoint the connection is at
        start_dist = np.linalg.norm(np.array(connection_point) - self.start)
        end_dist = np.linalg.norm(np.array(connection_point) - self.end)
        
        if start_dist < end_dist:
            self.start_connections.add(other_strut.id)
            other_start_dist = np.linalg.norm(np.array(connection_point) - other_strut.start)
            other_end_dist = np.linalg.norm(np.array(connection_point) - other_strut.end)
            if other_start_dist < other_end_dist:
                other_strut.start_connections.add(self.id)
            else:
                other_strut.end_connections.add(self.id)

        else:
            self.end_connections.add(other_strut.id)

            # Also add to the other strut
            other_start_dist = np.linalg.norm(np.array(connection_point) - other_strut.start)
            other_end_dist = np.linalg.norm(np.array(connection_point) - other_strut.end)
            if other_start_dist < other_end_dist:
                other_strut.start_connections.add(self.id)
            else:
                other_strut.end_connections.add(self.id)
    
    #For checking the number of connections at a given end-point
    def get_connections_at_point(self, point, tolerance=0.5):
        #again, check which endpoint you're at first
        start_dist = np.linalg.norm(np.array(point) - self.start)
        end_dist = np.linalg.norm(np.array(point) - self.end)
        
        if start_dist <= tolerance:
            return len(self.start_connections)
        elif end_dist <= tolerance:
            return len(self.end_connections)
        else:
            return 0
    
    #Function that tells which struts are connected to which for path following and cbecking completion
    def get_connected_struts_at_point(self, point, tolerance=0.5):
        start_dist = np.linalg.norm(np.array(point) - self.start)
        end_dist = np.linalg.norm(np.array(point) - self.end)
        
        if start_dist <= tolerance:
            return list(self.start_connections)
        elif end_dist <= tolerance:
            return list(self.end_connections)
        else:
            return []

    def set_occupied(self, occupied):
        self.occupied = occupied
    
    def is_occupied(self):
        return self.occupied

#Define states in state machine
class AgentState(Enum):
    SEARCHING = "searching"
    MOVING_TO_STRUT = "moving_to_strut"
    FOLLOWING_STRUT = "following_strut"
    DEPOSITING = "depositing"
    HOMING = "homing"
    GATHERING = "gathering"

class Agent:
    def __init__(self, agent_id, entry_position, construction_bounds, verbose=False, strut_target_length=4, strut_errors=0, deposition_speed=0.1):
        # Agent identification
        self.id = agent_id
        
        # Position and movement characteristics
        self.position = np.array(entry_position)
        self.direction = random.uniform(0, 2 * math.pi)  # Random initial direction
        self.state = AgentState.SEARCHING #obviously it should start searching
        
        # Construction space bounds(can be thought of as how far an agent goes before having to turn back due to signal strength of something)
        self.construction_bounds = construction_bounds
        
        # Agent parameters which can be played around with a bit
        self.body_size = 0.1  
        self.view_distance = 8.0  
        self.view_angle = math.radians(45) 
        self.speed_default = 0.3  
        self.speed_depositing = deposition_speed
        self.speed_leaving = 0.5  
        self.speed_homing = 0.4  
        
        # Center position for homing and gathering
        center_x = (construction_bounds[0] + construction_bounds[2]) / 2
        center_y = (construction_bounds[1] + construction_bounds[3]) / 2
        self.center_position = np.array([center_x, center_y])
        
        #radius for gathering area
        self.gathering_radius = 2.0 
        #index for how much time agent has spent gathering
        self.gathering_time = 0 
        # Steps required to gather material
        self.gathering_duration = 30  
        
        # Bifurcation parameters 
        self.target_turning_angle = math.radians(60)
        self.target_strut_length = strut_target_length 
        self.min_strut_length = strut_target_length - strut_errors 
        self.max_strut_length = strut_target_length + strut_errors 
        self.strut_length_std = 0.3  
        
        # State-specific variables
        self.target_strut = None
        self.current_strut = None
        self.depositing_strut_start = None
        self.deposited_length = 0.0
        self.target_deposition_length = 0.0
        self.following_direction = 1  # 1 or -1 for strut following direction
        self.completed_strut = None  # Store the completed strut for retrieval
        
        self.verbose = verbose 

    def update(self, struts):
        #first check to see if the robot is out of bounds
        self.check_bounds()
        
        # For detecting transitions to release the strut
        previous_state = self.state
        
        if self.state == AgentState.SEARCHING:
            self.search_behavior(struts)
        elif self.state == AgentState.MOVING_TO_STRUT:
            self.move_to_strut_behavior()
        elif self.state == AgentState.FOLLOWING_STRUT:
            self.follow_strut_behavior(struts)
        elif self.state == AgentState.DEPOSITING:
            self.depositing_behavior(struts)
        elif self.state == AgentState.HOMING:
            self.homing_behavior()
        elif self.state == AgentState.GATHERING:
            self.gathering_behavior()
        
        # Release the previous strut
        if (previous_state == AgentState.FOLLOWING_STRUT and self.state != AgentState.FOLLOWING_STRUT and self.state != AgentState.DEPOSITING and self.current_strut):
            self.current_strut.set_occupied(False)
            self.current_strut = None
    
    def search_behavior(self, struts):
        #Determines each timestep where the robot is searching
        detected_struts = self.sense_environment(struts)
        
        # Filter out occupied struts
        available_struts = [s for s in detected_struts if not s.is_occupied()]
        
        if available_struts:
            # Move toward the closest available unoccupied strut
            closest_strut = min(available_struts, key=lambda s: self.distance_to_strut(s))
            self.target_strut = closest_strut
            self.state = AgentState.MOVING_TO_STRUT

            # Set direction toward strut
            strut_center = (np.array(closest_strut.start) + np.array(closest_strut.end)) / 2
            direction_to_strut = strut_center - self.position
            self.direction = math.atan2(direction_to_strut[1], direction_to_strut[0])
            if self.verbose: 
                print(f"Agent {self.id} detected available strut, moving toward it")
        else:
            # Improved random walk w/ bias toward center of construction space
            center_x = (self.construction_bounds[0] + self.construction_bounds[2]) / 2
            center_y = (self.construction_bounds[1] + self.construction_bounds[3]) / 2
            center = np.array([center_x, center_y])
            
            # If far from center, bias movement toward center. Accelerates the early stages of the searching process
            # Helps if agents get far from the center at the start to bring them back in
            distance_to_center = np.linalg.norm(self.position - center)
            if distance_to_center > 15: 
                direction_to_center = center - self.position
                center_angle = math.atan2(direction_to_center[1], direction_to_center[0])
                self.direction = 0.7 * center_angle + 0.3 * self.direction
            
            # Add some random variation
            self.direction += random.uniform(-0.3, 0.3) 
            self.move()
    
    def move_to_strut_behavior(self):
        #Move toward the target strut
        self.move() 
        
        # Check if reached strut
        if self.distance_to_strut(self.target_strut) < self.body_size * 2:
            if self.verbose:
                print(f"Agent {self.id} reached strut, starting to follow")

            # Start following the strut and mark it as occupied
            self.current_strut = self.target_strut
            self.current_strut.set_occupied(True) 
            self.target_strut = None 
            self.following_direction = random.choice([-1, 1])
            self.state = AgentState.FOLLOWING_STRUT
            
            # Move to nearest endpoint
            start_pos = np.array(self.current_strut.start)
            end_pos = np.array(self.current_strut.end)
            
            # Find which endpoint is closer to agent's current position
            distance_to_start = np.linalg.norm(self.position - start_pos)
            distance_to_end = np.linalg.norm(self.position - end_pos)
            
            # Move to the closest endpoint
            if distance_to_start < distance_to_end:
                # Move to start endpoint
                self.position = start_pos.copy()
                self.following_direction = 1  
                direction_vector = end_pos - start_pos
            else:
                #move toward starting point if you're closer
                self.position = end_pos.copy()
                self.following_direction = -1  
                direction_vector = start_pos - end_pos
                
            # Set direction along strut toward the other endpoint
            if np.linalg.norm(direction_vector) > 0:
                self.direction = math.atan2(direction_vector[1], direction_vector[0])
    
    def follow_strut_behavior(self, struts):
        #just follow the strut
        if self.current_strut is None:
            self.state = AgentState.SEARCHING
            return
        
        # Move along strut
        self.move()
        
        # Check if reached end of strut

        #first check the beginning and end of the strut
        start_pos = np.array(self.current_strut.start)
        end_pos = np.array(self.current_strut.end)
        
        # Determine which end we're moving toward
        if self.following_direction == 1:
            target_end = end_pos
        else:
            target_end = start_pos
        
        if np.linalg.norm(self.position - target_end) < self.body_size*2:
            if self.verbose: 
                print(f"Agent {self.id} reached end of strut, checking connections")
            
            # Check how many struts are already connected at this endpoint
            connected_struts = self.get_struts_connected_at_point(target_end, struts)
            num_connections = len(connected_struts)
            
            if self.verbose:
                print(f"Agent {self.id} found {num_connections} connections at endpoint")
            
            if num_connections >= 2:
                # Two struts already connected - pick one randomly and follow it
                self.follow_random_connected_strut(connected_struts, target_end)
            elif num_connections == 0:
                # No struts connected - create new strut at 60° or -60°
                self.create_new_strut_at_angle(target_end)
            else:  
                # One strut connected - create second strut to complete junction
                self.create_complementary_strut(connected_struts[0], target_end)
    
    def get_struts_connected_at_point(self, point, struts):
        connected = []
        tolerance = 0.5 
        
        for strut in struts:
            #skip the current strut
            if strut.id == self.current_strut.id:
                continue  
            
            # Skip occupied struts
            if strut.is_occupied():
                continue
            
            # Check if strut has an endpoint near the target point
            start_dist = np.linalg.norm(np.array(strut.start) - np.array(point))
            end_dist = np.linalg.norm(np.array(strut.end) - np.array(point))
            
            if start_dist <= tolerance or end_dist <= tolerance:
                connected.append(strut)
        
        return connected
    
    def follow_random_connected_strut(self, connected_struts, junction_point):
        # Filter out occupied struts
        available_struts = [s for s in connected_struts if not s.is_occupied()]
        
        if not available_struts:
            # No available struts to follow, start creating a new one
            if self.verbose:
                print(f"Agent {self.id} found no available struts at junction, creating new one")
            self.create_new_strut_at_angle(junction_point)
            return
        
        selected_strut = random.choice(available_struts)
        
        # Release current strut before taking new one
        if self.current_strut:
            self.current_strut.set_occupied(False)
        
        self.current_strut = selected_strut
        self.current_strut.set_occupied(True) 
        
        # Determine which direction to follow the new strut
        start_dist = np.linalg.norm(np.array(selected_strut.start) - np.array(junction_point))
        end_dist = np.linalg.norm(np.array(selected_strut.end) - np.array(junction_point))
        
        if start_dist < end_dist:
            # follow toward end
            self.following_direction = 1
            self.position = np.array(selected_strut.start).copy()
            direction_vector = np.array(selected_strut.end) - np.array(selected_strut.start)
        else:
            # follow toward start
            self.following_direction = -1
            self.position = np.array(selected_strut.end).copy()
            direction_vector = np.array(selected_strut.start) - np.array(selected_strut.end)
        
        # Set direction along the selected strut
        if np.linalg.norm(direction_vector) > 0:
            self.direction = math.atan2(direction_vector[1], direction_vector[0])
        
        if self.verbose:
            print(f"Agent {self.id} following connected strut {selected_strut.id}")
    
    def create_new_strut_at_angle(self, start_point):
        # Get the direction the agent was following along the current strut
        start_pos = np.array(self.current_strut.start)
        end_pos = np.array(self.current_strut.end)
        
        # Calculate the direction vector based on how the agent was following the strut
        if self.following_direction == 1:
            strut_vector = end_pos - start_pos
        else:
            strut_vector = start_pos - end_pos
        
        # Get exact base angle
        if np.linalg.norm(strut_vector) > 0:
            current_strut_angle = math.atan2(strut_vector[1], strut_vector[0])
        else:
            current_strut_angle = self.direction
        
        # Snap base angle to nearest hexagonal angle
        current_strut_angle = self.snap_to_hexagonal_angle(current_strut_angle)
        
        # Choose EXACTLY 60° or -60° turn
        turn_angle = random.choice([math.radians(60), math.radians(-60)])
        new_direction = current_strut_angle + turn_angle
        
        # Snap final direction to hexagonal grid
        self.direction = self.snap_to_hexagonal_angle(new_direction)
        
        if self.verbose:
            print(f"Agent {self.id} creating new strut at EXACT {math.degrees(turn_angle):.0f}° turn "
              f"(base: {math.degrees(current_strut_angle):.0f}°, "
              f"new: {math.degrees(self.direction):.0f}°)")
        self.start_deposition(start_point)

    def create_complementary_strut(self, existing_strut, junction_point):
        # Get exact current strut direction
        start_pos = np.array(self.current_strut.start)
        end_pos = np.array(self.current_strut.end)
        
        if self.following_direction == 1:
            current_strut_angle = math.atan2((end_pos - start_pos)[1], (end_pos - start_pos)[0])
        else:
            current_strut_angle = math.atan2((start_pos - end_pos)[1], (start_pos - end_pos)[0])
    
        # Snap to exact hexagonal angle
        current_strut_angle = self.snap_to_hexagonal_angle(current_strut_angle)
    
        # Calculate existing strut direction
        existing_start = np.array(existing_strut.start)
        existing_end = np.array(existing_strut.end)
        junction = np.array(junction_point)
        
        start_dist = np.linalg.norm(existing_start - junction)
        end_dist = np.linalg.norm(existing_end - junction)
        
        if start_dist < end_dist:
            existing_direction_vector = existing_end - existing_start
        else:
            existing_direction_vector = existing_start - existing_end
        
        existing_strut_angle = math.atan2(existing_direction_vector[1], existing_direction_vector[0])
        existing_strut_angle = self.snap_to_hexagonal_angle(existing_strut_angle)
        
        # Calculate exact angle difference
        angle_diff = self.normalize_angle(existing_strut_angle - current_strut_angle)
        
        # Enforce exact 120deg separation
        if abs(angle_diff - math.radians(60)) < abs(angle_diff + math.radians(60)):
            new_angle = current_strut_angle - math.radians(60)
        else:
            new_angle = current_strut_angle + math.radians(60)
        
        # Snap to exact hexagonal angle
        self.direction = self.snap_to_hexagonal_angle(new_angle)
        
        if self.verbose:
            print(f"Agent {self.id} creating complementary strut with EXACT angles "
              f"(current: {math.degrees(current_strut_angle):.0f}°, "
              f"existing: {math.degrees(existing_strut_angle):.0f}°, "
              f"new: {math.degrees(self.direction):.0f}°)")
        self.start_deposition(junction_point)
    
    def start_deposition(self, start_point):
        # Release current strut before starting deposition
        if self.current_strut:
            self.current_strut.set_occupied(False)
            self.current_strut = None
        
        # Calculate target length
        self.target_deposition_length = np.random.normal(
            loc=2.5, 
            scale=0.2 
        )
        
        # Enforce strict length bounds
        self.target_deposition_length = np.clip(
            self.target_deposition_length, 
            self.min_strut_length,  
            self.max_strut_length 
        )
        
        # Start depositing
        self.depositing_strut_start = np.array(start_point).copy()
        self.deposited_length = 0.0
        self.state = AgentState.DEPOSITING
    
    # Ensure all angles are exactly multiples of 60°
    def snap_to_hexagonal_angle(self, angle):
        angle_deg = math.degrees(angle)
        nearest_multiple = round(angle_deg / 60.0) * 60.0
        return math.radians(nearest_multiple)

    def depositing_behavior(self, struts):
        # Add state check to prevent continued deposition after leaving
        if self.state != AgentState.DEPOSITING:
            return
        
        # FIXED: Calculate exact end position based on target length and direction
        # Instead of incremental movement, calculate the exact end point
        exact_direction = self.snap_to_hexagonal_angle(self.direction)
        exact_end_position = (
            self.depositing_strut_start + 
            self.target_deposition_length * np.array([
                math.cos(exact_direction), 
                math.sin(exact_direction)
            ])
        )
        
        # Move toward exact end position in steps
        old_position = self.position.copy()
        direction_to_end = exact_end_position - self.position
        distance_to_end = np.linalg.norm(direction_to_end)
        
        if distance_to_end < self.speed_depositing:
            # Close enough - snap to exact end position
            self.position = exact_end_position.copy()
            self.deposited_length = self.target_deposition_length
        else:
            # Move toward exact end position
            direction_unit = direction_to_end / distance_to_end
            self.position += self.speed_depositing * direction_unit
            step_distance = np.linalg.norm(self.position - old_position)
            self.deposited_length += step_distance
        
        # Get current strut for intersection checking
        temp_id = f"agent_{self.id}_temp_{hash((tuple(self.depositing_strut_start), tuple(self.position)))}"
        current_strut = Strut(self.depositing_strut_start, self.position, 0.1, temp_id)
        
        # Check for contact with existing strut endpoints (only after minimum length)
        # This check comes BEFORE target length check to allow early termination on connection
        if self.deposited_length >= self.min_strut_length:
            for strut in struts:
                # Skip if this is the same strut we came from
                if (np.allclose(strut.start, current_strut.start, atol=1.0) or 
                    np.allclose(strut.end, current_strut.start, atol=1.0)):
                    continue
                
                # Check for proximity to strut ENDPOINTS only
                current_end = np.array(self.position)
                strut_start = np.array(strut.start)
                strut_end = np.array(strut.end)
                
                distance_to_start = np.linalg.norm(current_end - strut_start)
                distance_to_end = np.linalg.norm(current_end - strut_end)
                
                if distance_to_start < 0.2 or distance_to_end < 0.2:
                    # Snap to the closest endpoint for precise connection
                    if distance_to_start < distance_to_end:
                        connection_point = strut_start.copy()
                        if self.verbose:
                            print(f"Agent {self.id} snapping to start endpoint of strut {strut.id}")
                    else:
                        connection_point = strut_end.copy()
                        if self.verbose:
                            print(f"Agent {self.id} snapping to end endpoint of strut {strut.id}")
                    
                    # Update agent position to exact connection point
                    self.position = connection_point.copy()
                    if self.verbose:
                        print(f"Agent {self.id} contacted strut endpoint at length {self.deposited_length:.2f}mm, completing hexagon connection")

                    self.finish_deposition_with_connection(strut, connection_point)
                    return
        
        # Check target length termination condition (after proximity check)
        if self.deposited_length >= self.target_deposition_length * 0.99:  
            if self.verbose:
                print(f"Agent {self.id} reached exact target length ({self.target_deposition_length:.2f}mm)")
            # Ensure final position is exactly at calculated end point
            self.position = exact_end_position.copy()
            self.finish_deposition()
            return
    
    def finish_deposition(self):
        if self.verbose:
            print(f"Agent {self.id} finished depositing strut, starting homing")
        # Store the completed strut before homing (unoccupied)
        if self.depositing_strut_start is not None:
            strut_id = f"agent_{self.id}_completed_{hash((tuple(self.depositing_strut_start), tuple(self.position)))}"
            self.completed_strut = Strut(self.depositing_strut_start, self.position, 0.1, strut_id, occupied=False)
        self.state = AgentState.HOMING
    
    def finish_deposition_with_connection(self, connected_strut, connection_point):
        if self.verbose:
            print(f"Agent {self.id} finished depositing strut with connection to strut {connected_strut.id}")
        
        # Create the completed strut with exact connection point
        if self.depositing_strut_start is not None:
            strut_id = f"agent_{self.id}_connected_{hash((tuple(self.depositing_strut_start), tuple(connection_point)))}"
            self.completed_strut = Strut(self.depositing_strut_start, connection_point, 0.1, strut_id, occupied=False)
            
            # Register the connection between the new strut and existing strut
            self.completed_strut.add_connection(connected_strut, connection_point)
            
            if self.verbose:
                print(f"Agent {self.id} registered connection between new strut and existing strut {connected_strut.id} at point {connection_point}")
            if self.verbose:
                print(f"Hexagon completion detected! New strut: {self.depositing_strut_start} -> {connection_point}")
        
        self.state = AgentState.HOMING
    
    def homing_behavior(self):
        # Calculate direction to center
        direction_to_center = self.center_position - self.position
        distance_to_center = np.linalg.norm(direction_to_center)
        
        # Check if agent has reached the gathering area
        if distance_to_center <= self.gathering_radius:
            if self.verbose:
                print(f"Agent {self.id} reached gathering area, starting to gather material")
            self.state = AgentState.GATHERING
            self.gathering_time = 0
            return
        
        # Move toward center
        if distance_to_center > 0:
            self.direction = math.atan2(direction_to_center[1], direction_to_center[0])
            self.move(speed=self.speed_homing)
    
    def gathering_behavior(self):
        self.gathering_time += 1
        
        # Small random movement within gathering area to simulate material gathering
        if self.gathering_time % 5 == 0:  # Change direction every 5 steps
            self.direction = random.uniform(0, 2 * math.pi)
        
        # Move slowly within gathering radius
        distance_to_center = np.linalg.norm(self.position - self.center_position)
        if distance_to_center < self.gathering_radius * 0.9: 
            self.move(speed=self.speed_default * 0.3)  
        else:
            # If at edge of gathering area, move toward center
            direction_to_center = self.center_position - self.position
            self.direction = math.atan2(direction_to_center[1], direction_to_center[0])
            self.move(speed=self.speed_default * 0.3)
        
        # Check if gathering is complete
        if self.gathering_time >= self.gathering_duration:
            if self.verbose:
                print(f"Agent {self.id} finished gathering material, starting search")
            self.state = AgentState.SEARCHING
            # Reset direction for new search
            self.direction = random.uniform(0, 2 * math.pi)
    
    def move(self, speed=None):
        if speed is None:
            speed = self.speed_default
        self.position[0] += speed * math.cos(self.direction)
        self.position[1] += speed * math.sin(self.direction)
    
    def sense_environment(self, struts):
        detected_struts = []
        
        for strut in struts:
            if self.strut_in_view(strut):
                detected_struts.append(strut)
        
        return detected_struts
    
    def strut_in_view(self, strut):
        # Check distance to strut
        distance = self.distance_to_strut(strut)
        if distance > self.view_distance:
            return False
        
        # Check if strut is within view angle
        strut_center = (np.array(strut.start) + np.array(strut.end)) / 2
        direction_to_strut = strut_center - self.position
        angle_to_strut = math.atan2(direction_to_strut[1], direction_to_strut[0])
        
        # Normalize angle difference
        angle_diff = abs(self.normalize_angle(angle_to_strut - self.direction))
        
        return angle_diff <= self.view_angle / 2
    
    def distance_to_strut(self, strut):
        line_start = np.array(strut.start)
        line_end = np.array(strut.end)
        point = np.array(self.position)
        
        # Vector from start to end of line
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return np.linalg.norm(point - line_start)
        
        # Project point onto line
        t = np.dot(point - line_start, line_vec) / (line_len ** 2)
        t = max(0, min(1, t)) 
        
        # Find closest point on line segment
        closest_point = line_start + t * line_vec

        return np.linalg.norm(point - closest_point)
        #return self.point_to_line_distance(self.position, strut.start, strut.end)
    
    def at_boundary(self):
        x_min, y_min, x_max, y_max = self.construction_bounds
        return (self.position[0] <= x_min or self.position[0] >= x_max or
                self.position[1] <= y_min or self.position[1] >= y_max)
    
    def check_bounds(self):
        out_of_bounds = self.at_boundary()
        if out_of_bounds:
            self.state = AgentState.HOMING
    
    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def get_view_wedge(self):
        #just for the visualization. Get the geometry of the view edge
        # Calculate wedge boundaries
        left_angle = self.direction - self.view_angle / 2
        right_angle = self.direction + self.view_angle / 2
        
        # Wedge points
        left_point = self.position + self.view_distance * np.array([math.cos(left_angle), math.sin(left_angle)])
        right_point = self.position + self.view_distance * np.array([math.cos(right_angle), math.sin(right_angle)])
        
        return [self.position, left_point, right_point]
    
    def get_current_strut(self):
        if self.state == AgentState.DEPOSITING and self.depositing_strut_start is not None:
            # Create unique ID for the strut being deposited
            strut_id = f"agent_{self.id}_strut_{hash((tuple(self.depositing_strut_start), tuple(self.position)))}"
            # Strut being deposited is occupied
            return Strut(self.depositing_strut_start, self.position, 0.1, strut_id, occupied=True)
        elif self.state == AgentState.HOMING and self.completed_strut is not None:
            # Completed strut is not occupied
            self.completed_strut.set_occupied(False)
            return self.completed_strut
        return None
