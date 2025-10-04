import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
import random
import time
import math
from agents import Agent, AgentState, Strut

class SwarmLatticeSimulation:
    def __init__(self, max_agents, target_length=4, length_error=0, deposition_speed=1, verbose=False):
        # Define the construction space, assuming lengths of mm for now
        self.construction_width = 200 
        self.construction_height = 200 
        # Bound in the form of (min_x, min_y, max_x, max_y)
        self.construction_bounds = (0, 0, self.construction_width, self.construction_height)
        
        # Sim parameters
        self.max_agents = max_agents

        #Agent Paramters: 
        self.target_length = target_length
        self.length_error=length_error
        self.deposition_speed=deposition_speed
        
        # Data structures
        self.agents = []
        self.struts = []
        self.completed_struts = []
        self.agent_counter = 0
        self.step_counter = 0
        self.steps_without_new_material = 0
        self.last_total_strut_length = 0
        
        # For visualization, assign each state to a dot color
        self.state_colors = {AgentState.SEARCHING: 'blue',
            AgentState.MOVING_TO_STRUT: 'green',
            AgentState.FOLLOWING_STRUT: 'orange',
            AgentState.DEPOSITING: 'red',
            AgentState.HOMING: 'purple',
            AgentState.GATHERING: 'cyan'}
        
        # Setup for the animation
        self.frames = [] 
        self.record_video = False
        self.video_writer = None
        
        # Setup for real-time agent visualization
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.setup_visualization() 
        
        # Tracking parameters
        self.stats = {'total_struts_completed': 0,
                      'total_length_deposited': 0,
                      'simulation_time': 0}
        
        # Initial strut at the center
        initial_pos = [self.construction_width / 2, self.construction_height / 2]
        
        strut_length = self.target_length  
        angle = 0
        end_pos = [initial_pos[0] + strut_length * math.cos(angle), initial_pos[1] + strut_length * math.sin(angle)]
        
        initial_strut = Strut(initial_pos, end_pos, 0.1, "initial_strut_0", occupied=False)
        self.struts.append(initial_strut)
        self.completed_struts.append(initial_strut)
        
        print(f"Initial short strut created at center: {initial_pos} -> {end_pos} (length: {strut_length}mm)")
        
        # Spawn all agents at the center initially
        self.spawn()
        self.verbose = verbose 
    
    def setup_visualization(self):
        #just the initialization of the plot
        self.ax.set_xlim(-2, self.construction_width + 2)
        self.ax.set_ylim(-2, self.construction_height + 2)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Swarm Lattice Construction')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        
        # Boundary
        boundary_x = [0, self.construction_width, self.construction_width, 0, 0]
        boundary_y = [0, 0, self.construction_height, self.construction_height, 0]
        self.ax.plot(boundary_x, boundary_y, 'k-', linewidth=2, label='Construction Boundary')

        legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=state.value.replace('_', ' ').title()) for state, color in self.state_colors.items()]
        legend_elements.append(plt.Line2D([0], [0], linestyle=":", color='red', lw=10, alpha=0.3, label='Gathering Zone'))
        self.ax.legend(handles=legend_elements, loc='upper right')
    
    def spawn(self):
        center_pos = [self.construction_width / 2, self.construction_height / 2]
        
        for i in range(self.max_agents):
            # For realism, add a small random offset
            offset_x = random.uniform(-0.5, 0.5)
            offset_y = random.uniform(-0.5, 0.5)
            agent_pos = [center_pos[0] + offset_x, center_pos[1] + offset_y]
            
            new_agent = Agent(self.agent_counter, agent_pos, self.construction_bounds, strut_target_length=self.target_length, strut_errors=self.length_error, deposition_speed=self.deposition_speed)
            self.agents.append(new_agent)
            self.agent_counter += 1
        
        print(f"Spawned {self.max_agents} agents at center")
    
    def update_simulation(self):
        #Each step of the simulation
        self.step_counter += 1
        
        # Update agents individually 
        for agent in self.agents[:]: 
            agent.update(self.struts)
            
            # Check for newl struts
            current_strut = agent.get_current_strut()
            if (agent.state == AgentState.DEPOSITING and current_strut and not self.strut_exists(current_strut, self.struts)):
                self.struts.append(current_strut)
            
            # When agents finish depositing, they go back to the center to "pick up more material"
            if agent.state == AgentState.HOMING and agent.completed_strut:
                deposited_strut = agent.get_current_strut()
                if deposited_strut and not self.strut_exists(deposited_strut, self.completed_struts):
                    # Update connections and add to list of struts
                    self.update_strut_connections(deposited_strut)
                    self.completed_struts.append(deposited_strut)
                    self.stats['total_struts_completed'] += 1
                    if self.verbose: 
                        print(f"Agent {agent.id} completed strut: {deposited_strut.start} -> {deposited_strut.end} with angle {np.arctan2((deposited_strut.end[1]-deposited_strut.start[1]), (deposited_strut.end[0]-deposited_strut.start[0])) * 180/np.pi}")
                    # Clear the completed strut so it's only processed once
                    agent.completed_strut = None
        
        # Update struts list to include all depositing struts
        self.struts = self.completed_struts.copy()
        for agent in self.agents:
            # Only add struts from agents that are actively depositing
            if agent.state == AgentState.DEPOSITING:
                current_strut = agent.get_current_strut()
                if current_strut:
                    self.struts.append(current_strut)
        
        # Check termination condition
        current_total_length = sum(strut.get_length() for strut in self.completed_struts)
        if abs(current_total_length - self.last_total_strut_length) < 1e-6:
            self.steps_without_new_material += 1
        else:
            self.steps_without_new_material = 0
            self.last_total_strut_length = current_total_length
        
        self.stats['total_length_deposited'] = current_total_length
        self.stats['simulation_time'] = self.step_counter
        
        # Print progress periodically, even if not in verbose mode
        if self.step_counter % 1000 == 0:
            print(f"Step {self.step_counter}: {len(self.agents)} agents, {len(self.completed_struts)} completed struts, {current_total_length:.2f} mm total length")
    
    def strut_length(self, strut):
        #method for quickly grabbing length of a strut
        return strut.get_length()
    
    #compare position and strut ID to find if there is a strut in the 
    def strut_exists(self, target_strut, strut_list):
        if target_strut is None:
            return False
        
        for existing_strut in strut_list:
            if target_strut.id == existing_strut.id:
                return True
        
        #position-based comparison. makes sim run slower but this is just a check that robots could do visually
        for existing_strut in strut_list:
            start_match = np.allclose(target_strut.start, existing_strut.start, atol=1e-6)
            end_match = np.allclose(target_strut.end, existing_strut.end, atol=1e-6)
            
            start_end_match = np.allclose(target_strut.start, existing_strut.end, atol=1e-6)
            end_start_match = np.allclose(target_strut.end, existing_strut.start, atol=1e-6)
            
            if (start_match and end_match) or (start_end_match and end_start_match):
                return True
        
        return False
    
    #function for drawing each frame
    def draw_frame(self, save_frame=False):
        self.ax.clear()
        self.setup_visualization()
        
        # Draw gathering zone (permanent red circle at center)
        center_x = self.construction_width / 2
        center_y = self.construction_height / 2
        gathering_radius = 2.0  
        gathering_circle = Circle((center_x, center_y), gathering_radius, color='red', alpha=0.3, linewidth=2, fill=True, edgecolor='red')
        self.ax.add_patch(gathering_circle)
        
        # Draw completed struts
        for strut in self.completed_struts:
            self.ax.plot([strut.start[0], strut.end[0]], [strut.start[1], strut.end[1]], 'k-', linewidth=2, alpha=0.8)
        
        # Draw agents and their current struts
        for agent in self.agents:
            # Incerase size of gathering and homing since they won't have view triangle
            if agent.state in [AgentState.GATHERING, AgentState.HOMING]:
                agent_size = agent.body_size * 2 
            else:
                agent_size = agent.body_size
            
            # Draw agent position
            agent_circle = Circle(agent.position, agent_size, color=self.state_colors[agent.state], alpha=0.7)
            self.ax.add_patch(agent_circle)
            
            # Draw view wedge when doing anything but homing and gathering
            if agent.state not in [AgentState.HOMING, AgentState.GATHERING]:
                view_points = agent.get_view_wedge()
                if len(view_points) >= 3:
                    view_wedge = Polygon(view_points, color=self.state_colors[agent.state], alpha=0.2)
                    self.ax.add_patch(view_wedge)
            
            # Draw current strut being deposited
            current_strut = agent.get_current_strut()
            if current_strut and agent.state == AgentState.DEPOSITING: 
                self.ax.plot([current_strut.start[0], current_strut.end[0]], [current_strut.start[1], current_strut.end[1]], color=self.state_colors[agent.state], linewidth=1.5, alpha=0.6)
        
        # Update title with statistics
        title = (f'Swarm Lattice Construction - Step: {self.step_counter}, '
                f'Active Agents: {len(self.agents)}, '
                f'Completed Struts: {len(self.completed_struts)}, '
                f'Total Length: {self.stats["total_length_deposited"]:.1f} mm')
        self.ax.set_title(title)
        
        # Save frame for video if recording
        if save_frame and self.record_video and self.video_writer is not None:
            self.video_writer.grab_frame()
        
        if not save_frame: 
            plt.draw()
            plt.pause(0.01) 
    
    #function for matplotlib animation setup 
    def setup_video_recording(self, filename, fps=10):
        try:
            # Try different writers in order of preference
            writers = ['ffmpeg', 'pillow']
            
            for writer_name in writers:
                try:
                    Writer = animation.writers[writer_name]
                    self.video_writer = Writer(fps=fps, metadata=dict(artist='SwarmLattice'), bitrate=1800)
                    self.record_video = True
                    print(f"Video recording setup with {writer_name} writer")
                    return True
                except (KeyError, RuntimeError):
                    continue
            
            print("Warning: No suitable video writer found. Video recording disabled.")
            self.record_video = False
            return False
            
        except Exception as e:
            print(f"Error setting up video recording: {e}")
            self.record_video = False
            return False
    
    def start_video_recording(self, filename):
        if self.record_video and self.video_writer is not None:
            try:
                self.video_writer.setup(self.fig, filename, dpi=100)
                print(f"Started recording video to {filename}")
                return True
            except Exception as e:
                print(f"Error starting video recording: {e}")
                self.record_video = False
                return False
        return False
    
    def stop_video_recording(self):
        if self.record_video and self.video_writer is not None:
            try:
                self.video_writer.finish()
                print("Video recording completed and saved")
                return True
            except Exception as e:
                print(f"Error finishing video recording: {e}")
                return False
        return False
    
    def run_simulation(self, max_steps=50000, visualization=True, save_video=False, video_filename=None):
        print("Starting swarm lattice construction simulation...")
        print(f"Construction space: {self.construction_width} x {self.construction_height} mm")
        print(f"Max agents: {self.max_agents}")
        
        start_time = time.time()
        
        # Setup video recording if save_video=true
        if save_video:
            if video_filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_filename = f"swarm_lattice_animation_{timestamp}.mp4"
            
            if self.setup_video_recording(video_filename):
                self.start_video_recording(video_filename)
            else:
                save_video = False
                print("Video recording disabled due to setup failure")
        
        try:
            if visualization:
                plt.ion()
                self.draw_frame(save_frame=save_video)
            
            while self.step_counter < max_steps:
                self.update_simulation()
                
                #for just visualization, can have lower frame rate
                if visualization and self.step_counter % 10 == 0:  
                    self.draw_frame(save_frame=save_video)
                elif save_video and self.step_counter % 5 == 0: 
                    self.draw_frame(save_frame=True)
            
            if visualization:
                plt.ioff()  
                self.draw_frame(save_frame=save_video) 
                if not save_video: 
                    plt.show()
        
        except KeyboardInterrupt:
            print("stopped")
        
        finally:
            if save_video:
                self.stop_video_recording()
        
        end_time = time.time()
        
        # Final statistics
        print("Simulation Complete. Statistics:")
        print(f"Total simulation steps: {self.step_counter}\nTotal runtime: {end_time - start_time:.2f} seconds\nTotal struts completed: {self.stats['total_struts_completed']}\nTotal length deposited: {self.stats['total_length_deposited']:.2f} mm")
        
        termination_reason = "Maximum steps reached" if self.step_counter >= max_steps else "No new material deposited"
        print(f"Termination reason: {termination_reason}")
        
        if save_video:
            print(f"Animation video saved as: {video_filename}")
        
        return self.completed_struts
    
    def save_results(self, filename="swarm_lattice"):
        # Save strut data
        strut_data = []
        for i, strut in enumerate(self.completed_struts):
            strut_data.append({'id': i,
                'start_x': strut.start[0],
                'start_y': strut.start[1],
                'end_x': strut.end[0],
                'end_y': strut.end[1],
                'length': strut.get_length()})
        
        # Save final static image as well
        plt.figure(figsize=(12, 10))
        plt.xlim(-2, self.construction_width + 2)
        plt.ylim(-2, self.construction_height + 2)
        plt.gca().set_aspect('equal')
        plt.grid(True, alpha=0.3)
        
        # Draw construction boundary
        boundary_x = [0, self.construction_width, self.construction_width, 0, 0]
        boundary_y = [0, 0, self.construction_height, self.construction_height, 0]
        plt.plot(boundary_x, boundary_y, 'k-', linewidth=2)
        
        # Draw all struts
        for strut in self.completed_struts:
            plt.plot([strut.start[0], strut.end[0]], [strut.start[1], strut.end[1]], 'k-', linewidth=1.5)
        
        plt.title(
            f'Statistics: {len(self.completed_struts)} struts, '
            f'{self.stats["total_length_deposited"]:.1f} mm total length',
            fontsize=24
        )
        plt.xlabel('X Position (mm)', fontsize=24)
        plt.ylabel('Y Position (mm)', fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results saved with prefix: {filename}")
        print(f"Final structure image: {filename}.png")
        
        return strut_data
    
    #Double check to make sure strut connections are updated 
    def update_strut_connections(self, new_strut):
        connection_tolerance = 0.5 
        
        for existing_strut in self.completed_struts:
            if existing_strut.id == new_strut.id:
                continue  
            
            # Check all possible endpoint connections
            connections_to_check = [
                (new_strut.start, existing_strut.start),
                (new_strut.start, existing_strut.end),
                (new_strut.end, existing_strut.start),
                (new_strut.end, existing_strut.end)
            ]
            
            for new_point, existing_point in connections_to_check:
                distance = np.linalg.norm(np.array(new_point) - np.array(existing_point))
                if distance <= connection_tolerance:
                    # Create connection
                    new_strut.add_connection(existing_strut, new_point)
                    if self.verbose:
                        print(f"Connected strut from [{new_strut.start}, {new_strut.end}] to strut from  [{existing_strut.start}, {existing_strut.end}] at point {new_point}")
    
def main():
    # Chose between visualization/no visualization
    print("Sim Configuration:")
    print("For Real-Time Visualization, type rtv")
    print("For Sim w/o Visualization, type sim")
    print("To save the video, type vid")
    print("To run the full experimental setup for my report, type full_dataset")
    choice = input("Type your option here: ").strip()
    if not choice:
        choice = "rtv"
    
    # Set parameters based on user choice
    save_video = False
    if choice == "rtv":
        visualization = True
        video=False
        single_sim = True
    elif choice == "sim":
        visualization = False
        video=False
        single_sim = True
    elif choice == "vid":
        visualization = False 
        video=True
        single_sim = True
    elif choice == "full_dataset":
        visualization=False
        video=False 
        single_sim=False
    else:
        print("Invalid choice, using rtv")
        choice = "rtv"
        visualization = True
        video=False
        single_sim = True

    if single_sim: 
        # Get inputs from user: 
        print(f"Using {choice} settings. Now choising the parameters of the simulation")
        agents = int(input("What is the max number of agent's you'd like?(int)"))
        steps = int(input("What is the total number of steps you'd like?(int)"))
        avg_len = float(input("What average strut length would you like?(float)"))
        len_err = float(input("What is the std of the length error you'd like?(float)"))
        build_speed = float(input("What is the build speed you'd like?(float)"))

        # Create and run simulation
        sim = SwarmLatticeSimulation(agents, target_length=avg_len, length_error=len_err, deposition_speed=build_speed)
        
        try:
            completed_struts = sim.run_simulation(max_steps=steps, visualization=visualization,save_video=video)
            sim.save_results()
            
            # Show results
            print("Final Results:")
            if completed_struts:
                total_length = sum(strut.get_length() for strut in completed_struts)
                print(f"Structure Statistics:\n\tTotal struts: {len(completed_struts)}/n/tTotal length: {total_length:.2f}")
            else:
                print("No struts were completed during simulation. Something is wrong")
                
        except Exception as e:
            print("Some sort of error in the simulation")
            print(e)
    else: 
        # Only scenario in which there isn't just a single simulation is if we are doing the full dataset
        # As a small note, this takes absolutely forever to run
        experiment_conditions = {1: {"agents": 1, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1}, 
                                 2: {"agents": 3, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1}, 
                                 3: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1}, 
                                 4: {"agents": 7, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1}, 
                                 5: {"agents": 10, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1}, 
                                 6: {"agents": 20, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 7: {"agents": 5, "steps": 50000, "avg_len":1,"len_err": 0, "dep_speed": 0.1},  
                                 8: {"agents": 5, "steps": 50000, "avg_len":7,"len_err": 0, "dep_speed": 0.1}, 
                                 9: {"agents": 5, "steps": 50000, "avg_len":10,"len_err": 0, "dep_speed": 0.1},
                                 10: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0.5, "dep_speed": 0.1},
                                 11: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 1, "dep_speed": 0.1},
                                 12: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 2, "dep_speed": 0.1},
                                 13: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.4},
                                 14: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 1},
                                 15: {"agents": 5, "steps": 5000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 16: {"agents": 5, "steps": 10000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 17: {"agents": 5, "steps": 25000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 18: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 19: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 20: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 21: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 22: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 23: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 24: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 25: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1},
                                 26: {"agents": 5, "steps": 50000, "avg_len":4,"len_err": 0, "dep_speed": 0.1}}
        
        for run in experiment_conditions.keys(): 
            if run >= 0: #this conditional is just in place to select certain runs if I need to re-run them 
                condition_i = experiment_conditions[run]
                agents_i = condition_i["agents"]
                steps_i = condition_i["steps"]
                avg_len_i = condition_i["avg_len"]
                len_err_i = condition_i["len_err"]
                dep_speed_i = condition_i["dep_speed"]
                filename = f"exp{run}_{agents_i}a_{steps_i}s_{avg_len_i}len_{len_err_i}err_{dep_speed_i}ds"

                print(f"=*10 Starting Experiment {run} =*10")
                sim = SwarmLatticeSimulation(agents_i, target_length=avg_len_i, length_error=len_err_i, deposition_speed=dep_speed_i)

                try:
                    completed_struts = sim.run_simulation(max_steps=steps_i, visualization=visualization, save_video=video)
                    sim.save_results(filename=filename)
                    
                    # Show results
                    print("Final Results:")
                    if completed_struts:
                        total_length = sum(strut.get_length() for strut in completed_struts)
                        print(f"Structure Statistics:\n\tTotal struts: {len(completed_struts)}/n/tTotal length: {total_length:.2f}")
                    else:
                        print("No struts were completed during simulation. Something is wrong")
                        
                except Exception as e:
                    print("Some sort of error in the simulation")
                    print(e)

if __name__ == "__main__":
    main()
    