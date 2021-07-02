''' 
This environment describe a fixed scene (area) to conduct end-to-end lateral control tasks
for the autonomous ego vehicle. (This environment is relative simple and is only for training)
'''

import pygame
import weakref
import collections
import numpy as np
import math
import cv2
import sys
'''
add your path of the CARLA simulator, this script was originally run with CARLA(0.9.7), 
some functions (e.g., carla.set_velocity()) have been removed in the newer CARLA, 
please refer to CARLA official document for details if you want to run the script with a different version.
'''
sys.path.append('xxx/carla-0.9.X-py3.X-linux-x86_64.egg')

import carla
from carla import ColorConverter as cc

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser

from utils import get_path
path_generator = get_path()

velocity_target_ego = 5
x_bench = 335.0
y_bench = 200.0
WIDTH, HEIGHT = 80, 45

class scenario(object):
    def __init__(self, random_spawn=True, pedestrian=False, 
                 no_render=False, frame=25):

        self.observation_size = WIDTH * HEIGHT
        self.width = WIDTH
        self.height = HEIGHT
        self.action_size = 1
        

        ## set the carla World parameters
        self.pedestrian = pedestrian
        self.random_spawn = random_spawn
        self.no_render = no_render

        ## set the vehicle actors
        self.ego_vehicle = None
        self.obs1 = None
        self.obs2 = None
        self.obs3 = None

        ## set the sensory actors
        self.collision_sensor = None
        self.seman_camera = None
        self.viz_camera = None
        self.surface = None
        self.camera_output = np.zeros([720,1280,3])
        self.recording = False
        self.Attachment = carla.AttachmentType

        ## connect to the CARLA client
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(10.0)
        
        ## build the CARLA world
        self.world = self.client.load_world('Town01')
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1/frame
        settings.no_rendering_mode = self.no_render
        settings.SynchronousMode = True
        self.world.apply_settings(settings)
        
        ## initialize the pygame settings
        pygame.init()
        pygame.font.init()
        pygame.joystick.init()
        self.display = pygame.display.set_mode((1280, 720),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.infoObject = pygame.display.Info()

        ## initilize the joystick settings
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
            
        self._parser = ConfigParser()
        self._parser.read('./wheel_config.ini')
        self._steer_idx = int(self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(self._parser.get('G29 Racing Wheel', 'handbrake'))

        self.restart()

    def restart(self):

        ## reset the recording lists
        self.steer_history = []
        self.intervene_history = []

        ## reset the human intervention state
        self.intervention = False

        ## spawn three surrounding vehicles
        self.bp_obs1, self.spawn_point_obs1 = self._produce_vehicle_blueprint(1, 335.0+3.5, 100.0)
        self.obs1 = self.world.spawn_actor(self.bp_obs1,self.spawn_point_obs1)

        self.bp_obs2, self.spawn_point_obs2 = self._produce_vehicle_blueprint(1, 335.0, 200.0+25.0)
        self.obs2 = self.world.spawn_actor(self.bp_obs2,self.spawn_point_obs2)

        self.bp_obs3, self.spawn_point_obs3 = self._produce_vehicle_blueprint(1, 335.0+3.5, 200.0+50.0)
        self.obs3 = self.world.spawn_actor(self.bp_obs3,self.spawn_point_obs3)

        ## if pedestrians are considered, spawn two persons 
        if self.pedestrian:
            self.bp_walker1, self.spawn_point_walker1 = self._produce_walker_blueprint(338.0, 200+np.random.randint(10,15))
            self.bp_walker2, self.spawn_point_walker2 = self._produce_walker_blueprint(np.random.randint(3310,3350)/10, 235)

            self.walker1 = self.world.spawn_actor(self.bp_walker1, self.spawn_point_walker1)
            self.walker2 = self.world.spawn_actor(self.bp_walker2, self.spawn_point_walker2)

            walker1_control = carla.WalkerControl()
            walker1_control.speed = 0.1
            self.walker1.apply_control(walker1_control)

            walker2_control = carla.WalkerControl()
            walker2_control.speed = 0.1
            self.walker2.apply_control(walker2_control)
        
        ## spawn the ego vehicle (random / fixed)
        if self.random_spawn:
            y_spawn_random = np.random.randint(200, 240)
            random_lateral_disturb = 0.1 * (np.random.rand()-0.5)
            x_spwan_random = path_generator(y_spawn) + random_lateral_disturb
            self.bp_ego, self.spawn_point_ego = self._produce_vehicle_blueprint(1, x_spwan_random, y_spawn_random)
        else:
            self.bp_ego, self.spawn_point_ego = self._produce_vehicle_blueprint(1 , x_bench, y_bench)
        self.ego_vehicle = self.world.spawn_actor(self.bp_ego,self.spawn_point_ego)
        
        # set the initial velocity of the ego vehicle
        initial_velocity = carla.Vector3D(0, velocity_target_ego, 0)
        self.ego_vehicle.set_velocity(initial_velocity)

        # initilize the control variable for the ego vehicle
        self.control = carla.VehicleControl()


        ## configurate and spawn the collision sensor
        # clear the collision history list
        self.collision_history = []
        bp_collision = self.world.get_blueprint_library().find('sensor.other.collision')
        # spawn the collision sensor actor
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        self.collision_sensor = self.world.spawn_actor(
                bp_collision, carla.Transform(), attach_to=self.ego_vehicle)
        # obtain the collision signal and append to the history list
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: scenario._on_collision(weak_self, event))

        
        ## configurate and spawn the camera sensors
        # the candidated transform of camera's position: frontal
        self.camera_transforms = [
            (carla.Transform(carla.Location(x=-2, z=5), carla.Rotation(pitch=30.0)), self.Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-2, z=5), carla.Rotation(pitch=30.0)), self.Attachment.SpringArm)]
        self.camera_transform_index = 1
        # the candidated camera type: rgb (viz_camera) and semantic (seman_camera)
        self.cameras = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}]
            ]
                
        bp_viz_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp_viz_camera.set_attribute('image_size_x', '1280')
        bp_viz_camera.set_attribute('image_size_y', '720')
        bp_viz_camera.set_attribute('sensor_tick', '0.02')
        self.cameras[0].append(bp_viz_camera)

        bp_seman_camera = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp_seman_camera.set_attribute('image_size_x', '1280')
        bp_seman_camera.set_attribute('image_size_y', '720')
        bp_seman_camera.set_attribute('sensor_tick', '0.04')
        self.cameras[1].append(bp_seman_camera)
        

        # spawn the camera actors
        if self.seman_camera is not None:
            self.seman_camera.destroy()
            self.surface = None

        self.viz_camera = self.world.spawn_actor(
            self.cameras[0][-1],
            self.camera_transforms[self.camera_transform_index][0],
            attach_to=self.ego_vehicle,
            attachment_type=self.Attachment.SpringArm)

        self.seman_camera = self.world.spawn_actor(
            self.cameras[1][-1],
            self.camera_transforms[self.camera_transform_index - 1][0],
            attach_to=self.ego_vehicle,
            attachment_type=self.camera_transforms[self.camera_transform_index - 1][1])

        # obtain the camera image
        weak_self = weakref.ref(self)
        self.seman_camera.listen(lambda image: scenario._parse_seman_image(weak_self, image))
        self.viz_camera.listen(lambda image: scenario._parse_image(weak_self, image))

        
        ## reset the step counter
        self.count = 0

    
    def render(self, display):
        if self.surface is not None:
            m = pygame.transform.smoothscale(self.surface, 
                                 [int(self.infoObject.current_w), 
                                  int(self.infoObject.current_h)])
            display.blit(m, (0, 0))


    def _parse_seman_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.cameras[1][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.array(image.raw_data)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        self.camera_output = array
    

    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.cameras[0][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.array(image.raw_data)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_history.append((event.frame, intensity))
        if len(self.collision_history) > 4000:
            self.collision_history.pop(0)


    def get_collision_history(self):
        collision_history = collections.defaultdict(int)
        flag = 0
        for frame, intensity in self.collision_history:
            collision_history[frame] += intensity
            if intensity != 0:
                flag = 1
        return collision_history, flag
    
    
    def run_step(self,action):

        self.render(self.display)
        pygame.display.flip()
        
        self.parse_events()

        human_control = None
        
        # retrive the signals from the joystick (steering wheel)
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]

        ## if no human intervention
        if not self.intervention:
            steerCmd = action / 2
            self.control.steer = math.tan(1.1 * steerCmd)
        ## if intervention detected, convert the joystick signal to the steering command
        else:
            K1 = 1
            steerCmd = K1 * (2 * jsInputs[self._steer_idx])
            self.control.steer = steerCmd

            human_control = self.control.steer
        
        ## detect the intervention signal
        if len(self.steer_history) > 2:
            # the intervention is activated if human participants move the joystick
            if abs(self.intervene_history[-2] - self.intervene_history[-1]) > 0.02:
                self.intervention = True
        if len(self.steer_history) > 5:
            # the intervention is deactivated if the joystick continue to be stable for 0.2 seconds
            if abs(self.intervene_history[-5] - self.intervene_history[-1]) < 0.01:
                self.intervention = False

        ## record the intervention histroy (get "None" for non-intervened steps)
        self.intervene_history.append(jsInputs[0])
        
        # record the steering command history
        self.steer_history.append(steerCmd)
        
        ## configurate the control command for the ego vehicle
        # the velocity is calculated as :sqrt(vx**2+vy**2)
        velocity_ego = ((self.ego_vehicle.get_velocity().x)**2 + (self.ego_vehicle.get_velocity().y)**2)**(1/2)
        
        # the longitudinal control (throttle) of the ego vehicle is achieved by a proportional controller
        self.control.throttle = np.clip (velocity_target_ego - velocity_ego, 0, 1)
        self.control.brake = 0
        self.control.hand_brake = 0

        ## achieve the control to the ego vehicle
        self.ego_vehicle.apply_control(self.control)

        ## obtain the state transition and other variables after taking the action (control command)
        next_states, other_indicators = self.obtain_observation()

        ## detect if the step is the terminated step, by considering: collision, beyond the road, and episode fininsh
        collision = self.get_collision_history()[1]
        finish = (self.ego_vehicle.get_location().y > y_bench + 55.0)
        beyond = (self.ego_vehicle.get_location().x < x_bench - 1.2) or (self.ego_vehicle.get_location().x > x_bench + 4.8)
        done = collision or finish or beyond

        ## calculate the relative distance to the surrounding vehicles for the subsequent reward function
        dis_to_front = other_indicators['state_front']
        dis_to_side = min(other_indicators['state_left'],other_indicators['state_right'])
        dis_to_obs11 = other_indicators['state_corner_11']
        dis_to_obs12 = other_indicators['state_corner_12']
        dis_to_obs21 = other_indicators['state_corner_21']
        dis_to_obs22 = other_indicators['state_corner_22']
        dis_to_obs31 = other_indicators['state_corner_31']
        dis_to_obs32 = other_indicators['state_corner_32']
        
        ## calculate the reward signal of the step: r1-r3 distance reward, r4 terminal reward, r5-r6 smooth reward
        r1 = -1*np.square(1-dis_to_front)
        r2 = -2*np.square(1-dis_to_side)
        r3 = - (np.abs(1-dis_to_obs11)+np.abs(1-dis_to_obs12)+np.abs(1-dis_to_obs21)+np.abs(1-dis_to_obs22)+np.abs(1-dis_to_obs31)+np.abs(1-dis_to_obs32))
        r4 = finish*10 - collision*10 - beyond*10
        r5= -np.float32(abs(self.steer_history[-1]-steerCmd)>0.1)
        r6 = -3*abs(steerCmd)
        
        reward = r1+r2+r3+r4+r5+r6+0.2
        reward = np.clip(reward,-10,10)
        
        ## update the epsodic step
        self.count += 1
        
        ## record the physical variables
        yaw_rate = np.arctan(self.ego_vehicle.get_velocity().x/self.ego_vehicle.get_velocity().y) if self.ego_vehicle.get_velocity().y > 0 else 0
        physical_variables = {'velocity_y':self.ego_vehicle.get_velocity().y,
                 'velocity_x':self.ego_vehicle.get_velocity().x,
                 'position_y':self.ego_vehicle.get_location().y,
                 'position_x':self.ego_vehicle.get_location().x,
                 'yaw_rate':yaw_rate,
                 'yaw':self.ego_vehicle.get_transform().rotation.yaw,
                 'pitch':self.ego_vehicle.get_transform().rotation.pitch,
                 'roll':self.ego_vehicle.get_transform().rotation.roll,
                 'angular_velocity_y':self.ego_vehicle.get_angular_velocity().y,
                 'angular_velocity_x':self.ego_vehicle.get_angular_velocity().x
                 }
        
        if done:
            self.destroy()
            
        return next_states, human_control, reward, self.intervention, done, physical_variables
    

    def destroy(self):
        actors = [
            self.ego_vehicle,
            self.obs1,
            self.obs2,
            self.obs3,
            self.seman_camera,
            self.viz_camera,
            self.collision_sensor]
        self.seman_camera.stop()
        self.viz_camera.stop()
        for actor in actors:
            if actor is not None:
                actor.destroy()


    def obtain_observation(self):
        ## obtain image-based state space
        # state variable sets
        state_space = self.camera_output[:,:,0]
        state_space = cv2.resize(state_space,(WIDTH, HEIGHT))
        state_space = np.resize(state_space,(self.observation_size, 1))
        state_space = np.squeeze(state_space)/255
        
        ## obtain space variables for reward generation
        velocity_self = self.ego_vehicle.get_velocity()
        position_self = self.ego_vehicle.get_location()
        yaw_self = self.ego_vehicle.get_transform().rotation.yaw
        position_obs1 = self.obs1.get_location()
        position_obs2 = self.obs2.get_location()
        position_obs3 = self.obs3.get_location()

        ## obtain relative distance information for reward generation
        # pre-calculated parameters
        xa,ya,xb,yb,xc,yc,xd,yd = self._to_corner_coordinate(position_self.x,position_self.y,yaw_self)
        xfc = (xa+xb)/2
        yfc = (ya+yb)/2
        xa1,ya1,xb1,yb1,xc1,yc1,xd1,yd1 = 337.4,202.4,339.6,202.4,339.6,197.6,337.4,197.6
        xa2,ya2,xb2,yb2,xc2,yc2,xd2,yd2 = 333.9,227.4,336.1,227.4,336.1,222.6,333.9,222.6
        xa3,ya3,xb3,yb3,xc3,yc3,xd3,yd3 = 337.4,252.4,339.6,252.4,339.6,247.6,337.4,247.6
        
        # relative distance from ego vehicle to obstacle 1 (corner distance)
        if position_obs1.y - 4 < position_self.y < position_obs1.y + 4:
            state_corner_11 = self._sigmoid(np.clip(abs(xa1-xa),0,10),2.5)
            state_corner_12 = self._sigmoid(np.clip(abs(xa1-xb),0,10),2.5)
        else:
            state_corner_11 = 1
            state_corner_12 = 1
        # relative distance from ego vehicle to obstacle 2 (corner distance)
        if position_obs2.y - 4 < position_self.y < position_obs2.y + 4:
            state_corner_21 = self._sigmoid(np.clip(abs(xb2-xa),0,10),2.5)
            state_corner_22 = self._sigmoid(np.clip(abs(xb2-xb),0,10),2.5)
        else:
            state_corner_21 = 1
            state_corner_22 = 1
        # relative distance from ego vehicle to obstacle 3 (corner distance)
        if position_obs3.y - 4 < position_self.y < position_obs3.y + 4:
            state_corner_31 = self._sigmoid(np.clip(abs(xa3-xa),0,10),2.5)
            state_corner_32 = self._sigmoid(np.clip(abs(xa3-xb),0,10),2.5)
        else:
            state_corner_31 = 1
            state_corner_32 = 1
        # relative distance to both sides of road
        state_left = self._sigmoid(np.clip(340-xb,0,10),2)
        state_right = self._sigmoid(np.clip(xb-332,0,10),2)
        # relative distance front
        RIGHT = 1 if position_self.x < x_bench + 1.8 else 0
        if RIGHT:
            if position_self.y < y_bench + 25.0:
                state_front = np.clip(yc2 - position_self.y - 2.6, 0, 25)
                state_front = self._sigmoid(state_front,1)
            else:
                state_front = 1
        else:
            state_front = np.clip(yc3 - position_self.y - 2.4, 0,25)
            state_front = self._sigmoid(state_front,1)
    
        # other indicators facilitating producing reward function signal
        other_indicators = {'state_front':state_front,
                            'state_left':state_left,
                            'state_right':state_right,
                            'state_corner_11':state_corner_11,
                            'state_corner_12':state_corner_12,
                            'state_corner_21':state_corner_21,
                            'state_corner_22':state_corner_22,
                            'state_corner_31':state_corner_31,
                            'state_corner_32':state_corner_32}
        
        return state_space, other_indicators
    

    def obtain_real_observation(self):
        state_space = self.camera_output[:,:,0]
        return state_space
        

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    self.intervention = False
                elif event.button == self._reverse_idx:
                    self.control.gear = 1 if self.control.reverse else -1
                elif event.button == 1:
                    self._toggle_camera()
                elif event.button == 2:
                    self._next_sensor()


    def _produce_vehicle_blueprint(self, color, x, y, vehicle='bmw'):
        if vehicle=='bmw':
            bp = self.world.get_blueprint_library().filter('vehicle.bmw.*')[0]
        elif vehicle=='moto':
            bp = self.world.get_blueprint_library().filter('vehicle.harley-davidson.*')[0]
        elif vehicle=='bike':
            bp = self.world.get_blueprint_library().filter('vehicle.diamondback.century.*')[0]
        elif vehicle=='bus':
            bp = self.world.get_blueprint_library().filter('vehicle.volkswagen.*')[0]
        else:
            bp = self.world.get_blueprint_library().filter('vehicle.lincoln.*')[0]
        
        bp.set_attribute('color', bp.get_attribute('color').recommended_values[color])

        spawn_point = self.world.get_map().get_spawn_points()[0]
        spawn_point.location.x = x
        spawn_point.location.y = y
        spawn_point.location.z += 0.1

        return bp, spawn_point
    
    def _produce_walker_blueprint(self, x, y):
        
        bp = self.world.get_blueprint_library().filter('walker.*')[np.random.randint(2)]
        
        spawn_point = self.world.get_map().get_spawn_points()[0]
        spawn_point.location.x = x
        spawn_point.location.y = y
        spawn_point.location.z += 0.1      
        spawn_point.rotation.yaw = 0
        
        return bp, spawn_point
        
    def _toggle_camera(self):
        self.camera_transform_index = (self.camera_transform_index + 1) % len(self.camera_transforms)
    
    def _next_sensor(self):
        self.camera_index += 1
        
    def _dis_p_to_l(self,k,b,x,y):
        dis = abs((k*x-y+b)/math.sqrt(k*k+1))
        return self._sigmoid(dis,2)
    
    def _calculate_k_b(self,x1,y1,x2,y2):
        k = (y1-y2)/(x1-x2)
        b = (x1*y2-x2*y1)/(x1-x2)
        return k,b
    
    def _dis_p_to_p(self,x1,y1,x2,y2):
        return math.sqrt((x1-x2)**2+(y1-y2)**2)
    
    def _to_corner_coordinate(self,x,y,yaw):
        xa = x+2.64*math.cos(yaw*math.pi/180-0.43)
        ya = y+2.64*math.sin(yaw*math.pi/180-0.43)
        xb = x+2.64*math.cos(yaw*math.pi/180+0.43)
        yb = y+2.64*math.cos(yaw*math.pi/180+0.43)
        xc = x+2.64*math.cos(yaw*math.pi/180-0.43+math.pi)
        yc = y+2.64*math.cos(yaw*math.pi/180-0.43+math.pi)
        xd = x+2.64*math.cos(yaw*math.pi/180+0.43+math.pi)
        yd = y+2.64*math.cos(yaw*math.pi/180+0.43+math.pi)
        return xa,ya,xb,yb,xc,yc,xd,yd
    
    def _sigmoid(self,x,theta):
        return 2./(1+math.exp(-theta*x))-1

        

        


