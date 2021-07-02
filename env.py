
import sys
# sys.path.append('/home/am/OneDrive/project_HMI_lanechange/TD3-based human machine_CARLA/TD3-DRL-human-machine-driving training-scenario-by-CARLA')
sys.path.append('C:\\Users\\RRC1\\Desktop\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.9-py3.7-win-amd64.egg')
sys.path.append('/home/rrc4/Downloads/carla/PythonAPI/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg')


import carla
from carla import ColorConverter as cc

import pygame
import weakref
import collections
import numpy as np
import math
import sys
import cv2
from scipy.interpolate import interp1d

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser


vv_target = 5
position_bench_x = 335.0
position_bench_y = 200.0
waypoint_x_mark = np.array([200,212.5,225,237.5,250])
waypoint_y_mark = np.array([335,336.5,338,336.5,335])
waypoint = interp1d(waypoint_x_mark, waypoint_y_mark,kind='cubic')



class scenario(object):
    def __init__(self,trainable=True, random_spawn=True, pedestrain=False, use_latent=False, no_render=False, frame=25):

        self.count = 0
        self.display = None
        self.client = None
        self.world = None
        self.control = None
        self.ego_vehicle = None
        self.obs1 = None
        self.obs2 = None
        self.obs3 = None
        
        self.observation_size = 45*80
        self.action_size = 1

        self._joystick = None
        self._parser = None
        self._steer_idx = None
        self._throttle_idx = None
        self._brake_idx = None
        self._reverse_idx = None
        self._handbrake_idx = None

        self.steer_history = []
        self.intervene_history = []

        self.intervention = False

        self.collision_sensor = None
        
        # about the camera sensor
        self.camera = None
        self.camera_index = 1
        self.surface = None
        self.camera_output = np.zeros([720,1280,3])
        self.recording = False
        self.Attachment = carla.AttachmentType
        
        self.done = None
        
        self.trainable = trainable
        self.pedestrain = pedestrain
        self.random_spawn = random_spawn
        self.no_render = no_render
        self.use_latent = use_latent
        self.frame = 1/frame

        self.restart()

    def restart(self):
        
        # connect to the client
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(10.0)
        
        
        # build the world
        self.world = self.client.load_world('Town01')
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.frame
        settings.no_rendering_mode = self.no_render
        settings.SynchronousMode = True
        self.world.apply_settings(settings)
        
        
        # initialize the pygame setting
        pygame.init()
        pygame.font.init()
        pygame.joystick.init()
        self.display = pygame.display.set_mode((1280, 720),pygame.HWSURFACE | pygame.DOUBLEBUF)


        # initilize the joystick setting
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
            
        self._parser = ConfigParser()
        self._parser.read('./wheel_config.ini')
        self._steer_idx = int(self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(self._parser.get('G29 Racing Wheel', 'handbrake'))
        
        self.steer_history = []
        self.intervene_history = []
        
        
        # initilize the vehicles (egos and surroundings)
            
        # when training, the positions are fixed
        if self.trainable:

            self.bp_obs1, self.spawn_point_obs1 = self._produce_vehicle_blueprint(1,335.0+3.5,200.0)
            self.obs1 = self.world.spawn_actor(self.bp_obs1,self.spawn_point_obs1)
    
            self.bp_obs2, self.spawn_point_obs2 = self._produce_vehicle_blueprint(1,335.0,200.0+25.0)
            self.obs2 = self.world.spawn_actor(self.bp_obs2,self.spawn_point_obs2)
    
            self.bp_obs3, self.spawn_point_obs3 = self._produce_vehicle_blueprint(1,335.0+3.5,200.0+50.0)
            self.obs3 = self.world.spawn_actor(self.bp_obs3,self.spawn_point_obs3)

            if self.pedestrain:
                self.bp_walker1, self.spawn_point_walker1 = self._produce_walker_blueprint(338.0, 200+np.random.randint(10,15))
                self.bp_walker2, self.spawn_point_walker2 = self._produce_walker_blueprint(np.random.randint(3310,3350)/10, 235)
                self.bp_walker3, self.spawn_point_walker3 = self._produce_walker_blueprint(333.0, 230+np.random.randint(10,20))
                
                self.walker1 = self.world.spawn_actor(self.bp_walker1, self.spawn_point_walker1)
                self.walker2 = self.world.spawn_actor(self.bp_walker2, self.spawn_point_walker2)
                # self.walker3 = self.world.spawn_actor(self.bp_walker3, self.spawn_point_walker3)
                walker1_control = carla.WalkerControl()
                walker1_control.speed = 0.5
                self.walker1.apply_control(walker1_control)
                walker2_control = carla.WalkerControl()
                walker2_control.speed = 0.5
                self.walker2.apply_control(walker2_control)
                # walker3_control = carla.WalkerControl()
                # walker3_control.speed = 0.05
                # self.walker3.apply_control(walker2_control)
            
            if self.random_spawn:
                y_spawn = np.random.randint(200,240)
                x_spwan = waypoint(y_spawn) + (np.random.rand()-0.5)*0.1
                self.bp_ego, self.spawn_point_ego = self._produce_vehicle_blueprint(2,x_spwan,y_spawn)
            else:
                self.bp_ego, self.spawn_point_ego = self._produce_vehicle_blueprint(2,335.0,200.0)
            
            self.ego_vehicle = self.world.spawn_actor(self.bp_ego,self.spawn_point_ego)
            
        # when testing, the position can be modified arbitary
        else:
            self.bp_ego, self.spawn_point_ego = self._produce_vehicle_blueprint(2,335,200.0)
                
            self.ego_vehicle = self.world.spawn_actor(self.bp_ego,self.spawn_point_ego)
    
            self.bp_obs1, self.spawn_point_obs1 = self._produce_vehicle_blueprint(1,335.0+3.5,200.0)
            self.obs1 = self.world.spawn_actor(self.bp_obs1,self.spawn_point_obs1)
            
            self.bp_obs2, self.spawn_point_obs2 = self._produce_vehicle_blueprint(1,335.0,200.0+20.0)
            self.obs2 = self.world.spawn_actor(self.bp_obs2,self.spawn_point_obs2)
            
            self.bp_obs3, self.spawn_point_obs3 = self._produce_vehicle_blueprint(1,335.0+3.5,200.0+50.0)
            self.obs3 = self.world.spawn_actor(self.bp_obs3,self.spawn_point_obs3)

            self.obs1.set_velocity(carla.Vector3D(0,vv_target-5,0))
            self.obs2.set_velocity(carla.Vector3D(0,vv_target-5,0))
            self.obs3.set_velocity(carla.Vector3D(0,vv_target-5,0))
            
            if self.pedestrain:
                self.bp_walker1, self.spawn_point_walker1 = self._produce_walker_blueprint(338.0, 205)
                self.bp_walker2, self.spawn_point_walker2 = self._produce_walker_blueprint(331, 235)
                
                self.walker1 = self.world.spawn_actor(self.bp_walker1, self.spawn_point_walker1)
                self.walker2 = self.world.spawn_actor(self.bp_walker2, self.spawn_point_walker2)
                walker1_control = carla.WalkerControl()
                walker1_control.speed = 0
                self.walker1.apply_control(walker1_control)
                walker2_control = carla.WalkerControl()
                walker2_control.speed = 0
                self.walker2.apply_control(walker2_control)
        
        # initilize the collision sensor
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

        
        # initilize the camera sensors
        # the candidated transform of camera's position: frontal and bird-view
        self.camera_transforms = [
            (carla.Transform(carla.Location(x=-2, z=5), carla.Rotation(pitch=30.0)), self.Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-2, z=5.0), carla.Rotation(pitch=30.0)), self.Attachment.SpringArm)]
        self.camera_transform_index = 1
        # the candidated camera type: rgb and semantic
        self.cameras = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}]
            ]
#        for item in self.cameras:
#            bp_camera = self.world.get_blueprint_library().find(item[0])
#            bp_camera.set_attribute('image_size_x', '1920')
#            bp_camera.set_attribute('image_size_y', '1080')
#            bp_camera.set_attribute('sensor_tick', '0.04')
#            item.append(bp_camera)
        bp_camera = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp_camera.set_attribute('image_size_x', '1280')
        bp_camera.set_attribute('image_size_y', '720')
        bp_camera.set_attribute('sensor_tick', '0.04')
        self.cameras[1].append(bp_camera)
        
        bp_viz_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp_viz_camera.set_attribute('image_size_x', '1280')
        bp_viz_camera.set_attribute('image_size_y', '720')
        bp_viz_camera.set_attribute('sensor_tick', '0.02')
        self.cameras[0].append(bp_viz_camera)
        

        # spawn the camera actor : (by 4 characters: camera type,location,attached unit,attached type)
        if self.camera is not None:
            self.camera.destroy()
            self.surface = None
        self.camera = self.world.spawn_actor(
            self.cameras[1][-1],
            self.camera_transforms[self.camera_transform_index - 1][0],
            attach_to=self.ego_vehicle,
            attachment_type=self.camera_transforms[self.camera_transform_index - 1][1])
        self.viz_camera = self.world.spawn_actor(
            self.cameras[0][-1],
            self.camera_transforms[self.camera_transform_index][0],
            attach_to=self.ego_vehicle,
            attachment_type=self.Attachment.SpringArm)
        # obtain the camera image
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: scenario.parse_image(weak_self, image))
        self.viz_camera.listen(lambda image: scenario.drive_image(weak_self, image))
 
        
        # initilize the control variable for the ego vehicle
        self.control = carla.VehicleControl()
        
        # initilize the velocity variable data structure
        initial_velocity = carla.Vector3D(0,vv_target,0)
        self.ego_vehicle.set_velocity(initial_velocity)
        
        self.count = 0

    
    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))


    def parse_image(weak_self, image):
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
    
    def drive_image(weak_self, image):
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
        return collision_history,flag
    
    
    def run_step(self,action, out_guide=False):

        self.render(self.display)
        
        pygame.display.flip()
        
        self.parse_events()

        human_control = None
        
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]


        if not self.intervention:

            steerCmd = (action)-0.5
            
            self.control.steer = math.tan(1.1 * steerCmd)

        else:

            K1 = 1
            steerCmd = K1 * (2 * jsInputs[self._steer_idx])

            self.control.steer = steerCmd

            human_control = (self.control.steer + 0.5)

                    
        self.intervene_history.append(jsInputs[0])
        
        self.control.throttle = np.clip (vv_target - ((self.ego_vehicle.get_velocity().x)**2 + (self.ego_vehicle.get_velocity().y)**2)**(1./2) ,0,1)

        self.control.brake = 0.0
        self.control.hand_brake = 0

        self.ego_vehicle.apply_control(self.control)
        self.steer_history.append(steerCmd)

        next_states, other_indicators = self.obtain_observation()

        collision = self.get_collision_history()[1]
        finish = (self.ego_vehicle.get_location().y > position_bench_y + 55.0)
        beyond = (self.ego_vehicle.get_location().x < position_bench_x - 1.2) or (self.ego_vehicle.get_location().x > position_bench_x + 4.8)
        fail = collision or beyond
        done = fail or finish

        dis_to_front = other_indicators['state_front']
        dis_to_side = min(other_indicators['state_left'],other_indicators['state_right'])
        dis_to_obs11 = other_indicators['state_corner_11']
        dis_to_obs12 = other_indicators['state_corner_12']
        dis_to_obs21 = other_indicators['state_corner_21']
        dis_to_obs22 = other_indicators['state_corner_22']
        dis_to_obs31 = other_indicators['state_corner_31']
        dis_to_obs32 = other_indicators['state_corner_32']
        
        RIGHT = 1 if self.ego_vehicle.get_location().x < position_bench_x + 1.8 else 0
        r1 = -1*np.square(1-dis_to_front) # front vehicle penalty
        r2 = -2*np.square(1-dis_to_side) # road side penalty
        r3 = - (np.square(1-dis_to_obs11)+np.square(1-dis_to_obs12)+np.square(1-dis_to_obs21)+np.square(1-dis_to_obs22)+np.square(1-dis_to_obs31)+np.square(1-dis_to_obs32))
        r4 = -fail*10 + finish*10
        r5 = (self.ego_vehicle.get_location().y -210 )*0

        r6= -np.float32(abs(self.steer_history[-1]-steerCmd)>0.1)
        r7 = -3*abs(steerCmd)
        # r8 = -0.1* np.square(self.ego_vehicle.get_location().x - 335) if RIGHT else -0.1* np.square(self.ego_vehicle.get_location().x - 338)
        # r8 = -2 if self.intervention else 0
        reward_naive = r1+r2+r3+r4+r5
        
        reward = reward_naive+r6+r7
        reward = np.clip(reward,-10,10)
        
        self.count += 1
        
        yaw = np.arctan(self.ego_vehicle.get_velocity().x/self.ego_vehicle.get_velocity().y) if self.ego_vehicle.get_velocity().y > 0 else 0
        
        scope = {'velocity_y':self.ego_vehicle.get_velocity().y,
                 'velocity_x':self.ego_vehicle.get_velocity().x,
                 'position_y':self.ego_vehicle.get_location().y,
                 'position_x':self.ego_vehicle.get_location().x,
                 'yaw':yaw,
                 'pitch':self.ego_vehicle.get_transform().rotation.pitch,
                 'roll':self.ego_vehicle.get_transform().rotation.roll,
                 'angular_velocity_y':self.ego_vehicle.get_angular_velocity().y,
                 'angular_velocity_x':self.ego_vehicle.get_angular_velocity().x
                 }


        return next_states, human_control, reward, reward_naive, done, scope
    
    def destroy(self):
        actors = [
            self.ego_vehicle,
            self.obs1,
            self.obs2,
            self.obs3,
            self.camera,
            self.viz_camera,
            self.collision_sensor]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def obtain_observation(self):
        ### obtain image-based state space
        # state variable sets
        state_space = self.camera_output[:,:,0]
        state_space = cv2.resize(state_space,(80,45))
        state_space = np.resize(state_space,(45*80,1))
        state_space = np.squeeze(state_space)/255
        
        ### obtain space variables for reward generation
        velocity_self = self.ego_vehicle.get_velocity()
        position_self = self.ego_vehicle.get_location()
        yaw_self = self.ego_vehicle.get_transform().rotation.yaw
        position_obs1 = self.obs1.get_location()
        position_obs2 = self.obs2.get_location()
        position_obs3 = self.obs3.get_location()
        # lateral velocity
        state_vy = velocity_self.x if abs(velocity_self.x) > 1e-2 else 0
        state_vy = 1/(1+math.exp(-state_vy))
        # longitudinal velocity
        state_vx = velocity_self.y if abs(velocity_self.y) > 1e-2 else 0
        # yaw rate
#        state_yaw = np.arctan(self.ego_vehicle.get_velocity().x/self.ego_vehicle.get_velocity().y)
#        state_yaw = (state_yaw + 1)/2
        state_yaw = 0
        # lateral position
        state_y = position_self.x - position_bench_x
        state_y = (state_y+1.5)/7

        ### obtain relative distance information for reward generation
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
        RIGHT = 1 if position_self.x < position_bench_x + 1.8 else 0
        if RIGHT:
            if position_self.y < position_bench_y + 25.0:
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
            if len(self.steer_history)>1:
                if abs(self.steer_history[-2] - self.steer_history[-1]) > 0.02:
                    self.intervention = True
#                elif abs(self.steer_history[-3] - self.steer_history[-1]) < 0.01:
#                    self.intervention = False
            if len(self.intervene_history)>4:
                if abs(self.intervene_history[-4] - self.intervene_history[-1]) < 0.015:
                    self.intervention = False

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

        

        


