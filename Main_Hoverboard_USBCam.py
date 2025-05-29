############################################################################################
## Imports
############################################################################################
import time
import numpy as np
import serial
import struct
import ctypes
import pygame
import RPi.GPIO as GPIO
import collections
import busio
import digitalio
import board
import math
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import cv2
import mediapipe as mp
from flask import Flask, Response
import threading
if tuningPID:
    from Plotter import PlottingSession

############################################################################################
## States
############################################################################################
tuningPID = False
receive_feedback = False
usedistancesensor = True
usecamera = True
whiteline = True

############################################################################################
## PID Values
############################################################################################
P_follow = 0.15
D_follow = 0.05
I_follow = 0

P_manual = 2.2
D_manual = 0.08
I_manual = 0

P_self_driving = 0.007
D_self_driving = 0.0007
I_self_driving = 0

max_steer_PID = 100

############################################################################################
## Initialize Mode Variables
############################################################################################
mode = 'idle'

# Manual Control
manual_max_speed = 120      # RPM, wheel radius is 8cm so 2*pi*0.08*RPM/60 is speed in m/s
                            # 50 RPM is 0.42 m/s so 1.51 km/h
                            # 100 RPM is 0.84 m/s so 3.02 km/h
                            # 150 RPM is 1.26 m/s so 4.53 km/h
manual_max_steer = 50       # Steer changes the ratio between the 2 wheels based on RPM

# Self_driving
self_driving_speed = 40
calibration = False
previous_calibration_state = False
self_driving_state = False
previous_self_driving_state = False
abrubt_stop_self_driving = False

# Acceleration Control
max_acceleration = 40 # RPM/s
last_speed = 0
last_time_acc = time.time()

############################################################################################
## Initialize Sensor Connections and Variables
############################################################################################
# IR Sensor Array Connection (SPI) and Calibration Variables from txt file
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = digitalio.DigitalInOut(board.CE0)
mcp = MCP.MCP3008(spi, cs)
channels = [AnalogIn(mcp, getattr(MCP, f'P{i}')) for i in range(8)]

calibrated_min = []
calibrated_max = []
IR_calibration_counter = 0
with open('Sensor_calibration_values.txt', 'r') as file:
    for line in file:
        min_val, max_val = map(float, line.strip().split(','))
        calibrated_min.append(int(min_val))
        calibrated_max.append(int(max_val))
last_pos = collections.deque(maxlen=3)

# Playstation Controller Connection and Variables
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

button_hold_time = 1
button_press_times = {
    'idle': None,
    'manual_control': None,
    'IMU_test_control': None,
    'self_driving': None,
    'calibration': None,
    'driving self_driving': None
}

# Distance Sensor GPIO Connection and Variables
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
trigPin = 23
echoPin = 24
GPIO.setup(trigPin, GPIO.OUT)
GPIO.setup(echoPin, GPIO.IN)
start_brake_threshold = 20
stop_threshold = 15
distance_history = collections.deque(maxlen=10)

# Buzzer GPIO Connection
buzzerpin = 21
GPIO.setup(buzzerpin, GPIO.OUT)
GPIO.output(buzzerpin,0)
BeeperTimerCamera = 0

# Hoverboard and Sensorboard Connection (serial, UART)
ser_hoverboard = serial.Serial(port='/dev/ttyAMA2', baudrate=115200, timeout=0.1)  
ser_sensorboard = serial.Serial(port='/dev/ttyAMA3', baudrate=115200, timeout=0.1) 

# Hoverboard and IMU Sensorboard Variables
stop = b'\xcd\xab\x00\x00\x00\x00\xcd\xab'
ser_hoverboard.write(stop) 
ser_hoverboard.write(0)
FeedbackPacket = collections.namedtuple('FeedbackPacket', [
    'start', 'cmd1', 'cmd2', 'speedR_meas', 'speedL_meas',
    'batVoltage', 'boardTemp', 'cmdLed', 'checksum'
])

threshold_no_connection = 30
no_connection_counter = 0
virtual_yaw = 0

# Camera Connection (USB) and Variables, OPENCV setup, MediaPipe Model setup, Flask setup
if usecamera:
    STREAM_EVERY_NTH_FRAME = 25
    PROCESS_POSE_EVERY_NTH_FRAME = 2
    current_frame = None
    frame_lock = threading.Lock()
    TORSO_BOX_HEIGHT_GLOBAL = 0
    TORSO_BOX_CENTER_X_GLOBAL = 0
    Previous_TORSO_BOX_HEIGHT_GLOBAL = 0
    previous_TORSO_BOX_CENTER_X_GLOBAL = 0

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=0)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    app = Flask(__name__)


############################################################################################
## Camera Function Definitions For Streaming and Running Flask in a Thread
############################################################################################
def generate_frames():
    # Generate frames to send over html
    global current_frame
    frame_counter = 0 
    while True:
        frame_to_encode = None
        send_this_frame = False
        if current_frame is not None:
            with frame_lock:
                frame_counter += 1
                if frame_counter >= STREAM_EVERY_NTH_FRAME:
                    frame_to_encode = current_frame.copy() 
                    send_this_frame = True
                    frame_counter = 0 
        if send_this_frame and frame_to_encode is not None:
            # Convert frame to lower quality jpeg for better data transfer over the wifi connection with the pi
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, buffer = cv2.imencode('.jpg', frame_to_encode, encode_param)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
             time.sleep(0.001)

if usecamera:
    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/')
    def index():
        return """
            <html>
            <body>
                <h1>Live Pose Tracking</h1>
                <img src="/video_feed" width="640" height="480">
            </body>
            </html>
        """

    def run_flask():
        # Flask is the service for html streaming
        app.run(host='0.0.0.0', port=5000)

    # Open flask in separate thread to not limit main loop speed
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

############################################################################################
## General Function Definitions
############################################################################################
def camera_loop():
    # Function contains person recognition data processing and drawing on the frames captured by the camera
    global current_frame, TORSO_BOX_HEIGHT_GLOBAL, TORSO_BOX_CENTER_X_GLOBAL
    frame_counter_pose = 0
    last_known_pose_results = None 
    MIN_LANDMARKS_FOR_VALID_POSE = 10 

    while True:
        # Capture frame
        ret, frame_orig = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_counter_pose += 1
        frame_clean = frame_orig.copy()
        frame_display = frame_orig
        height, width, _ = frame_display.shape 

        if frame_counter_pose >= PROCESS_POSE_EVERY_NTH_FRAME:
            frame_counter_pose = 0 
        
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            current_cycle_results = pose.process(frame_rgb) 
            found_valid_human_this_cycle = False 

            # Check if the enough landmarks are detected to consider the person on the frame valid
            if current_cycle_results.pose_landmarks:
                num_detected_landmarks = sum(1 for lm in current_cycle_results.pose_landmarks.landmark if lm.visibility > 0.5)
                if num_detected_landmarks >= MIN_LANDMARKS_FOR_VALID_POSE:
                    found_valid_human_this_cycle = True
            
            if found_valid_human_this_cycle:
                last_known_pose_results = current_cycle_results
            else:
                last_known_pose_results = None 
        
        # Drawing the landmarks on the frame
        if last_known_pose_results and last_known_pose_results.pose_landmarks:
            landmarks = last_known_pose_results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(
                frame_display,
                last_known_pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Determine torso box
            torso_landmark_indices = [11, 12, 23, 24]
            torso_x_coords = []
            torso_y_coords = []
            all_key_landmarks_visible = True
            for torso_index in torso_landmark_indices:
                if torso_index < len(landmarks): 
                    lm = landmarks[torso_index]
                    if lm.visibility > 0.5:
                        torso_x_coords.append(lm.x * width)
                        torso_y_coords.append(lm.y * height)
                    else:
                        all_key_landmarks_visible = False
                        break
                else:
                    all_key_landmarks_visible = False
                    break
            
            # Draw torso box
            if torso_x_coords and torso_y_coords and all_key_landmarks_visible:
                min_x_torso, max_x_torso = int(min(torso_x_coords)), int(max(torso_x_coords))
                min_y_torso, max_y_torso = int(min(torso_y_coords)), int(max(torso_y_coords))
                TORSO_BOX_HEIGHT_GLOBAL = max_y_torso - min_y_torso
                TORSO_BOX_CENTER_X_GLOBAL = (min_x_torso + max_x_torso) / 2.0 
                cv2.rectangle(frame_display, (min_x_torso, min_y_torso), (max_x_torso, max_y_torso), (255, 0, 255), 2) 
                cv2.putText(frame_display, f"Torso H:{TORSO_BOX_HEIGHT_GLOBAL} Cx:{TORSO_BOX_CENTER_X_GLOBAL:.0f}",
                            (min_x_torso, min_y_torso - 5 if min_y_torso > 5 else min_y_torso + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
            if TORSO_BOX_HEIGHT_GLOBAL <= 45:
                frame_display = frame_clean.copy()
        else:
            TORSO_BOX_HEIGHT_GLOBAL = 0
            TORSO_BOX_CENTER_X_GLOBAL = 0
  
        with frame_lock:
            current_frame = frame_display.copy() 

def read_feedback():
    # Reading the feedback of the hoverboard over serial connection
    expected_bytes = 18 
    raw_data = ser_hoverboard.read(expected_bytes * 2) 
    for i in range(len(raw_data) - expected_bytes + 1):
        chunk = raw_data[i:i+expected_bytes]
        if len(chunk) == expected_bytes:
            start_frame = struct.unpack('<H', chunk[0:2])[0]
            if start_frame == 0xABCD:
                try:
                    unpacked = struct.unpack('<HhhhhhHHH', chunk)
                    feedback = FeedbackPacket(*unpacked)
                    calc_checksum = feedback.start ^ feedback.cmd1 ^ feedback.cmd2 ^ feedback.speedR_meas ^ \
                                    feedback.speedL_meas ^ feedback.batVoltage ^ feedback.boardTemp ^ feedback.cmdLed
                    if feedback.checksum == calc_checksum:
                        return feedback
                except struct.error as e:
                    continue
    return None

def check_num(num):
    if num < 0:
        num_out = ctypes.c_ushort(num).value
    else:
        num_out = num
    return num_out

def pack(steer, speed):
    # Make serial data package of speed and steer inputs to hoverboard to send over serial data connection
    start_frame = 43981
    u_steer = check_num(steer)
    u_speed = check_num(speed)
    package = struct.pack("<HHHH", start_frame, u_steer, u_speed, (start_frame ^ u_steer ^ u_speed))
    return package

def get_sensor_board():
    # Read the data of the IMU sensorboard
    global no_connection_counter
    global setpoint
    global previous_yaw
    data = ser_sensorboard.readline().decode('utf-8', errors='ignore').strip().split(' ')
    parsed_data = parse_data_sensorboard(data)
    if parsed_data:
        yaw = parsed_data['Yaw']
        no_connection_counter = 0
        return yaw
    # If no data received, the sensorboard is not initialized. An 'e' command has to be written to activate the angle data to be sent.
    elif no_connection_counter >= threshold_no_connection:
        ser_sensorboard.write(b'e')
        no_connection_counter = 0
        print("Waiting for IMU initialization")
        setpoint = 0
        previous_yaw = " "
        for i in range(200):
            yaw = get_sensor_board()
        print("IMU Initialized")
    else:
        no_connection_counter += 1
    return False

def parse_data_sensorboard(data):
    # Extract the angle data from the IMU data package received
    try:
        if 'Roll' in data[0]:
            values = {}
            for item in data:
                if ':' in item:
                    key, value = item.split(':')
                    values[key] = round(float(value)/100, 1)
            if 'Roll' in values and 'Pitch' in values and 'Yaw' in values:
                return values
    except (ValueError, IndexError) as e:
        print(f"Error parsing data: {e}")
    return False

def check_mode(button, mode_name):
    # Check if a button is pressed long enough to change modes, then beep.
    global mode
    if button >= 0.8: 
        if mode == mode_name:
            button_press_times[mode_name] = None
        elif button_press_times[mode_name] is None:
            button_press_times[mode_name] = time.time() 
        elif time.time() - button_press_times[mode_name] >= button_hold_time:
            mode = mode_name 
            print(mode, 'mode')
            GPIO.output(buzzerpin, 1) 
            time.sleep(0.1)
            GPIO.output(buzzerpin,0)
            time.sleep(0.1)
            GPIO.output(buzzerpin, 1)
            time.sleep(0.1)
            GPIO.output(buzzerpin,0)
    else:
        button_press_times[mode_name] = None

def PID_controller(setpoint, measured_value, P, I, D):
    # PID controller setup
    global previous_error, integral, last_time
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time
    error = setpoint - measured_value

    integral += error * dt
    derivative = (error - previous_error) / dt
    previous_error = error

    output = -( P * error + I * integral + D * derivative)
    max_output = max_steer_PID
    output = max(min(output, max_output), -max_output)
    return int(output)

def calculate_virtual_yaw(yaw):
    # Function to keep track of a 'virtual yaw' variable, as the IMU values are not on a continuous scale
    global previous_yaw
    global virtual_yaw
    if isinstance(previous_yaw, str):
        previous_yaw = yaw
        return 0
    delta_yaw = yaw-previous_yaw
    if delta_yaw >= 180:
        virtual_yaw = virtual_yaw + delta_yaw - 360
    elif delta_yaw <= -180:
        virtual_yaw = virtual_yaw + delta_yaw + 360
    else:
        virtual_yaw += delta_yaw
    previous_yaw = yaw
    return round(virtual_yaw, 1)
        
def read_distancesensor():
    # Send an ultrasonic signal and wait till receiving it
    GPIO.output(trigPin, 0)
    time.sleep(2E-6)
    GPIO.output(trigPin, 1)
    time.sleep(10E-6)
    GPIO.output(trigPin, 0)

    timeout = 0.05 
    start_time = time.time()

    while GPIO.input(echoPin) == 0:
        if time.time() - start_time > timeout:
            return None
    echoStartTime = time.time()
    while GPIO.input(echoPin) == 1:
        if time.time() - echoStartTime > timeout:
            return None
    echoStopTime = time.time()

    # Calculate distance travelled using speed of sound and time travelled by ultrasonic signal
    pingTravelTime = echoStopTime - echoStartTime
    distance_cm = (pingTravelTime * 34444) / 2  
    return distance_cm

def check_collision(speed, distancesensor):
    # Keep track of the latest readings of the distancesensor
    if distancesensor is not None:
        distance_history.append(distancesensor)
    if len(distance_history) < 3:
        return speed
    
    # To get rid of noise/outliers, use the average of the minimum 3 values of the 10 latest readings
    # Then proportionally reduce speed for certain distance and stop at stop threshold
    three_smallest = sorted(distance_history)[:3]
    avg_distance = sum(three_smallest) / 3
    if avg_distance < stop_threshold and speed <= 0:
        speed = 0
    elif avg_distance < start_brake_threshold and speed <= 0:
        speed = speed * ((avg_distance - stop_threshold) / (start_brake_threshold - stop_threshold))
    return speed

def Read_IR_sensor():
    # Over SPI, read the analog IR sensor values of all 8 sensors
    values = [channel.value for channel in channels]
    return values

def IR_calibration(values):
    # Update the calibration values in the txt file when calibrating
    global calibrated_min, calibrated_max
    for i in range(8):
        calibrated_min[i] = min(calibrated_min[i], values[i])
        calibrated_max[i] = max(calibrated_max[i], values[i])
    with open('Sensor_calibration_values.txt', 'w') as file:
        for i in range(8):
            file.write(f"{calibrated_min[i]},{calibrated_max[i]}\n")
        
def read_line_position(values):
    # Determine the line position using the IR sensor readings and calibration values, one for white line and one for black line
    global calibrated_min, calibrated_max, last_pos
    if not whiteline:
        total = 0
        weighted_total = 0
        last_pos_avg = sum(last_pos)/len(last_pos) if last_pos else None

        # If all values are low, sensor lost the line, keep correcting to side of sensor where line was lost.
        if all((values[i] < (calibrated_min[i] + calibrated_max[i]) / 2)*1.2 for i in range(8)):
            if last_pos_avg == None:
                return 8000
            pos = 0 if last_pos_avg < 3500 else 7000
            last_pos.append(pos) 
            return pos 

        # If all values are high, tape stop detected 
        if all(values[i] > ((calibrated_min[i] + calibrated_max[i]) / 2)*1.4 for i in range(8)):
            return 9000 

        # If line detected, calculate line position (weighted average of all values scaled from 0 to 1000)
        for i in range(8):
            min_val = calibrated_min[i]
            max_val = calibrated_max[i]
            val = values[i]
            if max_val != min_val:
                level = (val - min_val) * 1000 // (max_val - min_val)
            else:
                level = 0
            
            level = max(0, min(1000, level)) 
            position = i * 1000
            total += level
            weighted_total += level * position
            
        if total == 0:
            return 0 if last_pos_avg < 3500 else 7000

        pos = weighted_total // total
        last_pos.append(pos)
        return pos
    
    if whiteline:
        # See black line comments, same but opposite logic
        total = 0
        weighted_total = 0
        last_pos_avg = sum(last_pos)/len(last_pos) if last_pos else None

        if all((values[i] > (calibrated_min[i] + calibrated_max[i]) / 2) * 0.8 for i in range(8)):
            if last_pos_avg is None:
                return 8000
            pos = 0 if last_pos_avg < 3500 else 7000
            last_pos.append(pos)
            return pos

        if all(values[i] < ((calibrated_min[i] + calibrated_max[i]) / 2) * 0.6 for i in range(8)):
            return 9000

        for i in range(8):
            min_val = calibrated_min[i]
            max_val = calibrated_max[i]
            val = values[i]
            if max_val != min_val:
                level = (max_val - val) * 1000 // (max_val - min_val)
            else:
                level = 0

            level = max(0, min(1000, level))
            position = i * 1000
            total += level
            weighted_total += level * position

        if total == 0:
            return 0 if last_pos_avg < 3500 else 7000

        pos = weighted_total // total
        last_pos.append(pos)
        return pos

def check_self_driving_buttons(button, mode_name):
    # Check if a button in self driving mode is pressed long enough to either start driving or calibrating, then beep
    global calibration, self_driving_state
    if button >= 0.8:
        if button_press_times[mode_name] is None:
            button_press_times[mode_name] = time.time()
        elif time.time() - button_press_times[mode_name] >= button_hold_time:
            if mode_name == 'calibration' and calibration:
                pass
            elif mode_name == 'calibration' and  not calibration:
                calibration = True
                print('calibration started')
            elif mode_name == 'driving self_driving' and self_driving_state:
                self_driving_state = False
                print('driving self_drivingly stopped')
            elif mode_name == 'driving self_driving' and not self_driving_state:
                self_driving_state = True
                print('driving self_drivingly started')
            
            GPIO.output(buzzerpin, 1)
            time.sleep(0.1)
            GPIO.output(buzzerpin,0)
            button_press_times[mode_name] = None
    else:
        button_press_times[mode_name] = None 
        
def check_max_acceleration(speed,last_speed):
    # Limit the speed to defined maximum acceleration
    global last_time_acc
    current_time_acc = time.time()
    dt_acc = current_time_acc-last_time_acc
    last_time_acc=current_time_acc
    speed_diff = speed - last_speed
    max_speed_change = max_acceleration * dt_acc
    if abs(speed_diff) <= max_speed_change:
        return speed
    elif speed <0 and last_speed>0:
        return last_speed - 3*max_speed_change
    elif speed>0 and last_speed<0:
        return last_speed + 3*max_speed_change
    else:
        return last_speed + max_speed_change if speed_diff > 0 else last_speed-max_speed_change

############################################################################################
## Mode function definitions, main functions that are done for each main loop cycle, depending on active mode
############################################################################################
def idle():
    # In idle mode, always stationary
    angle = 0
    speed = 0
    return angle,speed

def follow():
    # In follow mode, PID control for keeping the center of the torso in the middle of the frame. Speed is regulated with height of torso
    global integral, previous_error, last_time

    # PID control for steering, middle of torso
    if TORSO_BOX_CENTER_X_GLOBAL > 0 and previous_TORSO_BOX_CENTER_X_GLOBAL<=0:
        integral = 0
        previous_error = 0
        last_time =time.time()
    if TORSO_BOX_CENTER_X_GLOBAL > 0:
        angle = PID_controller(170, TORSO_BOX_CENTER_X_GLOBAL, P_follow, I_follow, D_follow)
    else:
        angle = 0
    angle = min(max(angle, -30),30)

    # Speed control, height of torso, function for scaling speed over a range with peak somewhere in the middle of this range
    x_start = 50
    x_peak = 80
    x_end = 120
    if TORSO_BOX_HEIGHT_GLOBAL <= x_peak:
        scale = (TORSO_BOX_HEIGHT_GLOBAL - x_start) / (x_peak - x_start)
    else:
        scale = (x_end - TORSO_BOX_HEIGHT_GLOBAL) / (x_end - x_peak)
    w = 0.5 * (1 + np.cos(np.pi * (1 - scale)))

    if TORSO_BOX_HEIGHT_GLOBAL < x_start:
        speed = 0
    elif TORSO_BOX_HEIGHT_GLOBAL < x_end:
        speed = -50 * w -20
    else:
        speed = 0

    # Add data to plot session for plotting the PID controller results
    if tuningPID:
        plotter.add_data(TORSO_BOX_CENTER_X_GLOBAL, 170, angle)
    return angle, speed

def manual_control(drivejoystick, steerjoystick):
    # Joysticks used to scale the speed and angle 
    if drivejoystick < -0.2:
        speed = manual_max_speed * drivejoystick
    elif drivejoystick > 0.2:
        speed = manual_max_speed * drivejoystick
    else:
        speed = 0

    if steerjoystick < -0.2:
        angle = int(manual_max_steer * steerjoystick)
    elif steerjoystick > 0.2:
        angle = int(manual_max_steer * steerjoystick)
    else:
        angle = 0
    return angle,speed    

def IMU_test_control():
    # Mode for testing the velocity flash taken for granted
    global setpoint
    global virtual_yaw
    # Change angle setpoint with steer joystick
    yaw = get_sensor_board()
    if yaw:
        virtual_yaw = calculate_virtual_yaw(yaw) 
        print('---------------------------------------------')       
        print(f'IMU: {virtual_yaw}')
        
        if steerjoystick < -0.2 or steerjoystick > 0.2:
            setpoint -= steerjoystick * 2
        print(f'Setpoint: {round(setpoint,1)}')
        print('---------------------------------------------') 

        if drivejoystick < -0.2:
            speed = manual_max_speed * drivejoystick
        elif drivejoystick > 0.2:
            speed = manual_max_speed * drivejoystick
        else:
            speed = 0

        # Determine steering with PID control comparing IMU measurement to angle setpoint 
        angle = PID_controller(round(setpoint,1), virtual_yaw, P_manual, I_manual, D_manual)
        
        # Add data to plot session for plotting the PID controller results
        if tuningPID:
            plotter.add_data(virtual_yaw, round(setpoint,1), angle)
        return angle, speed
    else:
        angle = 0
        speed = 0        
        return angle, speed

def self_driving():
    # In this mode, nothing happens yet untill the user switches to calibration or self-driving state
    global abrubt_stop_self_driving, integral, previous_error, last_time, calibration, self_driving_state, calibrated_max, calibrated_min, previous_calibration_state, previous_self_driving_state, IR_calibration_counter
    if not self_driving_state:
        check_self_driving_buttons(joystick.get_button(11), 'calibration')
    if not calibration:
        check_self_driving_buttons(joystick.get_button(12), 'driving self_driving')

    if calibration and calibration != previous_calibration_state:
        calibrated_min = [65535] * 8
        calibrated_max = [0] * 8
        IR_calibration_counter = 0
    elif not calibration and calibration != previous_calibration_state:
        GPIO.output(buzzerpin, 1)
        time.sleep(0.1)
        GPIO.output(buzzerpin,0)
        
    if self_driving_state and self_driving_state != previous_self_driving_state:
        previous_error = 0
        integral = 0
        last_time = time.time()
        last_pos.clear()
        if tuningPID:
            plotter.start_collection()
    elif not self_driving_state and self_driving_state != previous_self_driving_state and tuningPID:
        plotter.stop_collection_and_plot()
        
    if not self_driving_state and self_driving_state != previous_self_driving_state and abrubt_stop_self_driving:
        GPIO.output(buzzerpin, 1)
        time.sleep(0.2)
        GPIO.output(buzzerpin,0)
        time.sleep(0.2)
        GPIO.output(buzzerpin, 1)
        time.sleep(0.2)
        GPIO.output(buzzerpin,0)
        abrubt_stop_self_driving = False
    previous_calibration_state = calibration
    previous_self_driving_state = self_driving_state

    # While calibrating, move left and right to make sure all sensor see the line and background surface
    if calibration:
        IR_readings = Read_IR_sensor()
        IR_calibration(IR_readings)
        IR_calibration_counter += 1
        if IR_calibration_counter <= 30:
            return 15,0
        elif IR_calibration_counter <= 40:
            return 0,0
        elif IR_calibration_counter <= 150:
            return -15,0
        elif IR_calibration_counter <= 160:
            return 0,0
        elif IR_calibration_counter <= 270:
            return 15, 0
        elif IR_calibration_counter <= 280:
            return 0,0
        elif IR_calibration_counter <= 390:
            return -15, 0
        elif IR_calibration_counter <= 400:
            return 0,0
        elif IR_calibration_counter <= 510:
            return 15, 0
        elif IR_calibration_counter <= 520:
            return 0,0
        elif IR_calibration_counter <= 630:
            return -15,0
        elif IR_calibration_counter <= 640:
            return 0,0
        elif IR_calibration_counter <= 680:
            return 15,0
        else:
            calibration = False
            print('IR sensor calibration done')
            return 0,0
            
    # While self-driving, speed input is a constant and the steering value is PID by comparing line position to desired line position (middle of the sensor at 3500)
    if self_driving_state:
        IR_readings = Read_IR_sensor()
        line_position = read_line_position(IR_readings)
        print(f'Line Position: {line_position}')
        if line_position < 7500:
            angle = -PID_controller(3500, line_position, P_self_driving, I_self_driving, D_self_driving)
        elif line_position == 8000:
            print('No line detected when activated')
            self_driving_state = False
            abrubt_stop_self_driving = True
            return 0,0
        elif line_position == 9000:
            print('Tape stop detected')
            self_driving_state = False
            abrubt_stop_self_driving = True
            return 0,0
        speed = -self_driving_speed
        
        # Add data to plot session for plotting the PID controller results
        if tuningPID:
            plotter.add_data(line_position, 3500, angle)
    else: 
        angle = 0
        speed = 0
    return angle,speed


############################################################################################
## Start Plotting Session when using plotter and Camera Processing Thread when using camera
############################################################################################
if tuningPID:
    plotter = PlottingSession()
    
# Start cameraprocessing in a seprate thread so the main loop is not limited by this relatively heavy task
if usecamera:
    camera_thread = threading.Thread(target=camera_loop, name="CameraThread")
    camera_thread.start()
    
############################################################################################
## Main Loop
############################################################################################
try:
    while True:
        # Update playstation controller
        pygame.event.pump() 
        drivejoystick = joystick.get_axis(1)
        steerjoystick = joystick.get_axis(3)
        
        # Update mode
        previous_mode = mode
        if usecamera:
            check_mode(joystick.get_button(9), 'follow')
        else:
            check_mode(joystick.get_button(9), 'idle')
        check_mode(joystick.get_axis(2), 'manual_control')
        check_mode(joystick.get_axis(5), 'IMU_test_control')
        check_mode(joystick.get_button(8), 'self_driving')
        
        ## Mode Change Settings/Initializations
        if mode != previous_mode and mode == 'IMU_test_control':
            if tuningPID:
                plotter.start_collection()
            previous_yaw = " "
            setpoint = 0
            start_time_setpoint = time.time()
            previous_error = 0
            integral = 0
            last_time = time.time()
        elif mode != previous_mode and previous_mode == 'IMU_test_control' and tuningPID:
            plotter.stop_collection_and_plot()
            
        if mode != previous_mode and mode == 'follow':
            previous_error = 0
            integral = 0
            last_time = time.time()            
            if tuningPID:
                plotter.start_collection()
        elif mode != previous_mode and previous_mode == 'follow' and tuningPID:
            plotter.stop_collection_and_plot()

        if mode != previous_mode and previous_mode == 'self_driving':
            self_driving_state = False
            calibration = False

        # Call mode function 
        if mode == 'idle':
            angle, speed = idle()
        elif mode == 'follow':
            angle, speed = follow()
        elif mode == 'manual_control':
            angle, speed = manual_control(drivejoystick, steerjoystick)
        elif mode == 'IMU_test_control':
            angle, speed = IMU_test_control()
        else:
            angle, speed = self_driving()
        
        # Receive feedback if True
        if receive_feedback:
            feedback = read_feedback()
            if feedback:
                print(f"[FEEDBACK] R: {feedback.speedR_meas}, L: {feedback.speedL_meas}, V: {feedback.batVoltage}, T: {feedback.boardTemp}")

        # Correct for max acceleration
        speed = check_max_acceleration(speed, last_speed)
        
        # Collision avoidance
        if usedistancesensor:
            distancesensor = read_distancesensor()
            speed = check_collision(speed, distancesensor)
        
        # Camera person detection adjust speed and beeping
        beepertime = time.time()
        if usecamera and speed < 0 and mode == 'self_driving' and TORSO_BOX_HEIGHT_GLOBAL >= 80:
            speed = -0.5 * self_driving_speed
            if TORSO_BOX_HEIGHT_GLOBAL >=120:
                speed = 0
            if BeeperTimerCamera < 0.5:
                GPIO.output(buzzerpin, 1)
            elif BeeperTimerCamera < 1.5:
                GPIO.output(buzzerpin, 0)
            else:
                BeeperTimerCamera = 0
            BeeperTimerCamera += beepertime-last_beepertime

        if usecamera and (TORSO_BOX_HEIGHT_GLOBAL < 80 or mode != 'self_driving'):
            GPIO.output(buzzerpin,0)
            
        if usecamera:
            last_beepertime = beepertime
            Previous_TORSO_BOX_HEIGHT_GLOBAL = TORSO_BOX_HEIGHT_GLOBAL
            previous_TORSO_BOX_CENTER_X_GLOBAL = TORSO_BOX_CENTER_X_GLOBAL
        
        last_speed = speed

        time.sleep(0.001) 

        # Communication Hoverboard
        if speed >= 0:
            speed = math.ceil(speed)
        elif speed < 0:
            speed = math.floor(speed)
        ser_hoverboard.write(pack(angle, speed))
        ser_hoverboard.write(0)
finally:
    GPIO.cleanup()


