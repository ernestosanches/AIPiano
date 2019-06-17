from driver import *


import csv
import time
import math
import sys
from collections import deque
import numpy as np
import pygame


import serial
from glob import glob

import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Sensor:
    def __init__(self):
        self.min = 0
        self.max = 1
        self.sigma = 0
        self.samples = np.array([])
        self.buffer = []
        self.is_calibrated = False
        self.name = 'unknown'
    
    def calibrate_noise(self, start, end):
        self.sigma= np.std(self.samples[start:end])
        print("Noise:", self.sigma)

    def calibrate_values(self, start, end, reset=True):
        self.min = 0.999 * np.quantile(self.samples[start:end], 0.01)
        self.max = 1.001 * np.quantile(self.samples[start:end], 0.99)
        self.is_calibrated = True
        self.samples = self.scale(self.samples)
        print("Range:", self.min, self.max)

    def reset(self):
        self.samples = np.array([])        
        
    def scale(self, data):
        if self.is_calibrated:
            data_scaled = (np.asarray(data) - self.min ) / (self.max - self.min)
            return np.clip(data_scaled, -10, 10)
        else:
            return data
        
    def read_samples(self, ):
        self.buffer = []
    
    def save_samples(self, needed_samples):
        if self.buffer:
            current_samples = len(self.buffer)
            new_data = np.interp(
                    np.linspace(0, 1, needed_samples),
                    np.linspace(0, 1, current_samples),
                    self.scale(self.buffer))
        else:
            last_value = 0 if len(self.samples) == 0 else self.samples[-1]
            new_data = np.full(needed_samples, last_value)
        self.samples = np.concatenate((self.samples, new_data))
        self.buffer = []

import mido

class MidiSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.name = 'midi'
        device_names = mido.get_input_names()
        if device_names:
            self.device = mido.open_input(mido.get_input_names()[0])
        else:
            self.device = None
            
    def read_samples(self):
        self.buffer = []
        for msg in self.device.iter_pending():
            #print("MIDI:", msg)
            self.buffer.append(msg.velocity)
            
    def calibrate_noise(self, start, end):
        # midi is deterministic, just there are only 127 values
        self.sigma = 1/127.0 
        
    def calibrate_values(self, start, end, reset=True):
        self.min = 0
        self.max = 127
        self.is_calibrated = True
        self.samples = self.scale(self.samples)
        
class LidarSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.name = 'lidar'
        serial_port = glob('/dev/ttyUSB*')[0]
        lidar = RPLidar(port=serial_port)
        print("Sent RESET command...")
        lidar.reset()
        time.sleep(1)
        model, fw, hw, serial_no = lidar.get_device_info()
        health_status, err_code = lidar.get_device_health()   
        print(
            '''
            ===
        
            Opened LIDAR on serial port {}
            Model ID: {}
            Firmware: {}
            Hardware: {}
            Serial Number: {}
            Device Health Status: {} (Error Code: 0x{:X})
        
            ===
            '''.format(
                serial_port, model, fw, hw, serial_no.hex(),
                health_status, err_code
            )
        ) 
        lidar.force_start_scan()
        self.lidar = lidar
        
    def read_samples(self):
        samples = self.lidar.poll_scan_samples()
        self.buffer = [sample[1] for sample in samples] # distances
        #print("[{}] Read {} samples... Mean: {}".format(
        #    time.time(), len(self.buffer), np.mean(self.buffer)))
  
from msgpack import Unpacker
    
class AccelerometerSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.name = 'gyro'
        serial_port = glob('/dev/ttyACM*')[0]
        ser = serial.Serial(
            port=serial_port,
            baudrate=115200,
            timeout=0.0)
        self.ser = ser
        self.unpacker = Unpacker()
        
    def read_samples(self): 
        self.unpacker.feed(self.ser.read(self.ser.inWaiting()))
        self.buffer = [x for x in self.unpacker if type(x) is float]
        
    def calibrate_values(self, start, end, reset=True):
        self.min = 0
        self.max = 1.001 * np.quantile(self.samples[start:end], 0.99)
        self.is_calibrated = True
        self.samples = self.scale(self.samples)
        print("Range:", self.min, self.max)


from pykalman import KalmanFilter

class Filter:
    def __init__(self, sigma=0, fs=2000):
        #dt = 1 / fs
        #PROCESS_NOISE = 1
        #SENSOR_NOISE = sigma ** 2 
        #G = np.array([[1/6 * dt**3, 1/2* dt**2, dt]]).T * PROCESS_NOISE
        dt = 1 / fs
        SENSOR_NOISE = 1 * sigma ** 2 
        PROCESS_NOISE = 10000
        G = np.array([[1/6 * dt**3, 1/2* dt**2, dt]]).T * PROCESS_NOISE
        A = np.array([[1, dt, 0.5 * dt**2], 
                      [0, 1, 1 * dt],
                      [0, 0, 1]])
        H = np.array([1, 0, 0])
        Q = np.dot(G, G.T)
        self.kf = KalmanFilter(
                transition_matrices=A,
                observation_matrices=H,                  
                transition_covariance=Q,
                observation_covariance=SENSOR_NOISE,
                initial_state_mean = np.array([0,0,0]),
                initial_state_covariance = Q)
        self.state_mean = None
        self.state_cov = self.kf.initial_state_covariance

    def update(self, observations):
        #print("Update:", self.state_mean.shape)
        
        n_observations = len(observations)
        n_states = self.kf.transition_matrices.shape[0]
        
        if self.state_mean is None:
            self.state_mean = np.array([observations[0],0,0])

        self.kf.initial_state_mean = self.state_mean
        self.kf.initial_state_covatiance = self.state_cov
        means, covs = self.kf.filter(observations)
        self.state_mean, self.state_cov = means[-1], covs[-1]
        return means, covs
        ###
        means = np.zeros((n_observations+1, n_states))
        covs = np.zeros((n_observations+1, n_states, n_states))
        covs[0] = self.state_cov
        means[0] = self.state_mean
        for t in range(len(observations)):
            data = observations[t]
            means[t], covs[t] = self.kf.filter_update(
                    means[t-1], covs[t-1], data)
        self.state_mean, self.state_cov = means[-1], covs[-1]
        return means[1:, :], covs[1:, :]
        
class Processor:
    def __init__(self, sensors, fs=2000):
        self.sensors = sensors
        self.fs = fs
        self.stored_samples = 0
        self.WAITING = 0
        self.CALIBRATE_NOISE = 2 * self.fs
        self.CALIBRATE_START = 5 * self.fs
        self.CALIBRATE_END = 10 * self.fs
        self.state = self.WAITING
        self.start_time = time.time()
        self.playing = 0
        self.out_port = mido.open_output(mido.get_output_names()[0])
        self.filter = None # will be initialized after calibration
        self.kalman_means = None
        self.kalman_covs = None
        
    def sensor_fusion(self, needed_samples):
        data = self.sensors[0].samples[-needed_samples:]
        #print("Fusion:",  len(data), data)
        means, covs = self.filter.update(data)
        if self.kalman_means is None:
            self.kalman_means = means
            self.kalman_covs = covs
        else:
            self.kalman_means = np.concatenate(
                    (self.kalman_means, means), axis=0)
            self.kalman_covs = np.concatenate(
                    (self.kalman_covs, covs), axis=0)
            
    def trigger(self, needed_samples):
        position = self.sensors[0].samples[-1]
        if self.playing == 1 and position < 0.45:
            self.playing = 0
            self.out_port.send(mido.Message(
                    'note_off', note=60))
        elif self.playing == 0 and position > 0.55:
            self.playing = 1
            self.out_port.send(mido.Message(
                    'note_on', note=60, velocity=60))
    
    def state_machine(self, needed_samples):
        if self.state == self.WAITING:
            if self.stored_samples > self.CALIBRATE_NOISE:
                print("Noise calibration...")
                self.state = self.CALIBRATE_NOISE
        elif self.state == self.CALIBRATE_NOISE:
            if self.stored_samples > self.CALIBRATE_START:
                for sensor in self.sensors:
                    sensor.calibrate_noise(
                            self.CALIBRATE_NOISE, self.CALIBRATE_START)
                print("Position calibration...")
                self.state = self.CALIBRATE_START
        elif self.state == self.CALIBRATE_START:
            if self.stored_samples > self.CALIBRATE_END:
                for sensor in self.sensors:
                    sensor.calibrate_values(
                            self.CALIBRATE_START, self.CALIBRATE_END)
                    sensor.calibrate_noise(
                            self.CALIBRATE_NOISE, self.CALIBRATE_START)
                # starting over after reset
                self.reset()
                self.filter = Filter(
                        self.sensors[0].sigma, self.fs)                    
                print("Calibration done!")
                self.state = self.CALIBRATE_END
        else:
            # normal operation
            #self.sensor_fusion(needed_samples)
            self.trigger(needed_samples)
            pass

    def reset(self):
        for sensor in self.sensors:
            sensor.reset()
        self.stored_samples = 0
        self.start_time = time.time()
        
    def process(self):
        for sensor in self.sensors:
            sensor.read_samples()
        elapsed = time.time() - self.start_time
        expected_samples = int(np.round(elapsed * self.fs))
        needed_samples = expected_samples - self.stored_samples
        for sensor in self.sensors:
            sensor.save_samples(needed_samples)        
        self.stored_samples += needed_samples
        self.state_machine(needed_samples)


class Grapher:
    def __init__(self, processor, max_len):
        self.processor = processor
        self.max_len = max_len
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-1.1, 1.1)
        self.n_kalman_states = 3
        sensor_names = {0: "Lidar position", 
                        1: "Midi velocity",
                        2: "Gyroscope velocity"
                        }
        #kalman_names = {0: "position est.", 
        #                1:"velocity est.", 
        #                2: "acceleration est."}
        self.lines = [self.ax.plot(np.zeros(max_len), 
                                   label=sensor_names[i])[0] 
                      for i in range(len(processor.sensors))]
        #self.kalman_lines = [self.ax.plot(np.zeros(max_len) + (i+1) / 10,
        #                                  label=kalman_names[i])[0] 
        #              for i in range(self.n_kalman_states)]
        self.fig.legend()

    def update(self):
        for sensor, line in zip(processor.sensors, self.lines):
            samples = sensor.samples[-self.max_len:]
            if len(samples) < self.max_len:
                samples = np.pad(samples, self.max_len - len(samples), 
                                 mode='constant')
            # print(np.mean(samples))
            line.set_ydata(samples)
       
        #if processor.kalman_means is not None:
        #    for i in range(self.n_kalman_states):
        #        samples = processor.kalman_means[-self.max_len:, i]
        #        if len(samples) < self.max_len:
        #            samples = np.pad(samples, self.max_len - len(samples), 
        #                             mode='constant')
        #        #print(np.mean(samples))
        #        self.kalman_lines[i].set_ydata(samples)
        
        
    def animate(self, frame):
        processor.process()        
        self.update()
        #return self.lines + self.kalman_lines
        return self.lines

    def get_animation(self):
        ani = animation.FuncAnimation(self.fig, self.animate,
                                      interval=1, blit=True)
        self.fig.show()
        return ani
            
sensors = [LidarSensor(), MidiSensor(),AccelerometerSensor()]
processor = Processor(sensors)
grapher = Grapher(processor, max_len=4000)   

ani = grapher.get_animation()



from os import path

def save_data(sensors):
    folder = path.expanduser("~/Projects/Sound/AIPiano/data/lidar2/")
    for sensor in sensors:
        filename = path.join(folder, "samples_{}".format(sensor.name))
        np.save(filename, sensor.samples)

#save_data(sensors)
