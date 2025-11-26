#!/usr/bin/env python3
"""
IMU Data Simulation Sender for IMUCoCo Demo UI

This script loads processed TotalCapture data and sends it as UDP packets
to simulate 4 IMU devices.
"""

import argparse
import socket
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import sys
import os
import threading
from typing import Optional, Callable, Dict, List, Tuple
import pygame
from utils import imu_config
import path_config
from articulate.math import quaternion_to_rotation_matrix, r6d_to_rotation_matrix, rotation_matrix_to_axis_angle, axis_angle_to_quaternion

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Device configuration (matching imucoco_demo_ui.py)
NUM_IMU_DEVICES = 4
SERVER_PORTS = [8001, 8002]
IMU_DEVICE_SOCKET_PORT = [8001, 8001, 8002, 8002]
IMU_DEVICE_TYPE = ['watch', 'phone', 'watch', 'phone']

# UDP configuration
SERVER_IP = '127.0.0.1'
UDP_SOCKET_BUFFER_SIZE = 2048
FPS = 60  # Target FPS for data transmission
FRAME_DURATION = 1.0 / FPS

# Coordinate subscription configuration
COORDINATE_CACHE_COORDS_FILE = "demo_cache_device_coordinates.npy"
COORDINATE_CACHE_SELECTOR_FILE = "demo_cache_attach_device_selector.npy"
COORDINATE_POLL_INTERVAL = 0.5  # Check for coordinate changes every 0.5 seconds


IMU_YUP_TO_ZUP = torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]).float()

class IMUAttachmentMonitor:
    """
    Subscribes to coordinate changes from live_demo.py.
    Monitors cache files for coordinate updates and maps them to TotalCapture joint indices.
    """
    def __init__(self, poll_interval: float = 0.5):
        """
        Initialize coordinate subscriber.
        
        Args:
            poll_interval: Interval in seconds to check for file changes
        """
        self.poll_interval = poll_interval
        
        # Coordinate state
        self.device_coordinates: Optional[np.ndarray] = None
        self.attached_selector: Optional[np.ndarray] = None
        self.last_file_mtime = {}
        
        # Callbacks
        self.coordinate_callbacks: List[Callable] = []
        
        # Threading
        self.monitoring = False
        self.monitor_thread = None
      
    def subscribe(self, callback: Callable[[np.ndarray, np.ndarray], None]):
        """
        Subscribe to coordinate changes.
        
        Args:
            callback: Function called when coordinates change.
                     Signature: callback(coordinates, attached_selector)
        """
        self.coordinate_callbacks.append(callback)
        
    def _load_coordinates_from_cache(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load coordinates from cache files."""
        coords_file = COORDINATE_CACHE_COORDS_FILE
        selector_file = COORDINATE_CACHE_SELECTOR_FILE
        
        try:
            if os.path.exists(coords_file) and os.path.exists(selector_file):
                coordinates = np.load(coords_file)
                selector = np.load(selector_file)
                return coordinates, selector
        except Exception as e:
            print(f"Error loading coordinate cache: {e}")
        
        return None, None
    
    def _check_file_changes(self) -> bool:
        """Check if coordinate cache files have changed."""
        coords_file = COORDINATE_CACHE_COORDS_FILE
        selector_file = COORDINATE_CACHE_SELECTOR_FILE
        
        changed = False
        for file_path in [coords_file, selector_file]:
            if os.path.exists(file_path):
                current_mtime = os.path.getmtime(file_path)
                if file_path not in self.last_file_mtime:
                    self.last_file_mtime[file_path] = current_mtime
                    changed = True
                elif current_mtime > self.last_file_mtime[file_path]:
                    self.last_file_mtime[file_path] = current_mtime
                    changed = True
        
        return changed
    
    def _notify_callbacks(self, coordinates: np.ndarray, selector: np.ndarray):
        """Notify all subscribed callbacks of coordinate changes."""
        for callback in self.coordinate_callbacks:
            try:
                callback(coordinates.copy(), selector.copy())
            except Exception as e:
                print(f"Error in coordinate callback: {e}")
    
    def _monitor_files(self):
        """Monitor coordinate cache files for changes."""
        while self.monitoring:
            try:
                if self._check_file_changes():
                    coordinates, selector = self._load_coordinates_from_cache()
                    if coordinates is not None and selector is not None:
                        # Check if coordinates actually changed
                        if (self.device_coordinates is None or 
                            not np.array_equal(self.device_coordinates, coordinates) or
                            not np.array_equal(self.attached_selector, selector)):
                            
                            self.device_coordinates = coordinates
                            self.attached_selector = selector
                            print(f"Coordinate change detected: {len(coordinates)} devices attached")
                            self._notify_callbacks(coordinates, selector)
                
                time.sleep(self.poll_interval)
            except Exception as e:
                print(f"Error in file monitoring: {e}")
                time.sleep(self.poll_interval)
    
    def start_monitoring(self):
        """Start monitoring for coordinate changes."""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        # Load initial coordinates if available
        coordinates, selector = self._load_coordinates_from_cache()
        if coordinates is not None and selector is not None:
            self.device_coordinates = coordinates
            self.attached_selector = selector
            print(f"Initial coordinates loaded: {len(coordinates)} devices attached")
            self._notify_callbacks(coordinates, selector)
        
        # Start file monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_files, daemon=True)
        self.monitor_thread.start()
        
        print("Coordinate monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring for coordinate changes."""
        self.monitoring = False
        print("Coordinate monitoring stopped")
    


class IMUDataSimulator:
    def __init__(self, data_file_path):
        """
        Initialize the IMU data simulator.
        
        Args:
            data_file_path (str): Path to the processed .pt data file
        """
        self.data_file_path = data_file_path
        self.data = None
        self.current_frame = 0
        self.total_frames = 0
        self.running = False
        
        # UDP sockets for each port
        self.sockets = {}
        self.setup_udp_sockets()
        
        # Coordinate subscription
        self.coordinate_subscriber = IMUAttachmentMonitor(poll_interval=COORDINATE_POLL_INTERVAL)
        self.coordinate_subscriber.subscribe(self._on_imu_attachment_changed)
        
        self.imu_attachment_vertex_ids = [0] * NUM_IMU_DEVICES
        
        # Load and prepare data
        self.all_imu_data = torch.load(self.data_file_path, map_location='cpu')['vimu']['vimu_mesh'].float()
        self.total_frames = self.all_imu_data.shape[0]
        print(f"Loaded data with {self.total_frames} frames")
        print(f"IMU data shape: {self.all_imu_data.shape}")

        self.vertex_coordinates = imu_config.vertex_coordinates
        
    def setup_udp_sockets(self):
        """Setup UDP sockets for data transmission."""
        for port in SERVER_PORTS:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, UDP_SOCKET_BUFFER_SIZE)
                # Enable reuse address (useful for debugging/restarting)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # Keep socket in blocking mode (default) for reliable sending
                self.sockets[port] = sock
                print(f"✓ Setup UDP socket for port {port} (target: {SERVER_IP}:{port})")
            except Exception as e:
                print(f"Failed to setup UDP socket for port {port}: {e}")
    
    def _on_imu_attachment_changed(self, coordinates: np.ndarray, selector: np.ndarray):
        """Callback for IMU attachment changes."""
        imu_attachment_device_indices = [i for i, selected in enumerate(selector) if selected]
        for i, coord in enumerate(coordinates):
            # find closest vertex id 
            vid = np.argmin(np.linalg.norm(self.vertex_coordinates - coord, axis=1))
            device_idx = imu_attachment_device_indices[i]
            self.imu_attachment_vertex_ids[device_idx] = vid
        print(f"IMU attachment changed: {len(coordinates)} devices attached")
        
    def create_motion_message(self, device_idx, frame_data, timestamp):
        """
        Create a motion message in the format expected by the receiver.
        
        Args:
            device_idx (int): Device index
            frame_data (dict): Frame data containing orientation and acceleration
            timestamp (float): Timestamp for the frame
            
        Returns:
            str: Formatted motion message
        """
        imu_ori_r6d = frame_data[0:6]
        imu_acc = frame_data[6:9]

        # convert to z-up coordinate system so it is same as the real data from mobile devices
        # the z-up will still be converted to a y-up coordinate system in the demo server
        imu_ori = r6d_to_rotation_matrix(imu_ori_r6d)
        imu_ori = IMU_YUP_TO_ZUP @ imu_ori @ IMU_YUP_TO_ZUP.T
        imu_acc = IMU_YUP_TO_ZUP @ imu_acc

        imu_quat = axis_angle_to_quaternion(rotation_matrix_to_axis_angle(imu_ori))[0]   # wxyz

        user_id = 'simulation'
        device_type = IMU_DEVICE_TYPE[device_idx]
        
        # Create motion text in the expected format:
        # timestamp user_acc.x user_acc.y user_acc.z grav.x grav.y grav.z gyro.x gyro.y gyro.z mag.x mag.y mag.z attitude.roll attitude.pitch attitude.yaw quat.x quat.y quat.z quat.w
        motion_values = [
            timestamp,                    # timestamp
            imu_acc[0].item(),            # user_acc.x
            imu_acc[1].item(),            # user_acc.y  
            imu_acc[2].item(),            # user_acc.z
            0.0, 0.0, 0.0,                # gravity (not used in simulation)
            0.0, 0.0, 0.0,                # gyro (not used in simulation)
            imu_quat[1].item(),           # quat.x
            imu_quat[2].item(),           # quat.y
            imu_quat[3].item(),           # quat.z
            imu_quat[0].item()            # quat.w
        ]
        motion_text = ' '.join([f"{val:.6f}" for val in motion_values])
        message = f"{user_id};{device_type};motion:{motion_text}"
        
        return message

    def send_init_message(self):
        """Send initial data to all devices."""
        timestamp = time.time()
        for device_index in range(NUM_IMU_DEVICES):
            port = IMU_DEVICE_SOCKET_PORT[device_index]
            if port not in self.sockets or self.sockets[port] is None:
                print(f"✗ Socket not available for device {device_index} on port {port}")
                continue
                
            try:
                user_id = 'simulation'
                device_type = IMU_DEVICE_TYPE[device_index]
                init_message = f"{user_id};{device_type};init"
                sock = self.sockets[port]
                bytes_sent = sock.sendto(init_message.encode('utf-8'), (SERVER_IP, port))
                print(f"✓ Sent init message to device {device_index} ({device_type}) to {SERVER_IP}:{port} ({bytes_sent} bytes)")
            except socket.error as e:
                print(f"✗ Error sending init message to device {device_index} on port {port}: {e}")
            except Exception as e:
                print(f"✗ Unexpected error sending init message to device {device_index}: {e}")
    
    def send_stop_message(self):
        """Send stop message to all devices."""
        for device_index in range(NUM_IMU_DEVICES):
            port = IMU_DEVICE_SOCKET_PORT[device_index]
            if port not in self.sockets or self.sockets[port] is None:
                print(f"✗ Socket not available for device {device_index} on port {port}")
                continue
                
            try:
                user_id = 'simulation'
                device_type = IMU_DEVICE_TYPE[device_index]
                stop_message = f"{user_id};{device_type};stop"
                sock = self.sockets[port]
                bytes_sent = sock.sendto(stop_message.encode('utf-8'), (SERVER_IP, port))
                print(f"✓ Sent stop message to device {device_index} ({device_type}) to {SERVER_IP}:{port} ({bytes_sent} bytes)")
            except socket.error as e:
                print(f"✗ Error sending stop message to device {device_index} on port {port}: {e}")
            except Exception as e:
                print(f"✗ Unexpected error sending stop message to device {device_index}: {e}")

    def send_motion_message(self, frame_idx):
        """Send data for a single frame to attached devices only."""
        timestamp = time.time()
            
        for device_index in range(NUM_IMU_DEVICES):
            port = IMU_DEVICE_SOCKET_PORT[device_index]
            if port not in self.sockets or self.sockets[port] is None:
                print(f"✗ Socket not available for device {device_index} on port {port}")
                continue
            try:
                vertex_id = self.imu_attachment_vertex_ids[device_index]
                frame_data = self.all_imu_data[frame_idx, vertex_id]
                message = self.create_motion_message(device_index, frame_data, timestamp)
                # Send via UDP
                sock = self.sockets[port]
                bytes_sent = sock.sendto(message.encode('utf-8'), (SERVER_IP, port))
                if frame_idx % 600 == 0:
                    print(f"✓ Sent motion message to device {device_index} ({IMU_DEVICE_TYPE[device_index]}) to {SERVER_IP}:{port} ({bytes_sent} bytes)") 
            except socket.error as e:
                print(f"✗ Socket error sending data for device {device_index} on port {port}: {e}")
            except KeyError as e:
                print(f"✗ Key error sending data for device {device_index}: {e}")
            except Exception as e:
                print(f"✗ Error sending data for device {device_index}: {e}")
                
    def run_simulation(self, loop=True, start_frame=0):
        """
        Run the simulation.
        
        Args:
            loop (bool): Whether to loop the data
            start_frame (int): Frame to start from
        """
        self.running = True
        self.current_frame = start_frame
        
        # Start coordinate monitoring if enabled
        if self.coordinate_subscriber:
            self.coordinate_subscriber.start_monitoring()
        
        print(f"Starting simulation at {FPS} FPS")
        print(f"Total frames: {self.total_frames}")
        print(f"Loop mode: {loop}")

        self.send_init_message()

        # Initialize pygame clock for FPS control
        # Initialize pygame (minimal init for clock only, no display needed)
        pygame.init()
        clock = pygame.time.Clock()
        
        try:
            while self.running:
                self.send_motion_message(self.current_frame)
                self.current_frame += 1
                # Handle looping
                if self.current_frame >= self.total_frames:
                    if loop:
                        self.current_frame = 0
                        print("Looping data...")
                    else:
                        print("Simulation completed")
                        break
                
                # Maintain target FPS using pygame clock
                clock.tick(FPS)
                    
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        finally:
            self.stop_simulation()
            
    def stop_simulation(self):
        """Stop the simulation and close sockets."""
        self.running = False
        self.send_stop_message()
        
        # Stop coordinate monitoring
        if self.coordinate_subscriber:
            self.coordinate_subscriber.stop_monitoring()
        
        # Close UDP sockets
        for port, sock in self.sockets.items():
            if sock is not None:
                try:
                    sock.close()
                    print(f"✓ Closed socket for port {port}")
                except Exception as e:
                    print(f"✗ Error closing socket for port {port}: {e}")
        self.sockets.clear()
        print("Simulation stopped")

def main():
    parser = argparse.ArgumentParser(description='IMU Data Simulation Sender')
    parser.add_argument('--data_file',
                        help='Path to the processed .pt data file',
                        default=os.path.join(path_config.parsed_pose_dataset_dir, "TotalCapture", "s1_walking3_Xsens_AuxFields.pt"))
    parser.add_argument('--loop', action='store_true', default=True, 
                       help='Loop the data (default: True)')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Frame to start from (default: 0)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Target FPS (default: 60)')
    
    args = parser.parse_args()
    
    # Update global FPS if specified
    global FPS, FRAME_DURATION
    FPS = args.fps
    FRAME_DURATION = 1.0 / FPS
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} not found")
        sys.exit(1)
        
    # Create and run simulator
    simulator = IMUDataSimulator(args.data_file)
    
    try:
        simulator.run_simulation(loop=args.loop, start_frame=args.start_frame)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        simulator.stop_simulation()

if __name__ == "__main__":
    main()