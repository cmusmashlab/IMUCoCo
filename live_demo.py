import argparse
import logging
import os
import select
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Optional, Deque

import imgui
import numpy as np
import torch
import trimesh
from aitviewer.remote.renderables.meshes import RemoteMeshes
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.spheres import Spheres
from aitviewer.viewer import Viewer
from pygame.time import Clock
from scipy.spatial.transform import Rotation as R

from models.imucoco import IMUCoCo
from models.dtp import Poser
import articulate
from utils import imu_config
import path_config



"""
Modify below variables to match your system configuration.
"""
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
#     os.environ["CONDA_PREFIX"], "plugins", "platforms"
# )
# os.environ["QT_XCB_GL_INTEGRATION"] = "glx"
# os.environ["QT_QPA_PLATFORM"] = "xcb"
# os.environ["LIBGL_ALWAYS_INDIRECT"] = "0"


DEBUG_IMU_ORIENTATION = True    # set to True to visualize the IMU orientation in the viewer

"""
Here it supports maxiumum 2 pair of watch and phones, each pair and watch share the same port.
You don't have to change the code if you only have no more than 4 devices connected.
If you have more than 5 or more device, or has different port numbers, simply modify accordingly:
- NUM_IMU_DEVICES
- SERVER_PORTS
- IMU_DEVICE_SOCKET_PORT
- IMU_DEVICE_OFFSET
"""

NUM_IMU_DEVICES = 4
SERVER_PORTS = [8001, 8002]
ACTION_PORT = 9000  # for internal communication between server and viewer
# which port each IMU device data is coming from.
IMU_DEVICE_SOCKET_PORT = [
    8001, 8001, 8002, 8002
]
IMU_DEVICE_OFFSET = [
    0, 1, 0, 1
]
LOAD_PREVIOUS_FLOOR_ALIGNMENT = False

BUFFER_MAX_LEN = 300
IMU_DEVICE_COLORS = [
    (np.random.rand()*0.5 + 0.3, np.random.rand()*0.5 + 0.3, np.random.rand()*0.5 + 0.3, 1.0) for _ in range(NUM_IMU_DEVICES)
]
SERVER_ACTIVE_STATUS = {p: False for p in SERVER_PORTS}

IMU_ZUP_TO_YUP = torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]).float()



# remove previous demo_cache_attach_device_selector.npy and demo_cache_device_coordinates.npy
if os.path.exists("demo_cache_attach_device_selector.npy"):
    os.remove("demo_cache_attach_device_selector.npy")
if os.path.exists("demo_cache_device_coordinates.npy"):
    os.remove("demo_cache_device_coordinates.npy")
if not LOAD_PREVIOUS_FLOOR_ALIGNMENT:
    if os.path.exists("demo_cache_floor_alignment.npy"):
        os.remove("demo_cache_floor_alignment.npy")

class ThreadSafeMotionBuffer:
    def __init__(self, max_length: int = 250):
        self.max_length = max_length
        self._buffers: Dict[str, Deque] = {}
        self._locks: Dict[str, threading.RLock] = {}

        # Initialize buffers for all devices
        for i in range(NUM_IMU_DEVICES):
            device_id = f"IMU-{i:02d}"
            self._buffers[device_id] = deque(maxlen=max_length)
            self._locks[device_id] = threading.RLock()

    def add_data(self, device_id: str, motion_data: np.ndarray) -> bool:
        """Add motion data to the buffer for a specific device.

        The deque with maxlen automatically removes the oldest data when full.
        """
        if device_id not in self._buffers:
            return False

        with self._locks[device_id]:
            was_full = len(self._buffers[device_id]) == self.max_length
            self._buffers[device_id].append(motion_data.copy())  # Copy to avoid reference issues

        return True

    def get_middle_data(self, device_id: str, start_ratio: float = 0.25, end_ratio: float = 0.85) -> Optional[np.ndarray]:
        """Get the middle portion of the data buffer for calibration."""
        if device_id not in self._buffers:
            return None

        with self._locks[device_id]:
            data = list(self._buffers[device_id])
            if len(data) < 50:  # Need at least 50 samples
                return None

            start_idx = int(start_ratio * len(data))
            end_idx = int(end_ratio * len(data))
            return np.array(data[start_idx:end_idx])

    def get_length(self, device_id: str) -> int:
        """Get the current length of the buffer for a device."""
        if device_id not in self._buffers:
            return 0
        with self._locks[device_id]:
            return len(self._buffers[device_id])

    def clear(self, device_id: str = None):
        """Clear the buffer for a specific device or all devices."""
        if device_id:
            if device_id in self._buffers:
                with self._locks[device_id]:
                    self._buffers[device_id].clear()
        else:
            for device_id in self._buffers:
                with self._locks[device_id]:
                    self._buffers[device_id].clear()

    def clear_all(self):
        """Clear all buffers for all devices."""
        for device_id in self._buffers:
            with self._locks[device_id]:
                self._buffers[device_id].clear()

    def get_device_ids(self) -> list:
        """Get list of all device IDs."""
        return list(self._buffers.keys())


# Initialize the thread-safe motion data buffer
MOTION_DATA_BUFFER = ThreadSafeMotionBuffer(max_length=BUFFER_MAX_LEN)

# Simple data storage for inference mode (faster access - no locking needed)
INFERENCE_DATA = {f"IMU-{i:02d}": None for i in range(NUM_IMU_DEVICES)}

# FPS tracking for inference data
INFERENCE_DATA_TIMESTAMPS = {f"IMU-{i:02d}": None for i in range(NUM_IMU_DEVICES)}
INFERENCE_FPS_WINDOW = 60  # Calculate FPS over last 60 updates
IMU_SERVER_PORT_OFFSET_TO_DEVICES = {
    (port, offset): device_id for port, offset, device_id in zip(IMU_DEVICE_SOCKET_PORT, IMU_DEVICE_OFFSET, MOTION_DATA_BUFFER.get_device_ids())
}
print("IMU_SERVER_PORT_OFFSET_TO_DEVICES", IMU_SERVER_PORT_OFFSET_TO_DEVICES)

IMU_R_GLB2ERTH = {f"IMU-{i:02d}": torch.eye(3).float() for i in range(NUM_IMU_DEVICES)}  # Rotation from Global (SMPL) Frame to Earth Frame

# load IMU_R_GLB2ERTH if it exists
for ii in range(NUM_IMU_DEVICES):
    imu_id = f"IMU-{ii:02d}"
    if os.path.exists(f"demo_cache_R_GLB2ERTH_{imu_id}.npy"):
        IMU_R_GLB2ERTH[imu_id] = torch.tensor(np.load(f"demo_cache_R_GLB2ERTH_{imu_id}.npy")).float()

IMU_R_BONE2SEN = {f"IMU-{i:02d}": torch.eye(3).float() for i in range(NUM_IMU_DEVICES)}  # Rotation from Sensor Frame to Bone Frame
IMU_ACC_OFFSET = {f"IMU-{i:02d}": torch.zeros(3).float() for i in range(NUM_IMU_DEVICES)}

IMU_COORDINATES = torch.zeros(0, 3)  # this is the coordinates of the attached devices
IMU_ATTACHED_SELECTOR = torch.zeros(NUM_IMU_DEVICES).bool()  # this is the selector of the attached devices to index all imu streams
MODEL_NEEDS_UPDATE_IMU_COORDINATES = False

# Thread-safe inference control
INFERENCE_ON = threading.Event()
INFERENCE_ON.clear()  # Start with inference disabled

# Thread-safe calibration control
CALIBRATION_ON = threading.Event()
CALIBRATION_ON.clear()  # Start with calibration disabled


def stop_calibration():
    """Stop any ongoing calibration process"""
    CALIBRATION_ON.clear()
    print("Calibration stopped.")


def is_calibration_data_ready():
    """Check if there's enough calibration data ready for processing"""
    for device_id in MOTION_DATA_BUFFER.get_device_ids():
        if MOTION_DATA_BUFFER.get_length(device_id) < 50:
            return False
    return True


BUFFER_MAX_LEN = 250  # about 5 seconds
CLIB_R_N2G = {}  # rotation from navigation frame  to global (model) frame
CLIB_R_B2S = {}  # rotation from bone frame to sensor frame.

UDP_SENSOR_BUFFER_SIZE = 2048 * 32
SERVER_IP = '0.0.0.0'


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


@dataclass
class IMUDevice:
    """Represents an IMU device with its state and properties."""
    id: str
    coordinates: Optional[Tuple[float, float, float]] = None
    is_attached: bool = False
    is_calibrated: bool = False
    connected: bool = False


class CalibrationState(Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class IMUPoseViewer(Viewer):
    """Extended aitviewer with IMU device management capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Override the custom_font with a larger size for better visibility
        import os
        from pathlib import Path
        font_dir = Path(__file__).parent / "resources" / "fonts"
        if not font_dir.exists():
            # Fallback to aitviewer font directory
            import aitviewer
            font_dir = Path(aitviewer.__file__).parent / "resources" / "fonts"

        if font_dir.exists():
            self.fonts = imgui.get_io().fonts
            self.custom_font = self.fonts.add_font_from_file_ttf(
                os.path.join(font_dir, "Custom.ttf"), 20  # Increased from 15 to 20
            )
            self.imgui.refresh_font_texture()

        # IMU device management
        self.imu_devices: Dict[str, IMUDevice] = {}
        self.imu_devices_id2index: Dict[str, int] = {}
        self.calibration_state = CalibrationState.IDLE
        self.calibration_progress = 0.0
        self.calibration_timer = 0.0
        self.calibration_duration = 8.0  # seconds

        # self.all_imu_coordinates = np.zeros((NUM_IMU_DEVICES, 3))

        self.floor_align_state = CalibrationState.IDLE
        self.floor_align_progress = 0.0
        self.floor_align_timer = 0.0
        self.floor_align_duration = 8.0  # seconds

        # UI state
        self._show_imu_controller = True
        self._pending_attachment_device_id: Optional[str] = None
        self._pending_detachment_device_id: Optional[str] = None
        self._attachment_mode = False

        # Initialize with some example IMU devices
        self._initialize_imu_devices()

        # Add IMU controller to GUI controls
        self.gui_controls["imu_controller"] = self.gui_imu_controller
        self.selected_mode = "view"

        self._imu_attachment_markers = {
        }
        self._inference_mode = False

        self.conn = None
        self.conn_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3000
        self.reconnect_delay = 1.0  # seconds

        self.calibration_t_pose_mesh = None
        # Setup TCP message listener for device connectivity
        self._setup_tcp_message_listener()

    def init_calibration_t_pose_mesh(self):
        identity_matrix = torch.eye(3).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 3, 3)
        t_pos_smpl = identity_matrix.repeat(1, 24, 1, 1)  # Shape (1, 24, 3, 3)
        initial_glb_pose, initial_joints_positions, initial_vertex_positions = imu_config.body_model.forward_kinematics(pose=t_pos_smpl, calc_mesh=True)
        self.calibration_t_pose_mesh = Meshes(
            initial_vertex_positions.detach().numpy(),
            imu_config.body_model.face,
            is_selectable=False,
            gui_affine=False,
            name=f"IMUCoCo Calibration T Pose",
        )
        self.scene.add(self.calibration_t_pose_mesh)

    def _setup_tcp_message_listener(self):
        """Setup TCP message listener for device connectivity updates."""
        self._connect_to_server()

    def _connect_to_server(self):
        """Connect to the server via TCP."""
        try:
            if self.conn:
                self.conn.close()

            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn.connect((SERVER_IP, ACTION_PORT))
            self.conn.setblocking(False)
            self.conn_connected = True
            self.reconnect_attempts = 0
            print(f"Connected to server at {SERVER_IP}:{ACTION_PORT}")
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            self.conn_connected = False
            self.conn = None

    def _attempt_reconnect(self):
        """Attempt to reconnect to the server."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print("Max reconnection attempts reached. Giving up.")
            return False

        self.reconnect_attempts += 1
        print(f"Waiting to connect to the server... (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})...")

        # Wait before reconnecting
        time.sleep(self.reconnect_delay)
        self._connect_to_server()

        return self.conn_connected

    def _check_tcp_messages(self):
        """Check for incoming TCP messages and handle device connectivity updates."""
        if not self.conn_connected or not self.conn:
            # Try to reconnect if not connected
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self._attempt_reconnect()
            return

        try:
            # Try to receive data (non-blocking)
            data = self.conn.recv(1024)
            if data:
                message = data.decode('utf-8')
                self._handle_tcp_message(message)
            elif data == b'':
                # Connection closed by server
                print("Server closed connection")
                self.conn_connected = False
                self.conn.close()
                self.conn = None
        except BlockingIOError:
            # No data available, which is normal
            pass
        except ConnectionResetError:
            # Connection was reset by server
            print("Connection reset by server")
            self.conn_connected = False
            if self.conn:
                self.conn.close()
                self.conn = None
        except Exception as e:
            print(f"Error checking TCP messages: {e}")
            self.conn_connected = False
            if self.conn:
                self.conn.close()
                self.conn = None

    def _handle_tcp_message(self, message: str):
        """Handle incoming TCP messages for device connectivity."""
        if message.startswith("DEVICE_CONNECTED:"):
            device_ids = message.split(":", 1)[1]
            for device_id in device_ids.split("+"):
                if device_id in self.imu_devices:
                    self.imu_devices[device_id].connected = True
                    print(f"Device connected via TCP: {device_id}")
        elif message.startswith("DEVICE_DISCONNECTED:"):
            device_ids = message.split(":", 1)[1]
            for device_id in device_ids.split("+"):
                if device_id in self.imu_devices:
                    self.imu_devices[device_id].connected = False
                    print(f"Device disconnected via TCP: {device_id}")

    """ Viewer--Server Communications """

    def _send_server_message(self, message: bytes):
        """Send a message to the server with reconnection handling."""
        if not self.conn_connected or not self.conn:
            print("Not connected to server, attempting to reconnect...")
            if not self._attempt_reconnect():
                print("Failed to reconnect, message not sent")
                return False

        try:
            self.conn.send(message)
            return True
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            print(f"Connection error while sending message: {e}")
            self.conn_connected = False
            if self.conn:
                self.conn.close()
                self.conn = None
            return False
        except Exception as e:
            print(f"Error sending message: {e}")
            return False

    def _initialize_imu_devices(self):
        """Initialize some example IMU devices."""
        for i in range(0, NUM_IMU_DEVICES):
            device_id = f"IMU-{i:02d}"
            self.imu_devices[device_id] = IMUDevice(id=device_id)
            self.imu_devices_id2index[device_id] = i

    """ UI Functions of IMU Config Panels"""

    def gui_imu_controller(self):
        """Render the IMU controller GUI."""
        if not self._show_imu_controller:
            return

        imgui.set_next_window_position(self.window_size[0] * 0.02, self.window_size[1] * 0.3, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(450, 800, imgui.FIRST_USE_EVER)

        expanded, self._show_imu_controller = imgui.begin("IMU Controller", self._show_imu_controller)
        if expanded:
            self._render_imu_device_list()
            imgui.separator()
            self._render_calibration_controls()
            self._render_floor_align_controls()
            self._render_inference_controls()

        imgui.end()

    def _render_imu_device_list(self):
        """Render the list of IMU devices."""
        imgui.text("IMU Devices:")
        imgui.spacing()

        # Create a child window for the device list with scrolling
        imgui.begin_child("device_list", height=150 * NUM_IMU_DEVICES, border=True)

        for device_id, device in self.imu_devices.items():
            self._render_device_entry(device)
            imgui.spacing()

        imgui.end_child()

    def _render_device_entry(self, device: IMUDevice):
        """Render a single IMU device entry."""
        # Device header with attachment status
        imgui.push_font(self.custom_font)

        # Color coding based on status
        if device.is_attached and device.is_calibrated:
            device_id_digit = int(device.id.replace("IMU-", ""))
            color = IMU_DEVICE_COLORS[device_id_digit]
        elif device.is_attached and not device.is_calibrated:
            device_id_digit = int(device.id.replace("IMU-", ""))
            color = IMU_DEVICE_COLORS[device_id_digit]
        else:
            color = (0.5, 0.5, 0.5, 0.5)  # Gray for detached

        if device.connected:
            imgui.text_colored('~', 0, 1, 0, 1)
        else:
            imgui.text_colored('~', 1, 0, 0, 1)
        imgui.same_line()
        imgui.text_colored(f"{device.id}", color[0], color[1], color[2], color[3])
        imgui.pop_font()

        if device.is_attached:
            imgui.text("Attached")
            imgui.same_line()
            if imgui.small_button(f"Detach##{device.id}"):
                self._detach_device(device.id)
        else:
            imgui.text("Detached")
            imgui.same_line()
            if imgui.small_button(f"Attach##{device.id}"):
                self._initiate_device_attachment(device.id)

        # Coordinates
        if device.coordinates is not None:
            coord_str = f"Coordinates: {device.coordinates[0]:.3f}, {device.coordinates[1]:.3f}, {device.coordinates[2]:.3f}"
            imgui.text(coord_str)
        else:
            imgui.text("Coordinates: Not set")

        # Calibration status
        calibration_text = "Calibrated? Yes" if device.is_calibrated else "Calibrated? No"
        imgui.text(calibration_text)

        imgui.text("")

    def _render_floor_align_controls(self):
        """Render calibration controls."""
        if self.floor_align_state == CalibrationState.IDLE:
            if not self.imu_devices[list(self.imu_devices.keys())[0]].connected:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                imgui.button("Start Floor Alignment", width=400)
                imgui.pop_style_var()
                if imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.text("Connect the first device to start calibration")
                    imgui.end_tooltip()
            else:
                if imgui.button("Start Floor Alignment", width=400):
                    self.floor_align_state = CalibrationState.IN_PROGRESS
                    self._start_floor_alignment_timer()
                    self._send_server_message(b"START_FLOOR_ALIGN")

        elif self.floor_align_state == CalibrationState.IN_PROGRESS:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            imgui.button("Aligning in Progress...", width=400)
            imgui.pop_style_var()

            # Progress bar
            imgui.progress_bar(self.floor_align_progress, size=(400, 0), overlay=f"{self.floor_align_progress * 100:.1f}%")

        elif self.floor_align_state == CalibrationState.COMPLETED:
            self._send_server_message(b"PROCESS_FLOOR_ALIGN")
            self.floor_align_state = CalibrationState.IDLE
            # imgui.button("Start Floor Alignment", width=320)

    def _render_calibration_controls(self):
        """Render calibration controls."""
        if self.calibration_state == CalibrationState.IDLE:
            # Check if any devices are attached
            attached_devices = [d for d in self.imu_devices.values() if d.is_attached]

            if not attached_devices:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                imgui.button("Start T-Pose Calibration", width=400)
                imgui.pop_style_var()
                if imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.text("Attach at least one device to start calibration")
                    imgui.end_tooltip()
            else:
                if imgui.button("Start T-Pose Calibration", width=400):
                    self._start_t_pose_calibration_timer()
                    self._send_server_message(b"START_T_POSE_CALIBRATION")

        elif self.calibration_state == CalibrationState.IN_PROGRESS:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            imgui.button("Calibration in Progress...", width=400)
            imgui.pop_style_var()

            # Progress bar
            imgui.progress_bar(self.calibration_progress, size=(400, 0), overlay=f"{self.calibration_progress * 100:.1f}%")

        elif self.calibration_state == CalibrationState.COMPLETED:
            # # imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.8, 0.0, 1.0)
            # # if imgui.button("Calibration Complete!", width=320):
            # #     self.calibration_state = CalibrationState.IDLE
            # imgui.pop_style_color()
            self._send_server_message(b"PROCESS_T_POSE_CALIBRATION")
            self.calibration_state = CalibrationState.IDLE
            # imgui.button("Start T-Pose Calibration", width=320)

    def _render_inference_controls(self):
        """Render calibration controls."""
        if self._inference_mode:
            # imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.0, 0.0, 1.0)
            # imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            if imgui.button("Stop Inference", width=400):
                self._inference_mode = False
                self._send_server_message(b"STOP_INFERENCE")

                self.calibration_t_pose_mesh.color = (0.5, 0.5, 0.5, 1.0)

                # make the attached imu devices visible
                for device_id, imu_device in self.imu_devices.items():
                    if imu_device.is_attached and device_id in self._imu_attachment_markers:
                        device_id_digit = int(imu_device.id.replace("IMU-", ""))
                        color = IMU_DEVICE_COLORS[device_id_digit]
                        self._imu_attachment_markers[device_id].color = (color[0], color[1], color[2], color[3])

        # imgui.pop_style_color()
        else:
            # imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.8, 0.0, 1.0)
            if imgui.button("Start Inference", width=400):
                self._inference_mode = True
                self._send_server_message(b"START_INFERENCE")

                # make the calibration pose invisible to show the inference pose
                self.calibration_t_pose_mesh.color = (0, 0, 0, 0)

                # make all attached imu devices invisible
                for device_id, imu_device in self.imu_devices.items():
                    if imu_device.is_attached and device_id in self._imu_attachment_markers:
                        self._imu_attachment_markers[device_id].color = (0, 0, 0, 0)

            # imgui.pop_style_color()

    """ Device Attachment Functions"""

    def _initiate_device_attachment(self, device_id: str):
        """Initiate attachment process for a device."""
        self._pending_attachment_device_id = device_id
        self._attachment_mode = True
        self.selected_mode = "inspect"
        print(f"Attachment mode activated for {device_id}. Click on a surface to attach the device.")

    def _get_all_attached_device_ids(self):
        """Get all attached device IDs."""
        return [device_id for device_id in self.imu_devices.keys() if self.imu_devices[device_id].is_attached]

    def _attach_device_at_coordinate(self, device_id: str, coordinates: Tuple[float, float, float]):
        """Attach device at specified coordinates."""
        if device_id in self.imu_devices:
            device = self.imu_devices[device_id]
            device.coordinates = coordinates
            device.is_attached = True
            device.is_calibrated = False  # Reset calibration status
            print(f"Device {device_id} attached at coordinates: {coordinates}")

            # get all attached device coordinates and save them
            all_attached_device_ids = self._get_all_attached_device_ids()
            attach_device_selector = [device_id in all_attached_device_ids for device_id in self.imu_devices_id2index.keys()]
            all_attached_device_coordinates = [self.imu_devices[device_id].coordinates for i, device_id in enumerate(self.imu_devices_id2index.keys()) if attach_device_selector[i]]
            np.save("demo_cache_device_coordinates.npy", all_attached_device_coordinates)
            np.save("demo_cache_attach_device_selector.npy", attach_device_selector)
            self._send_server_message(b"UPDATE_COORDINATES")

    def _add_virtual_marker(self, intersection):
        try:
            seq = intersection.node
            positions = seq.vertices[:, intersection.vert_id: intersection.vert_id + 1] + seq.position[np.newaxis]
            distance_label = f"{self._pending_attachment_device_id} | Coordinates {positions[0][0]}"
            if positions.shape[0] != 1 or positions.shape[1] != 1:
                return None # incorrect position
            if self._pending_attachment_device_id not in self._imu_attachment_markers:
                ms = Spheres(positions, name=distance_label, radius=0.02)  # Enlarged sphere size
                ms.current_frame_id = seq.current_frame_id
                np.set_printoptions(suppress=True)
                device_id_digit = int(self._pending_attachment_device_id.replace("IMU-", ""))
                ms.color = IMU_DEVICE_COLORS[device_id_digit]
                self.scene.add(ms)
                self._imu_attachment_markers[self._pending_attachment_device_id] = ms
            else:
                ms = self._imu_attachment_markers[self._pending_attachment_device_id]
                if len(positions.shape) == 3:
                    positions = positions[0]
                ms.current_sphere_positions = positions
                device_id_digit = int(self._pending_attachment_device_id.replace("IMU-", ""))
                ms.color = IMU_DEVICE_COLORS[device_id_digit]
            return ms
        except Exception as e:
            print("Error adding virtual marker", e)
            return None

    def _detach_device(self, device_id: str):
        """Detach a device."""
        if device_id in self.imu_devices:
            device = self.imu_devices[device_id]
            device.coordinates = None
            device.is_attached = False
            device.is_calibrated = False

            self.imu_devices[device_id].coordinates = np.asarray([0.0, 0.0, 0.0])
            # self.all_imu_coordinates[device_id_index] = None

            # get all attached device coordinates and save them
            all_attached_device_ids = self._get_all_attached_device_ids()
            attach_device_selector = [device_id in all_attached_device_ids for device_id in self.imu_devices_id2index.keys()]
            all_attached_device_coordinates = [self.imu_devices[device_id].coordinates for i, device_id in enumerate(self.imu_devices_id2index.keys()) if attach_device_selector[i]]
            np.save("demo_cache_device_coordinates.npy", all_attached_device_coordinates)
            np.save("demo_cache_attach_device_selector.npy", attach_device_selector)
            self._send_server_message(b"UPDATE_COORDINATES")
            print(f"Device {device_id} detached")

            self._imu_attachment_markers[device_id].color = (0, 0, 0, 0)


    def mouse_press_event(self, x: int, y: int, button: int):
        """Override mouse press event to handle IMU attachment."""
        if not self.imgui_user_interacting and self._attachment_mode and self.selected_mode == "inspect":
            result = self.mesh_mouse_intersection(x, y)
            if result is not None:
                self.interact_with_sequence(result, button)
        else:
            super().mouse_press_event(x, y, button)

    def interact_with_sequence(self, intersection, button):
        """Handle interaction with scene for IMU attachment."""
        if button == 1 and self._pending_attachment_device_id:  # left mouse
            result = self._add_virtual_marker(intersection)
            if result is not None:
                self.record_imu_attach_coordinate(intersection)

    def record_imu_attach_coordinate(self, intersection):
        """Record IMU attachment coordinates from intersection."""
        if self._pending_attachment_device_id and intersection:
            # Use world coordinates from the intersection
            coordinates = tuple(intersection.point_world)
            print("_pending_attachment_device_id", self._pending_attachment_device_id)
            print("coordinates:", coordinates)
            self._attach_device_at_coordinate(self._pending_attachment_device_id, coordinates)

            # Reset attachment mode
            self._pending_attachment_device_id = None
            self._attachment_mode = False
            self.selected_mode = "view"

    """ Calibration UI """

    def _start_t_pose_calibration_timer(self):
        """Start calibration process for all attached devices."""
        attached_devices = [d for d in self.imu_devices.values() if d.is_attached]
        if attached_devices:
            self.calibration_state = CalibrationState.IN_PROGRESS
            self.calibration_progress = 0.0
            self.calibration_timer = 0.0
            print(f"Starting calibration for {len(attached_devices)} attached devices...")

    def _start_floor_alignment_timer(self):
        """Start calibration process for all attached devices."""
        for device_id, device in self.imu_devices.items():
            # use the first device for alignment
            if device.is_attached:
                self.floor_align_state = CalibrationState.IN_PROGRESS
                self.floor_align_progress = 0.0
                self.floor_align_timer = 0.0
                print(f"Starting floor alignment based on {device_id}...")
                break

    def _update_calibration(self, frame_time: float):
        """Update calibration progress."""
        if self.calibration_state == CalibrationState.IN_PROGRESS:
            self.calibration_timer += frame_time
            self.calibration_progress = min(1.0, self.calibration_timer / self.calibration_duration)

            if self.calibration_progress >= 1.0:
                self._complete_calibration()

        if self.floor_align_state == CalibrationState.IN_PROGRESS:
            self.floor_align_timer += frame_time
            self.floor_align_progress = min(1.0, self.floor_align_timer / self.floor_align_duration)

            if self.floor_align_progress >= 1.0:
                self._complete_floor_alignment()

    def _complete_calibration(self):
        """Complete calibration process."""
        # Set all attached devices as calibrated
        for device in self.imu_devices.values():
            if device.is_attached:
                device.is_calibrated = True
        self.calibration_state = CalibrationState.COMPLETED
        print("Calibration completed for all attached devices")

    def _complete_floor_alignment(self):
        """Complete calibration process."""
        # Set all attached devices as calibrated
        for device in self.imu_devices.values():
            if device.is_attached:
                device.is_calibrated = True

        self.floor_align_state = CalibrationState.COMPLETED
        print("Floor alignment completed for all attached devices")

    """ Override Methods for AITViewer"""

    def render(self, time, frame_time, export=False, transparent_background=False):
        """Override render to update calibration progress."""
        # Update calibration if in progress
        if not export:
            self._update_calibration(frame_time)
            # Check for TCP messages (device connectivity updates)
            self._check_tcp_messages()

        # Call parent render
        super().render(time, frame_time, export, transparent_background)

    def key_event(self, key, action, modifiers):
        """Override key event to handle IMU controller shortcuts."""
        super().key_event(key, action, modifiers)

        # Add shortcut to toggle IMU controller (I key when not in inspect mode)
        if (action == self.wnd.keys.ACTION_PRESS and
                key == self.wnd.keys.I and
                not self.imgui.io.want_capture_keyboard and
                self.selected_mode != "inspect"):
            self._show_imu_controller = not self._show_imu_controller

    def __del__(self):
        """Cleanup when the viewer is destroyed."""
        if self.conn:
            self.conn.close()
            self.conn = None


def start_pose_viewer():
    from aitviewer.configuration import CONFIG as C
    from aitviewer.renderables.spheres import Spheres
    C.update_conf({"server_enabled": True})

    print("Starting Pose Viewer with IMU Controller.....")
    pose_viewer = IMUPoseViewer()

    # Add a sphere to let it set the level of floor
    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]
    j0, _ = imu_config.body_model.get_zero_pose_joint_and_vertex()
    b = articulate.math.joint_position_to_bone_vector(j0[lower_body].unsqueeze(0), lower_body_parent).squeeze(0)
    bone_orientation, bone_length = articulate.math.normalize_tensor(b, return_norm=True)
    b = bone_orientation * bone_length
    b[:3] = 0
    floor_y = j0[10:12, 1].min().item()

    sphere = Spheres(np.asarray([[0, floor_y, 0]]), radius=0.01, color=(0, 0, 0, 0))
    pose_viewer.scene.add(sphere)

    pose_viewer.init_calibration_t_pose_mesh()

    pose_viewer.render_gui = True  # Enable GUI to show IMU controller
    pose_viewer.shadows_enabled = False
    pose_viewer.run()


def parse_motion_message(message):
    """
    Parse incoming motion message and extract user_id, device_type, and motion data.

    Message format:
    ${userId};${deviceType};motion:${motion_text}&${motion_text}&${motion_text}...

    motion_text format:
    timestamp user_acc.x user_acc.y user_acc.z grav.x grav.y grav.z gyro.x gyro.y gyro.z mag.x mag.y mag.z attitude.roll attitude.pitch attitude.yaw quat.x quat.y quat.z quat.w
    """
    try:
        # Split message components
        parts = message.split(';')
        # user_id = parts[0]
        device_type = parts[1]

        # Extract motion data part
        motion_part = message.split(':', 1)[1].strip()

        motion_text = motion_part.split('&')[-1]  # only take the last one
        motion_values = motion_text.split(' ')[1:]  # not include timestamp
        # Parse motion data: userAcc(3) + gravity(3) + gyro(3) + quat(4) =  values
        motion_data = np.array([float(motion_values[ii]) for ii in [0, 1, 2, 12, 9, 10, 11]])
        return device_type, motion_data

    except Exception as e:
        logger.error(f"Error parsing motion message: {e}")
        return None, None


def handle_motion_data(server_port, device_type, motion_data):
    """
    Handle incoming motion data based on user_id and device_type.

    Args:
        user_id (str): User identifier
        device_type (str): 'phone' or 'watch'
        timestamps (np.ndarray): Timestamp array
        motion_data (np.ndarray): Motion data array
    """
    if device_type == 'watch':
        offset = 0
    elif device_type == 'phone':
        offset = 1
    else:
        raise NotImplementedError

    # Store data based on inference mode
    device_id = IMU_SERVER_PORT_OFFSET_TO_DEVICES[(server_port, offset)]

    # Get the most recent motion sample (last one in the batch)
    current_time = time.time()
    # print(f"Received motion data: {device_id} {len(motion_data)}")
    INFERENCE_DATA[device_id] = motion_data
    INFERENCE_DATA_TIMESTAMPS[device_id] = current_time

    if CALIBRATION_ON.is_set():
        MOTION_DATA_BUFFER.add_data(device_id, motion_data)

    return device_id


def handle_socket_message(server_port, message):
    """
    Handle incoming socket messages.

    Message types:
    - 'init': Initialize connection
    - 'stop': Stop connection
    - Motion data: userId;deviceType;motion:data...
    """

    if 'init' in message:
        logger.info('Socket initialized')
        return 'INIT'

    elif 'stop' in message:
        logger.info('Socket stopped')
        SERVER_ACTIVE_STATUS[server_port] = False
        return False

    elif 'motion' in message:
        device_type, motion_data = parse_motion_message(message)
        if device_type is not None:
            device_id = handle_motion_data(server_port, device_type, motion_data)
            return device_id
        else:
            return False

    else:
        logger.warning(f'Unknown message: {message[:50]}...')
        return 'UNKNOWN'


def mean_rotation_quaternion(quats):
    # rotations: (N, 3, 3)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    # Markley's method: compute symmetric 4x4 accumulator matrix
    A = np.zeros((4, 4))
    for q in quats:
        q = q.reshape(4, 1)  # Column vector
        A += q @ q.T

    # Compute the eigenvector of A with the largest eigenvalue
    eigvals, eigvecs = np.linalg.eigh(A)
    avg_quat = eigvecs[:, np.argmax(eigvals)]

    # Ensure the average quaternion is normalized
    avg_quat /= np.linalg.norm(avg_quat)
    return avg_quat


def _start_data_buffer_on_server():
    """Start data buffer"""
    if CALIBRATION_ON.is_set():
        print("Calibration already in progress. Please wait for current calibration to complete.")
        return

    print("Starting data buffer...")
    MOTION_DATA_BUFFER.clear_all()  # Clear existing buffer
    CALIBRATION_ON.set()  # Enable data collection
    print("Data buffer started. Hold device flat and still, then send PROCESS_FLOOR_ALIGN_FROM_BUFFER when ready...")


def _update_floor_alignment_from_buffer_on_server():
    """Process floor alignment from buffer"""
    # retrieve the middle 50% of the data buffer
    for device_id in MOTION_DATA_BUFFER.get_device_ids():
        motion_data_length = MOTION_DATA_BUFFER.get_length(device_id)
        print(f"Updating floor alignment for {device_id}, motion data length: {motion_data_length}")

        middle_motion_data = MOTION_DATA_BUFFER.get_middle_data(device_id)
        if middle_motion_data is not None and len(middle_motion_data) > 50:
            print(f"Middle motion data shape: {middle_motion_data.shape}")
            r_glb2erth_zup = middle_motion_data[:, 3:]
            r_glb2erth_zup = R.from_quat(quat=mean_rotation_quaternion(r_glb2erth_zup), scalar_first=True).as_matrix()
            # convert z-up to y-up
            IMU_R_GLB2ERTH[device_id] = IMU_ZUP_TO_YUP @ torch.tensor(r_glb2erth_zup).float() @ IMU_ZUP_TO_YUP.T
            print(f"Updated Floor Alignment (on {device_id}): {IMU_R_GLB2ERTH[device_id]}")

            np.save(f"demo_cache_R_GLB2ERTH_{device_id}.npy", IMU_R_GLB2ERTH[device_id].cpu().numpy())
        else:
            print(f"Warning: {device_id} does not have enough data to do floor alignment")

    CALIBRATION_ON.clear()  # Disable data collection
    print("Floor alignment calibration completed.")
    return


def _update_t_pose_calibration_from_buffer_on_server():
    """Process T-pose calibration using collected data"""
    # retrieve the middle 50% of the data buffer
    for device_id in MOTION_DATA_BUFFER.get_device_ids():
        motion_data_length = MOTION_DATA_BUFFER.get_length(device_id)
        print(f"Updating T Pose Calibration for {device_id}, motion data length: {motion_data_length}")

        middle_motion_data = MOTION_DATA_BUFFER.get_middle_data(device_id)
        if middle_motion_data is not None and len(middle_motion_data) > 50:
            print(f"Middle motion data shape: {middle_motion_data.shape}")
            r_sen2erth_zup = middle_motion_data[:, 3:]
            r_sen2erth_zup = R.from_quat(quat=mean_rotation_quaternion(r_sen2erth_zup), scalar_first=True).as_matrix().astype(np.float32)
            r_sen2erth = IMU_ZUP_TO_YUP @ r_sen2erth_zup @ IMU_ZUP_TO_YUP.T # convert z-up to y-up

            r_glb2erth = IMU_R_GLB2ERTH[device_id].float()
            # bone frame equals global frame at T-Pose.
            # r_bone2sen = (r_glb2erth.T @ r_sen2erth).T
            print("r_glb2erth", r_glb2erth)
            r_bone2sen = torch.tensor(r_sen2erth.T, dtype=torch.float) @ r_glb2erth
            print("r_bone2sen", r_bone2sen)
            # r_bone2sen = r_sen2erth.T @ r_glb2erth
            IMU_R_BONE2SEN[device_id] = r_bone2sen.float()

            acc_erth_zup = np.mean(middle_motion_data[:, :3], axis=0).astype(np.float32)
            # user acceleration at T-pose measured in earth frame
            acc_erth = IMU_ZUP_TO_YUP @ torch.tensor(acc_erth_zup).float() # convert z-up to y-up
            acc_glb_offset = r_glb2erth.T @ torch.tensor(acc_erth, dtype=torch.float) # user acceleration at global frame, which should be corrected to 0 at T-pose;
            print("acc_glb_offset", acc_glb_offset)

            IMU_ACC_OFFSET[device_id] = acc_glb_offset.float()

            # test, orientation at T pose should be I
            t_test = IMU_R_GLB2ERTH[device_id].T @ r_sen2erth @ IMU_R_BONE2SEN[device_id]

            print(f"Updated T Pose Calibration {device_id}: {r_bone2sen}; t_test: {t_test}")
        else:
            print(f"Warning: {device_id} does not have enough data to do calibration")

    CALIBRATION_ON.clear()  # Disable data collection
    print("T-pose calibration completed.")
    return


def _imu_calibration(raw_imu_quat_acc, device_id):
    # print("raw_imu_quat_acc", raw_imu_quat_acc.shape, raw_imu_quat_acc)
    quat = raw_imu_quat_acc[3:]
    acc_erth_z_up = raw_imu_quat_acc[:3]
    r_sen2erth_z_up = torch.tensor(R.from_quat(quat=quat, scalar_first=True).as_matrix()).float()

    r_sen2erth = IMU_ZUP_TO_YUP @ r_sen2erth_z_up @ IMU_ZUP_TO_YUP.T # convert z-up to y-up
    acc_erth = IMU_ZUP_TO_YUP @ acc_erth_z_up # convert z-up to y-up

    # print("original sen2earth", r_sen2erth)
    r_bone2glb = IMU_R_GLB2ERTH[device_id].T @ r_sen2erth @ IMU_R_BONE2SEN[device_id]
    # print("calibrated rbone2glb", r_bone2glb)
    acc_glb = IMU_R_GLB2ERTH[device_id].T @ torch.tensor(acc_erth).float() - IMU_ACC_OFFSET[device_id]

    return r_bone2glb.float(), acc_glb.float()


def _update_device_coordinates_on_server():
    global MODEL_NEEDS_UPDATE_IMU_COORDINATES
    global IMU_ATTACHED_SELECTOR, IMU_COORDINATES
    cached_coordinates = np.load("demo_cache_device_coordinates.npy")
    cached_selector = np.load("demo_cache_attach_device_selector.npy")
    print("Loaded cached coordinates (Data Server Process)", cached_coordinates.shape, cached_coordinates)
    print("IMU_ATTACHED_SELECTOR", cached_selector)
    IMU_COORDINATES = torch.tensor(cached_coordinates)
    IMU_ATTACHED_SELECTOR = torch.tensor(cached_selector).bool()
    MODEL_NEEDS_UPDATE_IMU_COORDINATES = True
    assert sum(IMU_ATTACHED_SELECTOR) == len(cached_coordinates)


from aitviewer.remote.viewer import RemoteViewer

viewer_remote = None


# Run server in its own event loop in background
def start_server():
    global viewer_remote, SERVER_ACTIVE_STATUS
    viewer_remote = RemoteViewer()

    def setup_logging():
        """Setup basic logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    logger = setup_logging()

    # Setup TCP action listener
    action_listener_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    action_listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    action_listener_socket.bind(("0.0.0.0", 9000))
    action_listener_socket.listen(1)
    logger.info("Waiting for action listener connection...")
    action_listener, addr = action_listener_socket.accept()
    logger.info(f"Action listener connected from {addr}")

    # Make action listener non-blocking
    action_listener.setblocking(False)

    # Setup UDP sockets for sensor data
    socks = []
    sock_ports = {}  # Map socket to port number for easier tracking

    def create_udp_socket(port_num):
        """Create and configure a UDP socket for the given port."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((SERVER_IP, port_num))
            sock.setblocking(False)  # Make non-blocking from the start
            logger.info(f"UDP Motion Server on {SERVER_IP}:{port_num}")

            # Clear any existing data in buffer
            try:
                while True:
                    _ = sock.recvfrom(1024)
                    logger.info('Clearing buffer...')
            except BlockingIOError:
                pass  # Buffer is clear

            return sock
        except Exception as e:
            logger.error(f"Failed to create UDP socket on port {port_num}: {e}")
            return None

    for port_num in SERVER_PORTS:
        sock = create_udp_socket(port_num)
        if sock:
            socks.append(sock)
            sock_ports[sock] = port_num

    logger.info('--- Started main loop ---')
    SERVER_ACTIVE_STATUS = {p: True for p in SERVER_PORTS}

    # Create list of all sockets to monitor
    all_sockets = [action_listener] + socks
    connected_devices = set()
    device_last_seen = {}  # Track when each device was last seen
    last_connect_confirm_time = 0

    end_conn = False
    try:
        while not end_conn:
            try:
                # Use select to wait for any socket to have data ready
                # Timeout of 0.1 seconds to allow periodic checks
                ready_sockets, _, error_sockets = select.select(all_sockets, [], all_sockets, 0.1)

                # Handle any sockets with errors
                for sock in error_sockets:
                    logger.error("Socket error detected")
                    if sock == action_listener:
                        logger.error("Action listener socket error")
                        end_conn = True
                        break
                    else:
                        # Handle UDP socket error - try to recreate the socket
                        if sock in sock_ports:
                            port_num = sock_ports[sock]
                            logger.error(f"UDP socket error on port {port_num}, attempting to recreate...")

                            # Remove the bad socket
                            socks.remove(sock)
                            del sock_ports[sock]
                            sock.close()

                            # Try to recreate the socket
                            new_sock = create_udp_socket(port_num)
                            if new_sock:
                                socks.append(new_sock)
                                sock_ports[new_sock] = port_num
                                SERVER_ACTIVE_STATUS[port_num] = True
                                logger.info(f"Successfully recreated UDP socket on port {port_num}")
                            else:
                                logger.error(f"Failed to recreate UDP socket on port {port_num}")
                                SERVER_ACTIVE_STATUS[port_num] = False

                # Process ready sockets
                for sock in ready_sockets:
                    if sock == action_listener:
                        # Handle TCP action commands
                        try:
                            data = sock.recv(1024)  # Use recv() for TCP, not recvfrom()
                            if not data:
                                logger.info("Action listener disconnected")
                                end_conn = True
                                break
                            # Process action commands
                            if data == b"START_T_POSE_CALIBRATION" or data == b"START_FLOOR_ALIGN":
                                _start_data_buffer_on_server()
                            elif data == b"PROCESS_T_POSE_CALIBRATION":
                                _update_t_pose_calibration_from_buffer_on_server()
                            elif data == b"PROCESS_FLOOR_ALIGN":
                                _update_floor_alignment_from_buffer_on_server()

                            elif data == b"UPDATE_COORDINATES":
                                _update_device_coordinates_on_server()

                            elif data == b'START_INFERENCE':
                                INFERENCE_ON.set()
                                print("Starting inference...")
                            elif data == b'STOP_INFERENCE':
                                INFERENCE_ON.clear()
                                print("Stopping inference...")
                            else:
                                logger.warning(f"Unknown action command: {data}")

                        except Exception as e:
                            logger.error(f"Error processing action command: {e}")

                    else:
                        # Handle UDP sensor data
                        try:
                            if sock not in sock_ports:
                                continue

                            port_num = sock_ports[sock]

                            if not SERVER_ACTIVE_STATUS[port_num]:
                                continue

                            data, addr = sock.recvfrom(UDP_SENSOR_BUFFER_SIZE)
                            if not data:
                                continue

                            # Try to decode as a string message
                            try:
                                message = data.decode('utf-8')
                                # logger.info(f"Received message from port {port_num}: {message}")
                                rt = handle_socket_message(port_num, message)
                                if rt and rt != 'INIT' and rt != 'UNKNOWN':
                                    if rt not in connected_devices:
                                        print("Device connected", rt)
                                        connected_devices.add(rt)
                                        # Send device connected message via TCP
                                        try:
                                            action_listener.send(f"DEVICE_CONNECTED:{rt}".encode('utf-8'))
                                        except Exception as e:
                                            logger.error(f"Error sending device connected message: {e}")

                                    # Update last seen time for this device
                                    device_last_seen[rt] = time.time()

                            except UnicodeDecodeError:
                                # Not a string message, might be binary data
                                logger.debug(f"Received binary data from port {port_num} (ignored)")

                        except Exception as e:
                            logger.error(f"Error processing UDP data: {e}")
                            # If there's an error with this socket, mark it for recreation
                            if sock in sock_ports:
                                port_num = sock_ports[sock]
                                logger.error(f"Marking port {port_num} for recreation due to error")
                                SERVER_ACTIVE_STATUS[port_num] = False

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                end_conn = True
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                continue

            # Check for device disconnections (devices that haven't been seen for 10 seconds)
            current_time = time.time()
            devices_to_disconnect = []
            for device_id, last_seen in device_last_seen.items():
                if current_time - last_seen > 10.0:  # 10 second timeout
                    devices_to_disconnect.append(device_id)

            for device_id in devices_to_disconnect:
                if device_id in connected_devices:
                    print("Device disconnected", device_id)
                    connected_devices.remove(device_id)
                    del device_last_seen[device_id]
            if len(devices_to_disconnect) > 0:
                # Send device disconnected message via TCP
                try:
                    print(f"sending message to disconnect device {'+'.join(devices_to_disconnect)}")
                    action_listener.send(f"DEVICE_DISCONNECTED:{'+'.join(devices_to_disconnect)}".encode('utf-8'))
                except Exception as e:
                    logger.error(f"Error sending device disconnected message: {e}")

            if len(connected_devices) > 0 and current_time - last_connect_confirm_time > 10.0:
                try:
                    print(f"sending device connected {'+'.join(connected_devices)} message.")
                    action_listener.send(f"DEVICE_CONNECTED:{'+'.join(connected_devices)}".encode('utf-8'))
                except Exception as e:
                    logger.error(f"Error sending device connected message: {e}")
                last_connect_confirm_time = current_time

            # Check for inactive UDP sockets and try to recreate them
            for port_num in SERVER_PORTS:
                if not SERVER_ACTIVE_STATUS[port_num]:
                    # Check if we already have a socket for this port
                    existing_sock = None
                    for sock in socks:
                        if sock in sock_ports and sock_ports[sock] == port_num:
                            existing_sock = sock
                            break

                    if not existing_sock:
                        # Try to recreate the socket
                        logger.info(f"Attempting to recreate UDP socket for port {port_num}")
                        new_sock = create_udp_socket(port_num)
                        if new_sock:
                            socks.append(new_sock)
                            sock_ports[new_sock] = port_num
                            SERVER_ACTIVE_STATUS[port_num] = True
                            logger.info(f"Successfully recreated UDP socket on port {port_num}")
                        else:
                            logger.error(f"Failed to recreate UDP socket on port {port_num}")

    finally:
        # Clean up all sockets
        action_listener.close()
        action_listener_socket.close()
        for sock in socks:
            sock.close()
        logger.info("All sockets closed")

    logger.info('--- Stopped main loop ---')


# Global variable to store debug IMU cube meshes and axes
debug_imu_cubes = {}
debug_imu_axes = {}

def _create_axis_meshes():
    # Create a simple cube mesh programmatically (1x1x1 cube centered at origin)
    cube_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    
    # Scale the cube to create arrow shapes
    # X axis: stretch along X, compress along Y and Z
    x_vertices = cube_mesh.vertices.copy()
    x_vertices[:, 0] *= 0.3  # Stretch along X
    x_vertices[:, 1] *= 0.05  # Compress along Y
    x_vertices[:, 2] *= 0.05  # Compress along Z
    
    # Y axis: stretch along Y, compress along X and Z
    y_vertices = cube_mesh.vertices.copy()
    y_vertices[:, 0] *= 0.05  # Compress along X
    y_vertices[:, 1] *= 0.3  # Stretch along Y
    y_vertices[:, 2] *= 0.05  # Compress along Z
    
    # Z axis: stretch along Z, compress along X and Y
    z_vertices = cube_mesh.vertices.copy()
    z_vertices[:, 0] *= 0.05  # Compress along X
    z_vertices[:, 1] *= 0.05  # Compress along Y
    z_vertices[:, 2] *= 0.3  # Stretch along Z
    
    return (x_vertices, cube_mesh.faces, (1.0, 0.0, 0.0, 1.0)), (y_vertices, cube_mesh.faces, (0.0, 1.0, 0.0, 1.0)), (z_vertices, cube_mesh.faces, (0.0, 0.0, 1.0, 1.0))

def _visualize_imu_orientations_debug(glb_ori, attached_device_ids, viewer_remote):
    """
    Visualize IMU orientations as cubes with XYZ axes when debug mode is enabled.
    
    Args:
        glb_ori: Tensor of shape (1, num_devices, 3, 3) containing rotation matrices
        attached_device_ids: List of device IDs that are attached
        viewer_remote: Remote viewer instance for rendering
    """
    global debug_imu_cubes, debug_imu_axes
    
    if viewer_remote is None:
        return
    
    try:
        # Create a simple cube mesh programmatically (1x1x1 cube centered at origin)
        cube_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        
        # Create axis meshes
        x_axis_data, y_axis_data, z_axis_data = _create_axis_meshes()
        
        # Get the number of devices
        num_devices = glb_ori.shape[1]
        
        # Create or update cube meshes and axes for each device
        for i, device_id in enumerate(attached_device_ids):
            if i >= num_devices:
                break
                
            # Get the rotation matrix for this device (3x3)
            rotation_matrix = glb_ori[0, i].cpu().numpy()
            
            # Create unique names for this device's cube and axes
            cube_name = f"Debug_IMU_Cube_{device_id}"
            x_axis_name = f"Debug_IMU_X_Axis_{device_id}"
            y_axis_name = f"Debug_IMU_Y_Axis_{device_id}"
            z_axis_name = f"Debug_IMU_Z_Axis_{device_id}"
            
            # Position cubes in a line for easy viewing
            position = (i * 0.5 - 0.75, 2, 0)  # Space them 0.5 units apart
            
            # Get device color
            device_id_digit = int(device_id.replace("IMU-", ""))
            color = IMU_DEVICE_COLORS[device_id_digit]
            
            # Create or update cube mesh
            if cube_name not in debug_imu_cubes:
                # Create new cube mesh
                cube_mesh_obj = RemoteMeshes(
                    viewer_remote,
                    cube_mesh.vertices,
                    cube_mesh.faces,
                    name=cube_name,
                    position=position,
                    scale=0.1,  # Make cubes smaller
                    color=color,
                    flat_shading=True,
                )
                debug_imu_cubes[cube_name] = cube_mesh_obj
            else:
                # Update existing cube mesh with new orientation
                cube_mesh_obj = debug_imu_cubes[cube_name]
                # Apply rotation to the cube vertices
                rotated_vertices = cube_mesh.vertices @ rotation_matrix.T
                cube_mesh_obj.update_frames(rotated_vertices[np.newaxis], 0)
            
            # Create or update X axis (red)
            x_vertices, x_faces, x_color = x_axis_data
            if x_axis_name not in debug_imu_axes:
                x_axis_obj = RemoteMeshes(
                    viewer_remote,
                    x_vertices,
                    x_faces,
                    name=x_axis_name,
                    position=position,
                    color=x_color,
                    flat_shading=True,
                )
                debug_imu_axes[x_axis_name] = x_axis_obj
            else:
                x_axis_obj = debug_imu_axes[x_axis_name]
                rotated_x_vertices = x_vertices @ rotation_matrix.T
                x_axis_obj.update_frames(rotated_x_vertices[np.newaxis], 0)
            
            # Create or update Y axis (green)
            y_vertices, y_faces, y_color = y_axis_data
            if y_axis_name not in debug_imu_axes:
                y_axis_obj = RemoteMeshes(
                    viewer_remote,
                    y_vertices,
                    y_faces,
                    name=y_axis_name,
                    position=position,
                    color=y_color,
                    flat_shading=True,
                )
                debug_imu_axes[y_axis_name] = y_axis_obj
            else:
                y_axis_obj = debug_imu_axes[y_axis_name]
                rotated_y_vertices = y_vertices @ rotation_matrix.T
                y_axis_obj.update_frames(rotated_y_vertices[np.newaxis], 0)
            
            # Create or update Z axis (blue)
            z_vertices, z_faces, z_color = z_axis_data
            if z_axis_name not in debug_imu_axes:
                z_axis_obj = RemoteMeshes(
                    viewer_remote,
                    z_vertices,
                    z_faces,
                    name=z_axis_name,
                    position=position,
                    color=z_color,
                    flat_shading=True,
                )
                debug_imu_axes[z_axis_name] = z_axis_obj
            else:
                z_axis_obj = debug_imu_axes[z_axis_name]
                rotated_z_vertices = z_vertices @ rotation_matrix.T
                z_axis_obj.update_frames(rotated_z_vertices[np.newaxis], 0)
                
    except Exception as e:
        print(f"Error in debug IMU visualization: {e}")


# Inference loop runs in separate thread
def start_inference():
    global MODEL_NEEDS_UPDATE_IMU_COORDINATES

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Inference thread started, waiting for remote viewer...")
    while viewer_remote is None:
        time.sleep(0.1)  # Small delay to avoid busy waiting
    time.sleep(5)  # let the viewer remote to finish

    print("Remote viewer ready, starting inference...")
    identity_matrix = torch.eye(3).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 3, 3)
    t_pos_smpl = identity_matrix.repeat(1, 24, 1, 1)  # Shape (1, 24, 3, 3)
    initial_glb_pose, initial_joints_positions, initial_vertex_positions = imu_config.body_model.forward_kinematics(pose=t_pos_smpl, calc_mesh=True)
    pose_remote_mesh = RemoteMeshes(
        viewer_remote,
        initial_vertex_positions.detach().numpy(),
        imu_config.body_model.face,
        is_selectable=False,
        gui_affine=False,
        name=f"IMUCoCo Pose Prediction",
    )

    vertex_coordinates_with_category = torch.tensor(imu_config.vertex_coordinates_with_category).float().to(device)
    joint_coordinates_with_category = torch.tensor(imu_config.joint_coordinates_with_category).float().to(device)


    # the max value of each dimension in the vertex_position_encoding_matrix
    coordinate_max, coordinate_min = (torch.max(vertex_coordinates_with_category[:, 1:], dim=0).values,
                                      torch.min(vertex_coordinates_with_category[:, 1:], dim=0).values)

    imu_coco_model = IMUCoCo(
        coordinate_origins=joint_coordinates_with_category,
        coordinate_max=coordinate_max,
        coordinate_min=coordinate_min,
        smpl_mesh_coordinates = vertex_coordinates_with_category,
        n_hidden=128,
        n_kr_hidden=32,
        n_mfe_layers=2,
        n_jnm_layers=3,
        n_sce_freq=4,
        n_sce_emb=40,
        online_mode=True,
        joint_node_allocation_map=path_config.saved_imucoco_loss_map_path,
        joint_node_max_err_tolerance=-1,
    ).to(device)
    poser_model = Poser(joint_feature_dim=128, 
                        n_hidden=300, 
                        n_glb=40, 
                        num_layer=3,
                        n_total_devices=24, 
                        load_tran_module=True).to(device)
    imu_coco_model.load_offline_state_dict_to_online_model(torch.load(path_config.saved_imucoco_checkpoint_path, map_location=device))
    imu_coco_model.eval()
    poser_model.load_state_dict(torch.load(path_config.saved_hpe_checkpoint_path, map_location=device), strict=False)
    poser_model.eval()

    # use parallel implementatation for live inference
    imu_coco_model.prepare_parallel_sce_implementation()
    imu_coco_model.prepare_parallel_joint_node_implementation()
    imu_coco_model.buffer_placement_codes_with_current_devices(parallel=True)

    init_t_pose = torch.eye(3).float().expand(24, 3, 3).unsqueeze(0)
    init_t_pose_r6d = init_t_pose[:, :, :, :2].transpose(2, 3).flatten(2)
    init_t_pose_r6d = init_t_pose_r6d.float().to(device)
    h_mfe, h_jnm = None, None
    h_pose = poser_model.init_hidden_states(
        v_init=torch.zeros(1, 24, 3).float().to(device),
        glb_init=init_t_pose_r6d
    )
    current_tran = None
    clock = Clock()
    infer_frame_count = 0

    print("Inference loop ready, waiting for INFERENCE_ON event...")
    prev_mesh = None
    attached_device_ids = [device_id for i, device_id in enumerate(INFERENCE_DATA.keys()) if IMU_ATTACHED_SELECTOR[i]]
    while True:
        clock.tick(60)
        if MODEL_NEEDS_UPDATE_IMU_COORDINATES:
            print("model update coordinates", IMU_COORDINATES.shape, IMU_COORDINATES)
            if len(IMU_COORDINATES) > 0:
                imu_coco_model.set_current_device_coordinates(IMU_COORDINATES.to(device))
                imu_coco_model.buffer_placement_codes_with_current_devices(parallel=True)
                attached_device_ids = [device_id for i, device_id in enumerate(INFERENCE_DATA.keys()) if IMU_ATTACHED_SELECTOR[i]]
                print("attached_device_ids", attached_device_ids)
            MODEL_NEEDS_UPDATE_IMU_COORDINATES = False

        if INFERENCE_ON.is_set():
            # if True:
            if infer_frame_count == 0:  # First frame of inference
                pose_remote_mesh.color = (0.5, 0.5, 0.5, 1.0)  # make the calibration pose invisible to show the inference pose
                print("Inference started! Processing motion data...")

            glb_ori_list = []
            glb_acc_list = []
            for device_id in attached_device_ids:
                motion_data = INFERENCE_DATA[device_id]
                if motion_data is not None:
                    glb_ori, acc_glb = _imu_calibration(motion_data, device_id)
                    glb_ori_list.append(glb_ori)
                    glb_acc_list.append(acc_glb)
                else:
                    raise ValueError(f"No motion data available for device {device_id}")

            if not glb_ori_list or len(glb_ori_list) == 0:
                continue  # Skip if no data available
            glb_ori = torch.stack(glb_ori_list, dim=0).unsqueeze(0)
            glb_acc = torch.stack(glb_acc_list, dim=0).unsqueeze(0)

            # Debug visualization of IMU orientations as cubes
            if DEBUG_IMU_ORIENTATION:
                _visualize_imu_orientations_debug(glb_ori, attached_device_ids, viewer_remote)

            glb_r6d = glb_ori[:, :, :, :2].transpose(2, 3).flatten(2)  # (T, D, 6)
            
            imu_tensor = torch.cat([glb_r6d, glb_acc], dim=2).unsqueeze(0).to(device)  # (B=1, T, D, 9)

            feat_m, h_mfe, h_jnm = imu_coco_model.inference_time_forward_mesh_online(imu_tensor, h_mfe, h_jnm)

            pose_local_pred, _, tran_pred, h_pose= poser_model.forward_online(feat_m, h_pose, current_tran=current_tran, compute_tran='transpose')

            _, _, pose_mesh = imu_config.body_model.forward_kinematics(pose_local_pred[0].cpu(), tran=tran_pred.cpu(), calc_mesh=True)
            pose_mesh = pose_mesh[-1:].detach().numpy()
            pose_remote_mesh.update_frames(pose_mesh, 0)
            infer_frame_count += 1

            current_tran = tran_pred

            if infer_frame_count % 50 == 0:
                print(f'\tOutput FPS: {clock.get_fps():.1f}')
            if infer_frame_count > 100000:
                infer_frame_count = 1
        else:
            # Inference is not active
            if infer_frame_count > 0:  # Was running before, now stopped
                print("Inference stopped.")
                pose_remote_mesh.update_frames(initial_vertex_positions[0].cpu().numpy(), 0)
                # Clear inference data when stopping (no locking needed)
                infer_frame_count = 0
                h_mfe, h_jnm = None, None
                h_pose = poser_model.init_hidden_states(
                    v_init=torch.zeros(1, 24, 3).to(device),
                    glb_init=init_t_pose_r6d
                )
                current_tran = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--viewer", help="run the pose viewer. you should not need to to turn this on my hand, as it will starts the pose viewer automatically", action="store_true"
    )
    args = parser.parse_args()
    if not args.viewer:
        # Start inference thread first (before server blocks)
        inference_thread = threading.Thread(target=start_inference, daemon=True)
        inference_thread.start()
        print("Inference thread started")
        start_server()
    else:
        start_pose_viewer()
