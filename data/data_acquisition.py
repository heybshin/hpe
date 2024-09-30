from pylsl import StreamInlet, resolve_stream
from collections import deque
import queue
import threading
import logging
import time
from queue import Queue, Empty

# class RingBuffer:
#     def __init__(self, size, dtype=float):
#         self.size = size
#         self.buffer = [None] * size
#         self.dtype = dtype
#         self.write_pos = 0
#         self.read_pos = 0
#         self.count = 0
#         self.lock = threading.Lock()

#     def write(self, data):
#         with self.lock:
#             for item in data:
#                 self.buffer[self.write_pos] = item
#                 self.write_pos = (self.write_pos + 1) % self.size
#                 if self.count < self.size:
#                     self.count += 1
#                 else:
#                     # Overwrite oldest data
#                     self.read_pos = (self.read_pos + 1) % self.size

#     def read(self, num_items):
#         with self.lock:
#             if self.count == 0:
#                 return []
#             data = []
#             for _ in range(min(num_items, self.count)):
#                 data.append(self.buffer[self.read_pos])
#                 self.read_pos = (self.read_pos + 1) % self.size
#                 self.count -= 1
#             return data

#     def is_empty(self):
#         with self.lock:
#             return self.count == 0

#     def clear(self):
#         with self.lock:
#             self.write_pos = 0
#             self.read_pos = 0
#             self.count = 0
#             self.buffer = [None] * self.size


class Buffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, data):
        with self.lock:
            self.buffer.extend(data)

    def extend(self, data_list):
        with self.lock:
            self.buffer.extend(data_list)

    def get_data(self, num_samples):
        with self.lock:
            if len(self.buffer) < num_samples:
                return []
            data = [self.buffer.popleft() for _ in range(num_samples)]
            return data

    def get_all(self):
        with self.lock:
            data = list(self.buffer)
            self.buffer.clear()
            return data
    
    def get_latest(self, num_samples):
        with self.lock:
            if len(self.buffer) < num_samples:
                return list(self.buffer)
            else:
                return list(deque(self.buffer, maxlen=num_samples))
            
    def clear(self):
        with self.lock:
            self.buffer.clear()

    def __len__(self):
        with self.lock:
            return len(self.buffer)


class DataAcquisition:
    def __init__(self, args):
        self.args = args
        self.running = False

        # Initialize LSL streams
        self.eeg_inlet = self.connect_lsl_stream('type', 'EEG')
        self.marker_inlet = self.connect_lsl_stream('name', 'Robotaxi ' + args.mode)

        # Initialize Buffers
        self.eeg_buffer = Buffer(maxlen=256*10)      # 10-second buffer
        self.marker_queue = deque()                    # Queue for marker events
        self.marker_lock = threading.Lock()
        # self.marker_queue = queue.Queue()

        # Initialize Threads
        self.eeg_thread = threading.Thread(target=self._stream_eeg, daemon=True)
        self.marker_thread = threading.Thread(target=self._stream_markers, daemon=True)

    def connect_lsl_stream(self, key, value, timeout=5):
        streams = resolve_stream(key, value, timeout=timeout)
        if streams:
            return StreamInlet(streams[0])
        else:
            raise RuntimeError(f"No LSL stream found with {key} = {value}")
            # logging.error(f"No LSL stream found with {key} = {value}")
            # return None

    def start(self):
        self.running = True
        self.eeg_thread.start()
        self.marker_thread.start()
        logging.info("Data acquisition started.")

    def stop(self):
        self.running = False
        self.eeg_thread.join()
        self.marker_thread.join()
        logging.info("Data acquisition stopped.")

    def _stream_eeg(self):      # def eeg_listener(self):
        while self.running:
            try:
                samples, timestamps = self.eeg_inlet.pull_chunk(max_samples=256, timeout=1.0)
                if samples:
                    for sample, timestamp in zip(samples, timestamps):
                        self.eeg_buffer.append((sample, timestamp))
            except Exception as e:
                logging.error(f"Error in EEG streaming: {e}")
                time.sleep(1)

    def _stream_markers(self):   # def marker_listener(self):
        while self.running:
            try:
                marker, timestamp = self.marker_inlet.pull_sample(timeout=1.0)
                if marker:
                    self.marker_queue.put((marker[0], timestamp))
            except Exception as e:
                logging.error(f"Error in Marker streaming: {e}")
                time.sleep(1)

    # def get_marker_event(self, timeout=1.0):
    #     """Waits for a marker event with an optional timeout."""
    #     try:
    #         return self.marker_queue.get(timeout=timeout)
    #     except queue.Empty:
    #         return None

    # def collect_data(self):
    #     # Retrieve data from the EEG queue
    #     while not self.eeg_queue.empty():
    #         sample, timestamp = self.eeg_queue.get()
    #         self.raw_eeg_data.append((sample, timestamp))
        
    #     # Retrieve markers from the marker queue
    #     while not self.marker_queue.empty():
    #         marker, timestamp = self.marker_queue.get()
    #         self.triggers.append((marker, timestamp))
        

    # def processed_eeg_data(self):
    #     # Process raw EEG data into epochs
    #     if self.raw_eeg_data and self.triggers:
    #         return process_eeg(self.raw_eeg_data, self.triggers)
    #     return None
    
    # def reset_trial_data(self):
    #     # Reset internal data buffers after each trial
    #     self.raw_eeg_data = []
    #     self.triggers = []




# class DataAcquisition:
#     def __init__(self, args):
#         # Initialize LSL streams
#         self.eeg_inlet = self.connect_lsl_stream('type', 'EEG')
#         self.marker_inlet = self.connect_lsl_stream('name', 'Robotaxi ' + args.mode)
#         self.collecting = False

#     def connect_lsl_stream(self, key, value):
#         streams = resolve_stream(key, value)
#         if streams:
#             return StreamInlet(streams[0])
#         else:
#             raise RuntimeError(f"No LSL stream found with {key} = {value}")

#     def start_trial(self):
#         self.collecting = True
#         self.eeg_data_buffer = []
#         self.marker_buffer = []

#     def get_eeg_data(self):
#         if self.collecting:
#             samples, timestamps = self.eeg_inlet.pull_chunk()
#             if samples:
#                 self.eeg_data_buffer.extend(zip(samples, timestamps))
#             return self.eeg_data_buffer
#         return []

#     def get_markers(self):
#         markers = []
#         while True:
#             marker_sample, timestamp = self.marker_inlet.pull_sample(timeout=0.0)
#             if marker_sample:
#                 markers.append(marker_sample[0])
#             else:
#                 break
#         return markers

#     def stop(self):
#         self.collecting = False
#         # Close LSL inlets if necessary
