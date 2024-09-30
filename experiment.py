import torch
import logging
import threading
import multiprocessing
import queue

from data.data_acquisition import DataAcquisition
from model.model_trainer import ModelTrainer
from model.bayesian_optimizer import BayesianOptimizer
from pretrained.detr.datasets.coco import make_coco_transforms
from data.data_utils import process_audio, process_touch, process_eeg, play_wave_file
from utils import get_screen


class EEGExperiment:
    def __init__(self, args, config):
        self.args = args
        self.device = args.device

        # Initialize components
        self.data_acquisition = DataAcquisition(args)
        self.model_trainer = ModelTrainer(args, config)
        self.bayesian_optimizer = BayesianOptimizer(args)
        self.transform = make_coco_transforms('val')
        self.gui = None

        # EEG trial-specific variables
        self.trial_active = False
        self.trial_counter = 1  # start Python first and then start game to keep consistent
        self.max_trials = args.max_trials if hasattr(args, 'max_trials') else 200
        self.timestamp_marker = None

        self.raw_eeg_data = []
        self.triggers = []
        self.label = []
        
        self.input_img = None
        self.audio = None
        self.audio_name = None
        self.touch = None
        
        # Queues for communication with GUI or other processes
        self.q_rcv = multiprocessing.Queue()
        self.q_send = multiprocessing.Queue()
        self.lock = threading.Lock()

    def start_gui(self):
        if self.args.mode == 'calibration': # Calibration task
            from robotaxi_calibration.play import start_gui
            start_gui(self.args)
        elif self.args.mode in ['navigation', 'surveillance']: # Main experimental task
            from robotaxi_integration.play import start_gui
            # Start GUI in a separate process
            self.gui = multiprocessing.Process(
                target=start_gui, args=(self.args, self.q_send, self.q_rcv)
            )
            self.gui.start()
            # or use threading instead of multiprocessing?
            # gui_thread = threading.Thread(target=start_gui, args=(args,))
            # gui_thread.start()
        else:
            logging.error("Invalid mode selected. Please choose 'calibration', 'navigation', or 'surveillance'.")
            raise ValueError("Invalid mode selected.")

    def process_markers(self):
        while not self.data_acquisition.marker_queue.empty():
            marker, timestamp = self.data_acquisition.marker_queue.get()
            logging.info(f"Received marker: {marker} at time {timestamp}")

            if marker == 'T 1' and not self.trial_active:
                # Start of trial
                self.trial_active = True
                self.timestamp_marker = timestamp
                logging.info(f"Trial {self.trial_counter} started.")

            elif marker == 'T 2' and self.trial_active:
                # End of trial
                self.trial_active = False
                logging.info(f"Trial {self.trial_counter} ended.")

            elif marker.startswith('E ') and self.trial_active:  # Error event
                # Sensory stimulus (warning signal) to participant
                self.process_sensory_stimulus(marker)

            elif marker.startswith('R ') and self.trial_active:
                # Behavioral response (button press) from participant
                self.process_response(marker)

            # Store markers for EEG processing
            self.triggers.append((marker, timestamp))

    def collect_eeg_data(self):
        while not self.data_acquisition.eeg_queue.empty():
            sample, timestamp = self.data_acquisition.eeg_queue.get()
            self.raw_eeg_data.append((sample, timestamp))

    def capture_screen(self, source):
        print("Get screenshot")
        screenshot, _ = self.transform(source, None)
        if self.input_img is None:
            self.input_img = screenshot.unsqueeze(0)
        else:   
            self.input_img = torch.cat((self.input_img, screenshot.unsqueeze(0)), dim=0)

    def reset_trial_data(self):
        self.raw_eeg_data = []
        self.markers = []
        self.input_img = []
        self.labels = []
        self.audio = None
        self.touch = None

    def run(self):
        # Start GUI
        self.start_gui()

        # Start LSL data acquisition
        self.data_acquisition.start()

        try:
            while self.running and self.trial_counter <= self.max_trials:
                self.run_trial()
                self.trial_counter += 1
        except KeyboardInterrupt:
            logging.info("Experiment interrupted by user.")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        finally:
            self.shutdown()

    def run_trial(self):
        marker, timestamp = self.data_acquisition.marker_queue.get(timeout=1.0)
        with self.lock:
            if self.timestamp_marker is None:
                self.timestamp_marker = float(str(timestamp))
            if marker is not None:
                print(marker)
                trigger = marker[0]
                self.triggers.append((trigger, timestamp))
                logging.info(f"Received marker: {trigger} at time {(timestamp - self.timestamp_marker):.2f}")

                if trigger == 'T 1':  # Start of trial
                    self.trial_active = True
                    # Perform Bayesian Optimization on the stimuli parameters
                    stim_params = self.bayesian_optimizer.run(self.trial_counter)

                    # Prepare stimuli based on optimized parameters
                    self.audio, self.audio_name = process_audio(self.args, stim_params)
                    self.touch = process_touch(stim_params)
                    
                elif trigger == 'T 2' and bool(self.raw_eeg_data) & self.trial_active:  # End of trial
                    self.on_trial_end(timestamp)
                    
                elif trigger.startswith('E ') and self.trial_active:  # Error event
                    self.q_send.put(self.touch)
                    play_thread = threading.Thread(target=play_wave_file, args=(self.audio_name,))
                    play_thread.start()

                elif trigger.startswith('R ') and self.trial_active:  # Response (button press) 
                    self.label.append(trigger)
                    try:
                        self.capture_screen(self.q_rcv.get(timeout=1.0))   
                    except queue.Empty:
                        logging.error("Queue is empty, no screenshot available.")

                elif (trigger[:2] == 'KP') & bool(self.raw_eeg_data) & self.trial_active:  # get current screenshot
                    if self.trial_counter % 2 != 0:  # Multisensory input
                        # play audio here
                        play_thread = threading.Thread(target=play_wave_file, args=(self.audio_name,))
                        play_thread.start()
                    self.capture_screen(get_screen(self.args, 'Trial_' + str(self.trial_counter) + '_KP_' + trigger[3]))
            else:
                logging.debug(f"Unknown marker: {marker} at {timestamp}")

        # # Wait for the 'T 1' marker to start the trial
        # while not self.trial_active:
        #     self.process_markers()
        #     threading.Event().wait(0.01)

        # # Trial is active
        # while self.trial_active:
        #     self.process_markers()
        #     self.collect_eeg_data()
        #     threading.Event().wait(0.01)

        # # Trial has ended
        # self.on_trial_end()

    def on_trial_end(self):
        logging.info(f"Trial {self.trial_counter} ended.")
        if self.input_img is not None and len(self.raw_eeg_data) > 0:
            logging.info("Processing EEG data...")
            epoched_data = process_eeg(self.raw_eeg_data, self.triggers)
            # Prepare training data
            training_data = (
                self.trial_counter,
                epoched_data,
                self.label,
                self.input_img,
                self.audio,
                self.touch,
                self.next_point_to_probe
            )
            # Train the model and get the target value for BO
            target = self.model_trainer.train(training_data)
            # Register the result with the Bayesian Optimizer
            self.bayesian_optimizer.register_result(target)
        else:
            logging.info("No valid data collected during trial. Skipping training.")

        self.trial_counter += 1
        self.trial_active = False

        # Reset variables for the next trial
        self.reset_trial_data()

    def shutdown(self):
        logging.info("Shutting down the experiment.")
        self.data_acquisition.stop()
        if self.gui:
            self.gui.join()




    # def run_trial(self):
    #     logging.info(f"Starting trial {self.trial_counter}")

    #     # Get next parameters from Bayesian Optimization
    #     next_params = self.bayesian_optimizer.suggest_next()
    #     self.data_processor.reset_trial_data()

    #     # Prepare stimuli based on parameters
    #     audio, touch = self.data_processor.prepare_stimuli(next_params)
    #     self.data_processor.audio = audio
    #     self.data_processor.touch = touch

    #     # Send touch data to GUI or hardware if needed
    #     if self.q_send:
    #         self.q_send.put(touch)

    #     # Start collecting data
    #     self.data_acquisition.start_trial()
    #     trial_active = True

    #     while trial_active:
    #         # Get markers and EEG data
    #         markers = self.data_acquisition.get_markers()
    #         eeg_data = self.data_acquisition.get_eeg_data()

    #         # Process markers
    #         for marker in markers:
    #             if marker == 'T 1':
    #                 # Trial start already handled
    #                 pass
    #             elif marker == 'T 2':
    #                 # End of trial
    #                 trial_active = False
    #                 break
    #             else:
    #                 self.data_processor.process_marker(marker, self.q_rcv)

    #         # Collect EEG data
    #         self.data_processor.collect_eeg_data(eeg_data)

    #     # End of trial processing
    #     training_data = self.data_processor.get_training_data()
    #     if training_data:
    #         target = self.model_trainer.train(training_data)
    #         self.bayesian_optimizer.register_result(target)
    #     else:
    #         logging.info("No valid data collected during trial. Skipping training.")

    #     logging.info(f"Trial {self.trial_counter} completed.")

