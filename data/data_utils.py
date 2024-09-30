import os

import simpleaudio as sa
from datasets import load_dataset
import shutil
import numpy as np
import mne
from meegkit import dss
import torch
from transformers import AutoFeatureExtractor


instrument = {
    0: "GrandPiano", 1: "EpicCloudFormation", 2: "FingerstyleBass", 3: "TweedPickedSynth",
    4: "StringEnsemble", 5: "SessionStrings", 6: "FullBrass", 7: "GrandPiano", 8: "GrandPiano",
    9: "SoCal", 10: "GrandPiano", 11: "ClassicElectricPiano", 12: "GrandPiano",
    13: "ReverseEngineering", 14: "GrandPiano", 15: "DistantAir"
}

params = {
    'timbre': ['sine', 'sawtooth', 'square'],
    'note': ['C2', 'C3', 'C4', 'C5', 'C6'],
    'volume': ['1', '64', '127'],
    'pattern': ['1', '2', '3'],
    'duration': [0.2, 0.4, 0.6],
    'intensity': [0.1, 0.2, 0.5, 0.8, 1.0],
}

mapping_timbre = {0: 'sine', 1: 'square', 2: 'sawtooth'}
mapping_note = {0: 'C2', 1: 'C3', 2: 'C4', 3: 'C5', 4: 'C6'}
mapping_volume = {0: 'Low', 1: 'Medium', 2: 'High'}
mapping_pattern = {0: 'SingleBeep', 1: 'TwoBeeps', 2: 'ThreeBeeps'}
mapping_duration = {0: 'Short', 1: 'Medium', 2: 'Long'}
mapping_intensity = {0: 0.1, 1: 0.2, 2: 0.5, 3: 0.8, 4: 1.0}


def is_all_zero(audio_array, threshold=0.1):
    # Function to check if an audio sample starts with all zeros
    return np.all(audio_array[:int(threshold * len(audio_array))] == 0)

def process_eeg(raw_data, triggers):
    """
    This function chunks the raw EEG data based on 'KP' triggers.
    Each chunk starts from 1 second before the trigger to 1 second after the trigger.

    Parameters:
    raw_data (list): List of tuples where each tuple is (sample, timestamp).
    triggers (list): List of tuples where each tuple is (trigger, timestamp).

    Returns:
    epoched_data (torch.Tensor): 4D tensor of shape (num_epochs, 1, num_channels, num_samples).
    """

    # Convert raw data to numpy array for easier processing
    # raw_data_array = np.array([sample for sample, timestamp in raw_data])
    raw_data_array = np.array(raw_data[0])
    raw_data_array = raw_data_array.T  # Shape: [num_channels, num_samples]
    ### need further process like re-referencing, filtering, etc.
    data = mne.io.RawArray(raw_data_array,
                           mne.create_info(ch_names=['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
                                                     'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
                                                     'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4',
                                                     'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1',
                                                     'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                                     'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6',
                                                     'AF8', 'AF4', 'F2', 'FCz'],
                                           sfreq=500,
                                           ch_types='eeg'), verbose=False)
    # data.set_montage('standard_1020')
    data = data.set_eeg_reference(ref_channels='average')
    # data = data.filter(l_freq=0.2, h_freq=None)  # for DC Offset removal
    data = data.filter(l_freq=0.5, h_freq=38)  # for EEG frequency band
    data, _ = dss.dss_line(data.get_data().T, fline=60, sfreq=data.info['sfreq'], nkeep=1)  # for power line noise
    processed_data_array = data[:, -1]  # select the channel of interest (FCz) for 64 chans setting

    # raw_timestamps = np.array([timestamp for sample, timestamp in raw_data])
    raw_timestamps = np.array(raw_data[1])
    # Sample rate (assuming EEG sample rate is 256 Hz)
    sample_rate = 500
    epoch_duration = 2  # 1 second before and 1 second after trigger
    epoch_samples = epoch_duration * sample_rate

    epoched_data = []

    for trigger, trigger_timestamp in triggers:
        # if trigger[:2] == 'KP':
        if trigger in ['R 1', 'R 2', 'R 3', 'R 4']:
            # Find the index of the trigger timestamp in the raw data
            trigger_idx = np.searchsorted(raw_timestamps, trigger_timestamp)

            # Calculate start and end indices for the epoch
            start_idx = trigger_idx - int(sample_rate / 2)  # 0.5 second before
            end_idx = trigger_idx + sample_rate  # 1 second after
            # Ensure indices are within bounds
            if start_idx >= 0 and end_idx < len(processed_data_array):
                epoch = processed_data_array[start_idx:end_idx]
                epoched_data.append(epoch)

    return np.array(epoched_data)[:, np.newaxis]


def process_audio(args, points):

    if not args.spatial_audio:
        audio_name = "./data/audios/train/{}_balanced/note_{}_beep_{}_volume_{}.wav".format(params['timbre'][points[0]],
                                                                                           params['note'][points[1]],
                                                                                             params['pattern'][points[2]],
                                                                                                params['volume'][points[3]])
    os.makedirs("./data/audios/temp", exist_ok=True)
    shutil.copy(f"{audio_name}", "./data/audios/temp")
    audio_dataset = load_dataset("audiofolder", data_dir="./data/audios/temp")
    labels = [label for label in audio_dataset['train'].features.keys() if label not in ['audio', 'label']]

    model_id = "ntu-spml/distilhubert"

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id, do_normalize=True, return_attention_mask=True
    )

    def preprocess_function(examples):
        max_duration = 1.0

        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
            return_attention_mask=True,
        )
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        labels_matrix = np.zeros((len(audio_arrays), len(labels)))
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]
        inputs["labels"] = labels_matrix.tolist()

        return inputs

    audio_encoded = audio_dataset.map(
        preprocess_function,
        remove_columns=audio_dataset['train'].column_names,
        batched=True,
        batch_size=1,
        num_proc=1,
    )
    audio_encoded.set_format("torch")

    shutil.rmtree("./data/audios/temp")
    return audio_encoded, audio_name


def format_points(points):
    # Return the formatted string
    return "Timb_{}_Note_{}_Vol_{}_Patt_{}_Dur_{}_Int_{}".format(
        mapping_timbre.get(points[0], 'Unknown Timbre'),
        mapping_note.get(points[1], 'Unknown Note'),
        mapping_volume.get(points[2], 'Unknown Volume'),
        mapping_pattern.get(points[3], 'Unknown Pattern'),
        mapping_duration.get(points[4], 'Unknown Duration'),
        mapping_intensity.get(points[5], 'Unknown Intensity')
    )


def process_touch(points):
    # return torch.tensor((params['duration'][points[4]], params['intensity'][points[5]]), dtype=torch.float32).unsqueeze(0)
    return params['duration'][points[4]], [params['intensity'][points[5]], params['intensity'][points[5]]], 'both'


def fetch_epoch_data(path, event_id=None):

    raw = mne.io.read_epochs_eeglab(path)
    if event_id is not None:
        raw = raw[event_id]

    data = raw.get_data()

    FCz = data[:, -1, :][:, np.newaxis] * 1e6

    return FCz


def fetch_raw_data(path, event_name=None, combination_ID=None):

    # trial_size = 35
    index = combination_ID * 35 # V 0 VLAL 1 VLALHL 2 xxxxx
    raw = mne.io.read_raw_eeglab(path)

    desc = raw.annotations.description

    # find the element contains 'TS'
    idx = [i for i, s in enumerate(desc) if 'TS' in s]
    # idx_begin, idx_end = idx[index], idx[index+35]  # 35 x 14
    idx_begin, idx_end = idx[0], idx[490]

    raw.annotations.ch_names = raw.annotations.ch_names[idx_begin:idx_end]
    raw.annotations.description = raw.annotations.description[idx_begin:idx_end]
    raw.annotations.onset = raw.annotations.onset[idx_begin:idx_end]
    raw.annotations.duration = raw.annotations.duration[idx_begin:idx_end]

    description = raw.annotations.description
    onset = raw.annotations.onset

    normal_trials = 0
    errp_trials = 0
    idx_gt = [i for i, s in enumerate(description) if 'GT' in s]
    idx_te = [i for i, s in enumerate(description) if 'TE' in s]
    idx_mp = [i for i, s in enumerate(description) if 'DP' in s]
    for i in range(len(idx_gt)):
        s = description[idx_gt[i]].split(' ')[1]
        if int(s) == 0:
            normal_trials += 1
            for j in range(idx_gt[i] + 1, idx_te[i]):
                if description[j] == 'BP':
                    # description[j] = 'n/MP' + tp   # normal  fp
                    description[j] = 'FP'
                elif not description[j].startswith('V'):
                    # description[idx_gt[i]+1: idx_te[i]] = 'n' + tp  # normal   tn
                    description[j] = 'TN/N'
        else:
            errp_trials += 1
            # find 1 in the string s  0010  0100
            idx_1 = [j for j, s in enumerate(s) if s == '1'].pop()
            errp_onset = onset[idx_gt[i] + idx_1 + 1]
            for j in range(idx_gt[i] + 1, idx_te[i]):
                flag = False
                if description[j] == 'BP':  # tp
                    flag = True
                    description[j] = 'TP'

                    # if j - idx_gt[i] != idx_1 + 2:  # not a asap judgement, but followed by later decision point
                    #     description[j] = 'e/MPN' + tp  # non-effective
                    # else:
                    #     if onset[j] - onset[idx_gt[i] + idx_1 + 1] > 1.5:  # too late MP
                    #         description[j] = 'e/MPN' + tp  # non-effective
                    #     else:
                    #         description[j] = 'e/MPP' + tp  # effective
                elif description[j] == 'DP ' + str(idx_1):
                    # description[j] = 'e' + tp
                    description[j] = 'GT'  # to remove
                    temp = str(j)
                elif not description[j].startswith('V'):
                    # description[j] = 'n' + tp  # fn
                    description[j] = 'TN/E'

            if not flag:
                description[int(temp)] = 'FN'

    annotations = raw.annotations

    new_descriptions = []
    new_durations = []
    new_onsets = []

    # Iterate through the annotations and filter out the ones that start with 'TE', 'TS', or 'GT'
    for desc, dur, onset in zip(annotations.description, annotations.duration, annotations.onset):
        if not (desc.startswith('TE') or desc.startswith('TS') or desc.startswith('GT') or desc.startswith('V')):
            new_descriptions.append(desc)
            new_durations.append(dur)
            new_onsets.append(onset)

    # Create new annotations with the filtered lists
    new_annotations = mne.Annotations(onset=new_onsets, duration=new_durations, description=new_descriptions)
    raw.set_annotations(new_annotations)
    events = mne.events_from_annotations(raw)

    # find the index of the event_name in the events
    event_id = None
    for key, value in events[1].items():
        if key == event_name:
            event_id = value
            break

    try:
        epochs = mne.Epochs(raw, events=events[0], event_id=event_id, tmin=-0.5, tmax=1, baseline=(None, None), preload=True)
        data = epochs.get_data()
        FCz = data[:, -1, :][:, np.newaxis] * 1e6

        print(f"Normal trials: {normal_trials}, ErrP trials: {errp_trials}")
        return FCz
    except:
        return None


def play_wave_file(file_path):
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()
