from pydub import AudioSegment
import os


# Function to extract and save individual clips
def save_clip(audio, start_time, clip_index, audio_name):
    end_time = start_time + clip_length_ms
    clip = audio[start_time:end_time]

    # Naming rule
    note = notes[clip_index]
    beeps = beep_counts[clip_index]
    volume = volumes[clip_index]
    os.makedirs(f"../../sound/{audio_name}", exist_ok=True)

    filename = f"../../sound/{audio_name}/note_{note}_beep_{beeps}_volume_{volume}.wav"
    # Export the audio clip
    clip.export(filename, format="wav")
    print(f"Saved: {filename}")


if __name__ == '__main__':

    audio_list = ['sawtooth_balanced', 'sawtooth_left', 'sawtooth_right', 'sine_balanced', 'sine_left', 'sine_right',
                    'square_balanced', 'square_left', 'square_right']

    for audio_name in audio_list:
        # Load the 46-second WAV file

        audio = AudioSegment.from_wav(f"../../sound/sound_synthesis_{audio_name}.wav")

        # audio = AudioSegment.from_wav("sound_synthesis_sawtooth_balanced.wav")

        # Set up constants
        clip_length_ms = 1000  # each clip is 1 second, so 1000 milliseconds
        total_clips = 45  # 45 clips, excluding the empty last second

        # Define the clip configurations
        volumes = [127] * 15 + [64] * 15 + [1] * 15  # Volume levels
        beep_counts = ([1] * 5 + [2] * 5 + [3] * 5) * 3  # Beep counts in each volume group
        notes = ['C6', 'C5', 'C4', 'C3', 'C2'] * 9  # Notes for each group

        # Iterate over the audio to slice into 1-second clips and save them
        for i in range(total_clips):
            save_clip(audio, i * clip_length_ms, i, audio_name)
