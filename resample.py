from pydub import AudioSegment

def resample_audio(input_path, output_path, target_sample_rate=16000):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(target_sample_rate)
    audio.export(output_path, format="wav")

# Example usage
input_path = "/home/mani/harp/gettysburg.wav"
output_path = "output_resampled.wav"
resample_audio(input_path, output_path, 16000)