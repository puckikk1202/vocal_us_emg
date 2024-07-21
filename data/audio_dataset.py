import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as transforms

class AudioDataset(Dataset):
    def __init__(self, audio_dir, seq_len):
        self.audio_dir = audio_dir
        self.seq_len = seq_len
        self.resample_rate = 60  # Target frames per second
        self.file_list = []
        self.load_and_segment_files()

    def load_and_segment_files(self):
        audio_files = [os.path.join(self.audio_dir, f) for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
        resampler = transforms.Resample(orig_freq=44100, new_freq=self.resample_rate)  # Assuming original sample rate is 44100 Hz

        for audio_path in audio_files:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample to target fps
            resampled_waveform = resampler(waveform)
            
            # Segment the resampled waveform into chunks of seq_len
            num_frames = resampled_waveform.size(1)
            for start_idx in range(0, num_frames, self.seq_len):
                if start_idx + self.seq_len <= num_frames:
                    segment = resampled_waveform[:, start_idx:start_idx + self.seq_len]
                    self.file_list.append(segment)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return self.file_list[idx]

def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.tensor([])
    return torch.stack(batch)