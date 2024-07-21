import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from torch import nn
from torch.optim.lr_scheduler import StepLR
from model.utils import get_parser

def get_model(cfg):
    ## old
    if cfg.arch == 'vocal_stage1':
        from model.audio_vqvae import VQAutoEncoder as Model
        model = Model(args=cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model


# class AudioDataset(Dataset):
#     def __init__(self, audio_dir, segment_ms, processor):
#         self.audio_dir = audio_dir
#         self.segment_ms = segment_ms
#         self.processor = processor
#         self.sample_rate = 16000  # Wav2Vec2 expects 16000 Hz
#         self.segment_len = int(self.sample_rate * (self.segment_ms / 1000))  # Segment length in samples
#         self.file_list = []
#         self.load_and_segment_files()

#     def load_and_segment_files(self):
#         audio_files = [os.path.join(self.audio_dir, f) for f in os.listdir(self.audio_dir) if f.endswith('.wav')]

#         for audio_path in audio_files:
#             waveform, sample_rate = torchaudio.load(audio_path)
            
#             # Resample to Wav2Vec2's expected sample rate
#             if sample_rate != self.sample_rate:
#                 waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(waveform)

#             if waveform.size(0) > 1:
#                 waveform = waveform.mean(dim=0, keepdim=True)
            
            
#             # Segment the waveform into chunks of segment_len
#             num_frames = waveform.size(1)
#             for start_idx in range(0, num_frames, self.segment_len):
#                 if start_idx + self.segment_len <= num_frames:
#                     segment = waveform[:, start_idx:start_idx + self.segment_len]
#                     self.file_list.append(segment)

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         segment = self.file_list[idx]
#         # inputs = self.processor(segment.squeeze(0), sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
#         return inputs.input_values.squeeze(0)

class AudioDataset(Dataset):
    def __init__(self, audio_dir, segment_ms):
        self.audio_dir = audio_dir
        self.segment_ms = segment_ms
        self.sample_rate = 16000  # Assuming a sample rate of 44100 Hz
        self.segment_len = int(self.sample_rate * (self.segment_ms / 1000))  # Segment length in samples
        self.file_list = []
        self.load_and_segment_files()

    def load_and_segment_files(self):
        audio_files = [os.path.join(self.audio_dir, f) for f in os.listdir(self.audio_dir) if f.endswith('.wav')]

        for audio_path in audio_files:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Verify the sample rate matches the expected rate
            
            if sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(waveform)
                # raise ValueError(f"Sample rate mismatch: expected {self.sample_rate}, got {sample_rate}")

            # if waveform.size(0) > 1:
            #     waveform = waveform.mean(dim=0, keepdim=True)

            # Segment the waveform into chunks of segment_len
            num_frames = waveform.size(1)
            for start_idx in range(0, num_frames, self.segment_len):
                if start_idx + self.segment_len <= num_frames:
                    segment = waveform[:, start_idx:start_idx + self.segment_len]
                    self.file_list.append(segment)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return self.file_list[idx]#.unsqueeze(0)

def collate_fn(batch):
    return torch.stack(batch)

# Parameters
audio_directory = '../us_raw_audio'
segment_duration_ms = 25  # Desired sequence length

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Create dataset and dataloader
audio_dataset = AudioDataset(audio_directory,  segment_duration_ms)#, processor)
audio_dataloader = DataLoader(audio_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

args = get_parser()
model = get_model(args)

# Iterate over the dataloader
for batch in audio_dataloader:
    # print('batch', batch.shape)
    output, emb_loss, info = model.encode(batch)
    print('outputs', output.shape)
    # with torch.no_grad():
    #     outputs = wav_model(batch)
    #     print(outputs.last_hidden_state.shape)
    # Perform operations on the batch