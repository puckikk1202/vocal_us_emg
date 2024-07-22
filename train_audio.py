import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from torch import nn
from torch.optim.lr_scheduler import StepLR
from model.utils import get_parser
from model.loss import calc_vq_loss
import wandb

wandb.init(project='vocal_us_emg')
def get_model(cfg):
    ## old
    if cfg.arch == 'vocal_stage1':
        from model.audio_vqvae import VQAutoEncoder as Model
        model = Model(args=cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model

class AudioDataset(Dataset):
    def __init__(self, audio_dir, segment_ms, is_test=False):
        self.audio_dir = audio_dir
        self.segment_ms = segment_ms
        self.sample_rate = 16000  # Assuming a sample rate of 44100 Hz
        self.segment_len = int(self.sample_rate * (self.segment_ms / 1000))  # Segment length in samples
        self.file_list = []
        self.is_test = is_test
        self.load_and_segment_files()

    def load_and_segment_files(self):
        if self.is_test:
            audio_files = [os.path.join(self.audio_dir, f) for f in os.listdir(self.audio_dir) if f.endswith('05.wav')]
        if not self.is_test:
            audio_files = [os.path.join(self.audio_dir, f) for f in os.listdir(self.audio_dir) if f.endswith('.wav') and not f.endswith('05.wav')]

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
segment_duration_ms = 250  # Desired sequence length

# Create dataset and dataloader
train_dataset = AudioDataset(audio_directory,  segment_duration_ms)
test_dataset = AudioDataset(audio_directory, segment_duration_ms, is_test=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

args = get_parser()
model = get_model(args).to("cuda:1")
optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

wandb.watch(model)

# Iterate over the dataloader
best_loss = float('inf')
for epoch in range(args.epochs):
    print(f'Epoch {epoch}')
    for data in train_loader:
        # print('batch', batch.shape)
        model.train()

        data = data.to("cuda:1")

        out, quant_loss, info = model(data)
        # print('emb_loss', quant_loss)

        loss, loss_details = calc_vq_loss(out, data, quant_loss, quant_loss_weight=args.quant_loss_weight)
        wandb.log({'train_rec': loss_details[0]})
        wandb.log({'train_emb': loss_details[1]})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_loss = 0
    for data in test_loader:
        model.eval()
        data = data.to("cuda:1")
        out, quant_loss, info = model(data)
        loss, loss_details = calc_vq_loss(out, data, quant_loss, quant_loss_weight=args.quant_loss_weight)
        test_loss += loss.item()

        wandb.log({'test_rec': loss_details[0]})
        wandb.log({'test_emb': loss_details[1]})

        
    scheduler.step()
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), './audio_vqvae_models/best_model.pth')

# Save the model
torch.save(model.state_dict(), './audio_vqvae_models/audio_vqvae.pth')
wandb.finish()

