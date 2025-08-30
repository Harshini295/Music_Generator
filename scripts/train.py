from torch.utils.data import DataLoader
import numpy as np
from music_generator.tokenization import REMITokenizer
from music_generator.dataset import MIDIDataset
from music_generator.model import CrossModalMusicTransformer
from music_generator.features import MultiModalFeatureExtractor
from music_generator.trainer import MusicGeneratorTrainer
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
from pathlib import Path
import pretty_midi
import librosa
from transformers import RobertaTokenizer, RobertaModel, CLIPProcessor, CLIPModel
import streamlit as st
import tempfile
import io
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # This will give you a better error trace
from typing import List, Dict, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings("ignore")
def main():
    tokenizer = REMITokenizer()
    feature_extractor = MultiModalFeatureExtractor()
    model = CrossModalMusicTransformer(vocab_size=tokenizer.vocab_size)
    dataset = MIDIDataset("./lmd_full", tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    trainer = MusicGeneratorTrainer(model, tokenizer, feature_extractor)
    for epoch in range(2):
        losses = trainer.train_epoch(dataloader)
        avg_loss = np.mean([l["total_loss"] for l in losses])
        print(f"Epoch {epoch+1}: Avg Loss {avg_loss:.4f}")
    trainer.save_model("music_generator_model.pt")

if __name__ == "__main__":
    main()
