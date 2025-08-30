# Helper functions
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
def get_mood_description(valence, arousal):
    """Get mood description from valence and arousal"""
    if valence > 0.3 and arousal > 0.6:
        return "Energetic & Joyful"
    elif valence > 0.3 and arousal <= 0.6:
        return "Peaceful & Content"
    elif valence <= 0.3 and arousal > 0.6:
        return "Intense & Dramatic"
    else:
        return "Calm & Melancholic"

def analyze_generated_music(midi_obj, tokens, tokenizer):
    """Analyze generated MIDI music"""
    
    analysis = {}
    
    # Basic statistics
    if midi_obj.instruments:
        notes = midi_obj.instruments[0].notes
        analysis['num_notes'] = len(notes)
        analysis['duration'] = midi_obj.get_end_time()
        
        if notes:
            pitches = [note.pitch for note in notes]
            velocities = [note.velocity for note in notes]
            
            analysis['pitch_range'] = max(pitches) - min(pitches)
            analysis['avg_velocity'] = np.mean(velocities)
            
            # Pitch histogram
            pitch_hist = {}
            for pitch in pitches:
                pitch_hist[pitch] = pitch_hist.get(pitch, 0) + 1
            analysis['pitch_histogram'] = pitch_hist
            
            # Estimate key and tempo
            analysis['key_signature'] = estimate_key_signature(pitches)
            analysis['tempo'] = estimate_tempo(midi_obj)
            analysis['time_signature'] = "4/4"  # Simplified
            
            # Rhythmic complexity (simplified)
            analysis['rhythmic_complexity'] = estimate_rhythmic_complexity(notes)
            
            # Predict emotion from musical features (simplified)
            analysis['predicted_valence'], analysis['predicted_arousal'] = predict_emotion_from_music(pitches, velocities, notes)
        else:
            analysis.update({
                'pitch_range': 0, 'avg_velocity': 0, 'pitch_histogram': {},
                'key_signature': 'C', 'tempo': 120, 'time_signature': '4/4',
                'rhythmic_complexity': 0, 'predicted_valence': 0, 'predicted_arousal': 0.5
            })
    else:
        analysis.update({
            'num_notes': 0, 'duration': 0, 'pitch_range': 0, 'avg_velocity': 0,
            'pitch_histogram': {}, 'key_signature': 'C', 'tempo': 120,
            'time_signature': '4/4', 'rhythmic_complexity': 0,
            'predicted_valence': 0, 'predicted_arousal': 0.5
        })
    
    return analysis

def estimate_key_signature(pitches):
    """Estimate key signature from pitch distribution"""
    if not pitches:
        return "C"
    
    # Simple key estimation based on most common notes
    pitch_classes = [p % 12 for p in pitches]
    pitch_counts = {}
    for pc in pitch_classes:
        pitch_counts[pc] = pitch_counts.get(pc, 0) + 1
    
    # Major scale templates
    major_scales = {
        'C': [0, 2, 4, 5, 7, 9, 11],
        'D': [2, 4, 6, 7, 9, 11, 1],
        'E': [4, 6, 8, 9, 11, 1, 3],
        'F': [5, 7, 9, 10, 0, 2, 4],
        'G': [7, 9, 11, 0, 2, 4, 6],
        'A': [9, 11, 1, 2, 4, 6, 8],
        'B': [11, 1, 3, 4, 6, 8, 10]
    }
    
    best_key = 'C'
    best_score = 0
    
    for key, scale in major_scales.items():
        score = sum(pitch_counts.get(pc, 0) for pc in scale)
        if score > best_score:
            best_score = score
            best_key = key
    
    return best_key

def estimate_tempo(midi_obj):
    """Estimate tempo from MIDI object"""
    tempo_changes = midi_obj.get_tempo_changes()
    if len(tempo_changes[1]) > 0:
        return tempo_changes[1][0]
    return 120.0  # Default tempo

def estimate_rhythmic_complexity(notes):
    """Estimate rhythmic complexity from note timings"""
    if len(notes) < 2:
        return 0.0
    
    # Calculate inter-onset intervals
    onsets = sorted([note.start for note in notes])
    intervals = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
    
    if not intervals:
        return 0.0
    
    # Complexity based on variation in intervals
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    
    if mean_interval == 0:
        return 0.0
    
    complexity = std_interval / mean_interval
    return min(complexity, 1.0)

def predict_emotion_from_music(pitches, velocities, notes):
    """Predict valence and arousal from musical features"""
    
    if not pitches or not velocities:
        return 0.0, 0.5
    
    # Valence prediction (major/minor tendency, pitch height)
    pitch_classes = [p % 12 for p in pitches]
    
    # Major scale notes vs minor scale notes
    major_notes = [0, 2, 4, 5, 7, 9, 11]  # C major
    minor_notes = [0, 2, 3, 5, 7, 8, 10]  # C minor
    
    major_count = sum(1 for pc in pitch_classes if pc in major_notes)
    minor_count = sum(1 for pc in pitch_classes if pc in minor_notes)
    
    valence = (major_count - minor_count) / len(pitch_classes) if pitch_classes else 0
    
    # Higher pitches tend to be more positive
    avg_pitch = np.mean(pitches)
    pitch_valence = (avg_pitch - 60) / 24  # Normalize around middle C
    
    valence = 0.7 * valence + 0.3 * pitch_valence
    valence = np.clip(valence, -1, 1)
    
    # Arousal prediction (tempo, dynamics, rhythmic activity)
    avg_velocity = np.mean(velocities)
    velocity_arousal = (avg_velocity - 64) / 64  # Normalize around medium velocity
    
    # Rhythmic activity (notes per second)
    if notes:
        duration = notes[-1].end - notes[0].start
        notes_per_second = len(notes) / duration if duration > 0 else 0
        rhythm_arousal = min(notes_per_second / 4, 1)  # Normalize
    else:
        rhythm_arousal = 0
    
    arousal = 0.6 * velocity_arousal + 0.4 * rhythm_arousal
    arousal = np.clip(arousal, 0, 1)
    
    return float(valence), float(arousal)
