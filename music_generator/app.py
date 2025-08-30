# =============================================================================
# Streamlit Web Application
# =============================================================================
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
import streamlit as st
from music_generator.tokenization import REMITokenizer
from music_generator.model import CrossModalMusicTransformer
from music_generator.features import MultiModalFeatureExtractor
from music_generator.trainer import MusicGeneratorTrainer
from music_generator.utils import get_mood_description, analyze_generated_music

MODEL_PATH = "music_generator_model.pt"
def create_streamlit_app():
    """Create Streamlit web application"""
    
    st.set_page_config(
        page_title="Cross-Modal Music Generator",
        page_icon="🎵",
        layout="wide"
    )
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'feature_extractor' not in st.session_state:
        st.session_state.feature_extractor = None
    
    # Header
    st.title("🎵 Cross-Modal Music Generator")
    st.markdown("Generate MIDI melodies from images, lyrics, and emotions using AI")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Model loading
        if st.button("Initialize Model"):
            with st.spinner("Loading models..."):
                st.session_state.tokenizer = REMITokenizer()
                st.session_state.feature_extractor = MultiModalFeatureExtractor()
                st.session_state.model = CrossModalMusicTransformer(
                    vocab_size=st.session_state.tokenizer.vocab_size
                )
            st.success("Models loaded successfully!")
        
        # Training section
        st.header("🏋️ Training")
        midi_dataset_path = st.text_input("MIDI Dataset Path", value="./lmd_full")
        
        if st.button("Start Training") and st.session_state.model is not None:
            if os.path.exists(midi_dataset_path):
                with st.spinner("Training model..."):
                    # Create dataset
                    dataset = MIDIDataset(midi_dataset_path, st.session_state.tokenizer)
                    # Windows + pickling: use num_workers=0; also use small batch size for low VRAM
                    batch_size = 1   # conservative for small GPU; increase later if stable
                    num_workers = 0 if os.name == "nt" else 2

                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

                    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
                    
                    # Create trainer
                    trainer = MusicGeneratorTrainer(
                        st.session_state.model, 
                        st.session_state.tokenizer,
                        st.session_state.feature_extractor
                    )
                    
                    # Train for a few epochs
                    for epoch in range(2):
                        losses = trainer.train_epoch(dataloader)
                        avg_loss = np.mean([l['total_loss'] for l in losses])
                        st.write(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
                    
                    # Save model
                    trainer.save_model("music_generator_model.pt")
                    
                st.success("Training completed!")
            else:
                st.error("MIDI dataset path not found!")
        if st.button("Load Pre-trained Model"):
            if os.path.exists("music_generator_model.pt"):
                trainer = MusicGeneratorTrainer(st.session_state.model, st.session_state.tokenizer, st.session_state.feature_extractor)
                trainer.load_model("music_generator_model.pt")
                st.session_state.trainer = trainer  # Store trained model in session
                st.success("Pre-trained model loaded!")
            else:
                st.error("No trained model found. Please train first.")
    
    # Main application
    if st.session_state.model is not None:
        
        # Input sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📸 Image Input")
            uploaded_image = st.file_uploader(
                "Upload an image for art style reference",
                type=['png', 'jpg', 'jpeg']
            )
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.header("📝 Text Input")
            lyrics_text = st.text_area(
                "Enter lyrics, poem, or descriptive text",
                height=200,
                placeholder="Enter your lyrics or descriptive text here..."
            )
        
        # Emotion controls
        st.header("🎭 Emotion Controls")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            valence = st.slider("Valence (Positive ↔ Negative)", -1.0, 1.0, 0.0, 0.1)
        with col4:
            arousal = st.slider("Arousal (Calm ↔ Energetic)", 0.0, 1.0, 0.5, 0.1)
        with col5:
            style_influence = st.slider("Style Influence", 0.0, 1.0, 0.5, 0.1)
        
        # Preset emotions
        st.subheader("🎨 Emotion Presets")
        preset_cols = st.columns(6)
        
        presets = {
            "Peaceful": (0.8, 0.3),
            "Joyful": (0.9, 0.8),
            "Melancholic": (-0.5, 0.2),
            "Intense": (-0.3, 0.9),
            "Serene": (0.1, 0.1),
            "Uplifting": (0.6, 0.6)
        }
        
        for i, (preset_name, (preset_val, preset_ar)) in enumerate(presets.items()):
            if preset_cols[i].button(preset_name):
                valence = preset_val
                arousal = preset_ar
                st.rerun()
        
        # Generation section
        st.header("🎼 Music Generation")
        
        generation_cols = st.columns(3)
        with generation_cols[0]:
            max_length = st.slider("Max Length (bars)", 4, 32, 8)
        with generation_cols[1]:
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        with generation_cols[2]:
            top_k = st.slider("Top-K Sampling", 1, 100, 50)
        
        # Generate button
        if st.button("🎼 Generate Music", type="primary") and st.session_state.trainer is not None:
            if not lyrics_text and uploaded_image is None:
                st.error("Please provide either text or an image for generation!")
            else:
                with st.spinner("Generating music..."):
                    
                    # Extract features
                    text_features = None
                    image_features = None
                    detected_valence = valence
                    detected_arousal = arousal
                    
                    if lyrics_text:
                        text_embedding, text_valence, text_arousal = st.session_state.feature_extractor.extract_text_features(lyrics_text)
                        text_features = torch.tensor(text_embedding, dtype=torch.float).unsqueeze(0)
                        
                        # Blend detected emotions with manual settings
                        detected_valence = 0.7 * text_valence + 0.3 * valence
                        detected_arousal = 0.7 * text_arousal + 0.3 * arousal
                    
                    if uploaded_image is not None:
                        image_embedding = st.session_state.feature_extractor.extract_image_features(image)
                        image_features = torch.tensor(image_embedding, dtype=torch.float).unsqueeze(0)
                    
                    # Detect device from model
                    device = next(st.session_state.model.parameters()).device
                    # Prepare emotion features
                    # emotion_features = torch.tensor([[detected_valence, detected_arousal]], dtype=torch.float)
                    # Move tensors to same device
                    emotion_features = torch.tensor([[detected_valence, detected_arousal]], dtype=torch.float, device=device)
                    if image_features is not None:
                        image_features = image_features.to(device)
                    if text_features is not None:
                        text_features = text_features.to(device)
                    # Generate music
                    generated_tokens = st.session_state.model.generate(
                        emotion_features=emotion_features,
                        image_features=image_features,
                        text_features=text_features,
                        max_length=max_length * 16,  # Approximate tokens per bar
                        temperature=temperature,
                        top_k=top_k
                    )
                    
                    # Convert to MIDI
                    midi_obj = st.session_state.tokenizer.tokens_to_midi(generated_tokens)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp_file:
                        midi_obj.write(tmp_file.name)
                        midi_path = tmp_file.name
                
                st.success("Music generated successfully!")
                
                # Display results
                st.header("📊 Analysis Results")
                
                result_cols = st.columns(4)
                
                with result_cols[0]:
                    st.metric("Detected Valence", f"{detected_valence:.2f}")
                    
                with result_cols[1]:
                    st.metric("Detected Arousal", f"{detected_arousal:.2f}")
                    
                with result_cols[2]:
                    mood = get_mood_description(detected_valence, detected_arousal)
                    st.metric("Predicted Mood", mood)
                    
                with result_cols[3]:
                    st.metric("Generated Length", f"{len(generated_tokens)} tokens")
                
                # Feature visualization
                if text_features is not None or image_features is not None:
                    st.header("🔍 Feature Analysis")
                    
                    # Create feature visualization
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Emotion space plot
                    axes[0].scatter(detected_valence, detected_arousal, s=200, c='red', alpha=0.7)
                    axes[0].set_xlim(-1, 1)
                    axes[0].set_ylim(0, 1)
                    axes[0].set_xlabel('Valence (Negative ← → Positive)')
                    axes[0].set_ylabel('Arousal (Calm ← → Energetic)')
                    axes[0].set_title('Emotion Space')
                    axes[0].grid(True, alpha=0.3)
                    
                    # Add quadrant labels
                    axes[0].text(0.5, 0.8, 'High Valence\nHigh Arousal\n(Joyful)', ha='center', va='center', alpha=0.6)
                    axes[0].text(-0.5, 0.8, 'Low Valence\nHigh Arousal\n(Intense)', ha='center', va='center', alpha=0.6)
                    axes[0].text(0.5, 0.2, 'High Valence\nLow Arousal\n(Peaceful)', ha='center', va='center', alpha=0.6)
                    axes[0].text(-0.5, 0.2, 'Low Valence\nLow Arousal\n(Sad)', ha='center', va='center', alpha=0.6)
                    
                    # Feature similarity plot (if both text and image available)
                    if text_features is not None and image_features is not None:
                        # Simple similarity visualization
                        text_norm = text_features[0].numpy()[:50]  # First 50 dimensions
                        image_norm = image_features[0].numpy()[:50]
                        
                        x_pos = np.arange(len(text_norm))
                        axes[1].plot(x_pos, text_norm, label='Text Features', alpha=0.7)
                        axes[1].plot(x_pos, image_norm, label='Image Features', alpha=0.7)
                        axes[1].set_title('Cross-Modal Feature Comparison')
                        axes[1].set_xlabel('Feature Dimension')
                        axes[1].set_ylabel('Feature Value')
                        axes[1].legend()
                    else:
                        axes[1].text(0.5, 0.5, 'Upload both text and image\nto see cross-modal analysis', 
                                   ha='center', va='center', transform=axes[1].transAxes)
                        axes[1].set_title('Cross-Modal Analysis')
                    
                    st.pyplot(fig)
                
                # Music analysis
                st.header("🎵 Generated Music Analysis")
                
                # Analyze generated MIDI
                try:
                    music_analysis = analyze_generated_music(midi_obj, generated_tokens, st.session_state.tokenizer)
                    
                    analysis_cols = st.columns(3)
                    
                    with analysis_cols[0]:
                        st.subheader("📈 Musical Statistics")
                        st.write(f"**Number of notes:** {music_analysis['num_notes']}")
                        st.write(f"**Duration:** {music_analysis['duration']:.1f} seconds")
                        st.write(f"**Pitch range:** {music_analysis['pitch_range']} semitones")
                        st.write(f"**Average velocity:** {music_analysis['avg_velocity']:.1f}")
                    
                    with analysis_cols[1]:
                        st.subheader("🎼 Musical Features")
                        st.write(f"**Key signature:** {music_analysis['key_signature']}")
                        st.write(f"**Tempo:** {music_analysis['tempo']:.1f} BPM")
                        st.write(f"**Time signature:** {music_analysis['time_signature']}")
                        st.write(f"**Rhythmic complexity:** {music_analysis['rhythmic_complexity']:.2f}")
                    
                    with analysis_cols[2]:
                        st.subheader("🎭 Emotional Alignment")
                        st.write(f"**Target valence:** {detected_valence:.2f}")
                        st.write(f"**Predicted valence:** {music_analysis['predicted_valence']:.2f}")
                        st.write(f"**Target arousal:** {detected_arousal:.2f}")
                        st.write(f"**Predicted arousal:** {music_analysis['predicted_arousal']:.2f}")
                        
                        alignment_score = 1.0 - 0.5 * (
                            abs(detected_valence - music_analysis['predicted_valence']) +
                            abs(detected_arousal - music_analysis['predicted_arousal'])
                        )
                        st.write(f"**Alignment score:** {alignment_score:.2f}")
                    
                    # Pitch distribution plot
                    if music_analysis['pitch_histogram']:
                        st.subheader("🎹 Pitch Distribution")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        
                        pitches = list(music_analysis['pitch_histogram'].keys())
                        counts = list(music_analysis['pitch_histogram'].values())
                        
                        bars = ax.bar(pitches, counts, alpha=0.7, color='skyblue')
                        ax.set_xlabel('MIDI Pitch')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Note Distribution in Generated Music')
                        
                        # Add note names to x-axis
                        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                        if pitches:
                            tick_labels = [f"{note_names[p % 12]}{p // 12}" for p in pitches[::12]]
                            tick_positions = pitches[::12]
                            ax.set_xticks(tick_positions)
                            ax.set_xticklabels(tick_labels)
                        
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error analyzing music: {e}")
                
                # Download section
                st.header("💾 Download")
                
                # Provide download button for MIDI
                with open(midi_path, 'rb') as f:
                    midi_bytes = f.read()
                
                st.download_button(
                    label="Download MIDI File",
                    data=midi_bytes,
                    file_name="generated_music.mid",
                    mime="audio/midi"
                )
                
                # Audio player (if possible to convert to audio)
                st.info("💡 **Tip:** Download the MIDI file and open it in your favorite DAW or MIDI player for playback!")
                
                # Clean up temporary file
                os.unlink(midi_path)
    
    else:
        st.info("👆 Please initialize the model in the sidebar to start generating music!")

def main():
    create_streamlit_app()

if __name__ == "__main__":
    main()