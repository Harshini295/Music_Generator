# 🎵 Cross-Modal Music Generator 
An AI-powered system that generates **MIDI music** conditioned on **text, images, and emotions**.  
Built with **Transformers, REMI tokenization, CLIP, and RoBERTa**, this project explores the intersection of **AI, music, and creativity**.  

---

## ✨ Features  
- 📝 **Text-to-Music** → Generate melodies from lyrics, poems, or descriptive text.  
- 📸 **Image-to-Music** → Use images/art styles to inspire musical themes.  
- 🎭 **Emotion Conditioning** → Control **valence** (positive/negative) and **arousal** (calm/energetic).  
- 🔄 **Cross-Modal Fusion** → Combine **text + image + emotion** for richer outputs.  
- 📊 **Music Analysis Dashboard** → Pitch distribution, tempo, rhythmic complexity, mood alignment.  
- 🌐 **Web Interface** → Interactive **Streamlit app** for easy use.  
- 🧠 **Deep Learning Backbone** → Transformer-based model trained on MIDI data with REMI tokenization.  

---
# ⚙️ Installation

**Install dependencies**

pip install -r requirements.txt

# 🚀 Usage
# 🎼 Run Web App
streamlit run music_generator/app.py
(or)
python -m streamlit run music_generator/app.py

Open the local URL shown in your terminal (usually http://localhost:8501).

---
# 🏋️ Train Model (optional, if you want to re-train)
python -m scripts.train

# 🎶 Generate via CLI
python -m scripts.generate
This will create a file: generated_output.mid

---
# 🎨 Example Workflow
1. Inlialize the model by clicking on Initial model 
2. (Optional)to re-train the model:
  -Enter the file path of the dataset
  -Start Training
  -This will re-train the model and saves the model into music_generator_model.pt
  -You can now load this model by clicking on Load pre-trained model
3. Load the pre-trained model
4. Upload an image for style inspiration or Enter lyrics or descriptive text.
5. Adjust valence/arousal sliders (calm ↔ energetic, sad ↔ joyful).
6. Hit Generate Music 🎼.
7. Download the generated MIDI file and play it in your favorite DAW or MIDI player.

---
# 📊 Music Analysis
The app provides analysis of generated music, including:

- Number of notes
- Duration
- Pitch range
- Average velocity
- Key signature & tempo
- Rhythmic complexity
- Predicted emotional alignment

---
# 🛠️ Tech Stack
PyTorch – Model training
Transformers (Hugging Face) – RoBERTa & CLIP embeddings
pretty_midi, mido – MIDI processing
Streamlit – Web interface
Matplotlib, Seaborn – Visualization

---
# Dataset used : [https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean](click here)
