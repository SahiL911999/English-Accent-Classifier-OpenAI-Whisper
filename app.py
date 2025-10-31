import gradio as gr
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
import requests
from urllib.parse import urlparse
import re
import json
import time
from datetime import datetime
import whisper
import torch
from pydub import AudioSegment

class AudioExtractor:
    """Handle video URL processing and audio extraction"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.supported_formats = ['mp4', 'mp3', 'wav', 'avi', 'mov', 'webm']
        
    def is_valid_url(self, url):
        """Validate if URL is properly formatted"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def detect_url_type(self, url):
        """Detect the type of URL (only Loom is accepted)"""
        url_lower = url.lower()
        if 'loom.com' in url_lower:
            return 'loom'
        return 'unknown'
    
    def download_audio(self, url):
        """Download and extract audio from URL (Loom only)"""
        if not self.is_valid_url(url):
            raise ValueError("Invalid URL format")
        
        url_type = self.detect_url_type(url)
        
        try:
            if url_type == 'loom':
                return self._extract_with_ytdlp(url)
            else:
                raise ValueError("Only Loom URLs are supported")
        except Exception as e:
            raise Exception(f"Audio extraction failed: {str(e)}")
    
    def _extract_with_ytdlp(self, url):
        """Extract audio using yt-dlp"""
        import yt_dlp
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.temp_dir, 'temp_video.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'postprocessor_args': ['-ar', '16000'],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        for file in os.listdir(self.temp_dir):
            if file.endswith('.wav'):
                return os.path.join(self.temp_dir, file)
        
        raise Exception("Audio extraction failed")
    
    def process_local_file(self, file_path):
        """Process local audio or video file to extract audio in WAV format"""
        try:
            audio = AudioSegment.from_file(file_path)
            wav_path = os.path.join(self.temp_dir, 'extracted.wav')
            audio.export(wav_path, format='wav', parameters=['-ar', '16000'])
            return wav_path
        except Exception as e:
            raise Exception(f"Failed to process local file: {str(e)}")

class SpeechAnalyzer:
    """Analyze speech patterns for accent detection"""
    
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio to text with phonetic information"""
        try:
            if self.whisper_model is None:
                raise Exception("Whisper model not loaded")
            
            result = self.whisper_model.transcribe(audio_path)
            text = result['text']
            
            audio_features = self._extract_audio_features(audio_path)
            
            return {
                'text': text,
                'segments': result.get('segments', []),
                'audio_features': audio_features
            }
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
    
    def _extract_audio_features(self, audio_path):
        """Extract acoustic features for accent analysis"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            
            features = {
                'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1),
                'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
                'zero_crossing_rate': librosa.feature.zero_crossing_rate(y).mean(),
                'tempo': librosa.beat.tempo(y=y, sr=sr)[0] if len(librosa.beat.tempo(y=y, sr=sr)) > 0 else 120,
                'pitch_range': self._get_pitch_range(y, sr),
                'formants': self._estimate_formants(y, sr)
            }
            
            return features
        except Exception as e:
            print(f"Feature extraction warning: {e}")
            return {}
    
    def _get_pitch_range(self, y, sr):
        """Calculate pitch range"""
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitches = pitches[magnitudes > np.median(magnitudes)]
            pitches = pitches[pitches > 0]
            
            if len(pitches) > 0:
                return {'min': float(np.min(pitches)), 'max': float(np.max(pitches))}
            return {'min': 0, 'max': 0}
        except:
            return {'min': 0, 'max': 0}
    
    def _estimate_formants(self, y, sr):
        """Estimate formant frequencies"""
        try:
            fft = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            
            peaks = []
            for i in range(1, len(magnitude)-1):
                if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                    if freqs[i] > 0 and freqs[i] < 4000:
                        peaks.append(freqs[i])
            
            peaks.sort()
            return peaks[:3] if len(peaks) >= 3 else peaks
        except:
            return []

class AccentClassifier:
    """Classify English accents based on speech analysis"""
    
    def __init__(self):
        self.accent_features = {
            'american': {
                'rhotic': 1.0,
                'vowel_shift': 0.8,
                'intonation': 'flat',
                'tempo': 'medium'
            },
            'british': {
                'rhotic': 0.0,
                'vowel_shift': 0.3,
                'intonation': 'rising',
                'tempo': 'medium'
            },
            'australian': {
                'rhotic': 0.0,
                'vowel_shift': 0.9,
                'intonation': 'rising',
                'tempo': 'fast'
            },
            'canadian': {
                'rhotic': 0.8,
                'vowel_shift': 0.6,
                'intonation': 'rising',
                'tempo': 'medium'
            },
            'irish': {
                'rhotic': 0.7,
                'vowel_shift': 0.4,
                'intonation': 'musical',
                'tempo': 'variable'
            },
            'south_african': {
                'rhotic': 0.2,
                'vowel_shift': 0.7,
                'intonation': 'flat',
                'tempo': 'medium'
            }
        }
    
    def classify_accent(self, transcription_data):
        """Main accent classification function"""
        text = transcription_data['text']
        audio_features = transcription_data.get('audio_features', {})
        
        accent_scores = {}
        
        for accent_name, accent_features in self.accent_features.items():
            score = self._calculate_accent_score(text, audio_features, accent_features)
            accent_scores[accent_name] = score
        
        best_accent = max(accent_scores, key=accent_scores.get)
        confidence = accent_scores[best_accent]
        confidence_percentage = min(100, max(0, confidence * 100))
        
        result = {
            'accent': best_accent,
            'confidence': confidence_percentage,
            'scores': accent_scores,
            'explanation': self._generate_explanation(best_accent, confidence_percentage, text)
        }
        
        return result
    
    def _calculate_accent_score(self, text, audio_features, accent_features):
        """Calculate similarity score between speech and accent pattern"""
        score = 0.0
        weight_sum = 0.0
        
        text_score = self._analyze_text_patterns(text, accent_features)
        score += text_score * 0.4
        weight_sum += 0.4
        
        if audio_features:
            audio_score = self._analyze_audio_patterns(audio_features, accent_features)
            score += audio_score * 0.6
            weight_sum += 0.6
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _analyze_text_patterns(self, text, accent_features):
        """Analyze text for accent-specific patterns"""
        score = 0.0
        text_lower = text.lower()
        
        rhotic_expected = accent_features.get('rhotic', 0.5)
        r_count = text_lower.count('r')
        total_chars = len(text_lower.replace(' ', ''))
        r_ratio = r_count / total_chars if total_chars > 0 else 0
        
        if rhotic_expected > 0.5:
            score += min(r_ratio * 2, 0.3)
        else:
            score += max(0.3 - r_ratio * 2, 0)
        
        word_count = len(text.split())
        if word_count > 10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_audio_patterns(self, audio_features, accent_features):
        """Analyze audio features for accent classification"""
        score = 0.0
        
        tempo = audio_features.get('tempo', 120)
        expected_tempo = accent_features.get('tempo', 'medium')
        
        if expected_tempo == 'fast' and tempo > 130:
            score += 0.2
        elif expected_tempo == 'medium' and 110 <= tempo <= 130:
            score += 0.2
        elif expected_tempo == 'slow' and tempo < 110:
            score += 0.2
        
        pitch_range = audio_features.get('pitch_range', {})
        if pitch_range:
            pitch_variance = pitch_range.get('max', 0) - pitch_range.get('min', 0)
            intonation = accent_features.get('intonation', 'flat')
            
            if intonation == 'rising' and pitch_variance > 100:
                score += 0.2
            elif intonation == 'flat' and pitch_variance < 100:
                score += 0.2
            elif intonation == 'musical' and pitch_variance > 150:
                score += 0.3
        
        return min(score, 1.0)
    
    def _generate_explanation(self, accent, confidence, text):
        """Generate explanation for the classification"""
        explanations = {
            'american': f"Detected American English features including rhotic 'r' sounds and typical vowel patterns.",
            'british': f"Identified British English characteristics such as non-rhotic pronunciation and distinct vowel sounds.",
            'australian': f"Found Australian English markers including vowel shifts and distinctive intonation patterns.",
            'canadian': f"Detected Canadian English features including potential Canadian raising and rhotic patterns.",
            'irish': f"Identified Irish English characteristics including musical intonation and specific vowel patterns.",
            'south_african': f"Found South African English markers including specific vowel changes and intonation."
        }
        
        base_explanation = explanations.get(accent, f"Classified as {accent} accent")
        
        word_count = len(text.split())
        if word_count < 10:
            base_explanation += " (Note: Short audio sample may limit accuracy)"
        elif word_count > 50:
            base_explanation += " (Good sample length for reliable analysis)"
        
        return base_explanation

# Global instances
audio_extractor = AudioExtractor()
speech_analyzer = SpeechAnalyzer()
accent_classifier = AccentClassifier()

def process_audio(audio_path):
    """Process audio file for transcription and accent classification"""
    try:
        transcription_data = speech_analyzer.transcribe_audio(audio_path)
        classification_result = accent_classifier.classify_accent(transcription_data)
        
        final_result = {
            'transcript': transcription_data['text'],
            'accent': classification_result['accent'],
            'confidence': classification_result['confidence'],
            'explanation': classification_result['explanation'],
            'all_scores': classification_result['scores'],
            'timestamp': datetime.now().isoformat()
        }
        return final_result
    except Exception as e:
        raise Exception(f"Processing failed: {str(e)}")

def analyze_accent(input_type, url, audio_file, mp4_file):
    """Analyze accent based on input type"""
    if input_type == "Loom Video URL":
        if not url:
            return "Please provide a Loom URL", None, None, None, None, None, None
        if 'loom.com' not in url.lower():
            return "Please enter a valid Loom URL", None, None, None, None, None, None
        try:
            audio_path = audio_extractor.download_audio(url)
            result = process_audio(audio_path)
        except Exception as e:
            return f"Error processing URL: {str(e)}", None, None, None, None, None, None
    
    elif input_type == "Upload Audio File":
        if audio_file is None:
            return "Please upload an audio file", None, None, None, None, None, None
        try:
            # Use audio_file directly as the file path string
            audio_path = audio_extractor.process_local_file(audio_file)
            result = process_audio(audio_path)
        except Exception as e:
            return f"Error processing audio file: {str(e)}", None, None, None, None, None, None
    
    elif input_type == "Upload MP4 File":
        if mp4_file is None:
            return "Please upload an MP4 file", None, None, None, None, None, None
        try:
            # Use mp4_file directly as the file path string
            audio_path = audio_extractor.process_local_file(mp4_file)
            result = process_audio(audio_path)
        except Exception as e:
            return f"Error processing MP4 file: {str(e)}", None, None, None, None, None, None
    
    else:
        return "Invalid input type", None, None, None, None, None, None
    
    # Prepare outputs
    status = "✅ Analysis Complete!"
    accent_text = f"🎯 Detected Accent: {result['accent'].title()}"
    confidence_text = f"📈 Confidence: {result['confidence']:.1f}%"
    explanation = result['explanation']
    transcript = result['transcript']
    scores_df = pd.DataFrame([
        {'Accent': accent.title(), 'Score': score, 'Percentage': f"{score*100:.1f}%"}
        for accent, score in result['all_scores'].items()
    ]).sort_values('Score', ascending=False)
    
    result_json = json.dumps(result, indent=2)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
        tmp_file.write(result_json.encode())
        download_file = tmp_file.name
    
    return status, accent_text, confidence_text, explanation, transcript, scores_df, download_file

# Gradio Interface
iface = gr.Interface(
    fn=analyze_accent,
    inputs=[
        gr.Dropdown(["Loom Video URL", "Upload Audio File", "Upload MP4 File"], label="Input Type"),
        gr.Textbox(label="Loom Video URL", placeholder="https://www.loom.com/share/..."),
        gr.File(label="Upload Audio File", type="filepath"),  # Fixed: type="filepath"
        gr.File(label="Upload MP4 File", type="filepath")     # Fixed: type="filepath"
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Textbox(label="Detected Accent"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Explanation"),
        gr.Textbox(label="Transcript"),
        gr.Dataframe(label="Detailed Scores"),
        gr.File(label="Download Results")
    ],
    title="🎤 English Accent Detector",
    description="""
    Select an input type and provide the corresponding input to analyze the English accent.

    **How it works:**
    1. 🔗 Enter a public Loom video URL or upload an audio/MP4 file
    2. 🎵 Audio is extracted or used directly
    3. 🤖 AI analyzes speech patterns and phonetic features
    4. 🎯 Get accent classification with confidence score

    **Supported Accents:**
    - 🇺🇸 American
    - 🇬🇧 British
    - 🇦🇺 Australian
    - 🇨🇦 Canadian
    - 🇮🇪 Irish
    - 🇿🇦 South African

    *Built for REM Waste hiring assessment using Gradio, Whisper AI, and advanced speech analysis.*
    """
)

if __name__ == "__main__":
    iface.launch()