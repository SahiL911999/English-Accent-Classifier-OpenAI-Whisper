# 🎤 English Accent Classifier

An AI-powered English accent detection system that analyzes audio from Loom videos or uploaded files to classify accents with confidence scoring. Built with Gradio, OpenAI Whisper, and advanced speech analysis techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-purple.svg)

## 🌟 Features

- **🎬 Multiple Input Methods**: Support for Loom video URLs, audio files (MP3, WAV), and MP4 video files
- **🤖 AI-Powered Transcription**: Utilizes OpenAI Whisper for accurate speech-to-text conversion
- **🎯 Accent Classification**: Detects and classifies 6 major English accent types
- **📊 Confidence Scoring**: Provides detailed confidence percentages (0-100%)
- **🔊 Advanced Audio Analysis**: Extracts acoustic features including MFCCs, pitch range, tempo, and formants
- **📈 Detailed Results**: Shows all accent scores with visual comparison
- **💾 Export Results**: Download classification results as JSON
- **🌐 Web Interface**: User-friendly Gradio interface for easy interaction

## 🎤 Supported Accents

| Accent | Region | Key Features |
|--------|--------|--------------|
| 🇺🇸 **American** | United States | Rhotic 'r' sounds, flat intonation |
| 🇬🇧 **British** | United Kingdom | Non-rhotic, rising intonation |
| 🇦🇺 **Australian** | Australia | Vowel shifts, rising intonation |
| 🇨🇦 **Canadian** | Canada | Canadian raising, rhotic patterns |
| 🇮🇪 **Irish** | Ireland | Musical intonation, rhotic |
| 🇿🇦 **South African** | South Africa | Specific vowel changes, flat intonation |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (required for audio processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SahiL911999/English-Accent-Classifier-OpenAI-Whisper.git
   cd English-Accent-Classifier-OpenAI-Whisper
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (if not already installed)
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`

### Running the Application

**Launch the Gradio interface:**
```bash
python app.py
```

The application will start and provide a local URL (typically `http://127.0.0.1:7860`) that you can open in your browser.

## 📖 Usage

### Using the Web Interface

1. **Select Input Type**:
   - Loom Video URL
   - Upload Audio File
   - Upload MP4 File

2. **Provide Input**:
   - For Loom URLs: Paste the public Loom video link
   - For files: Upload your audio (MP3, WAV) or video (MP4) file

3. **Analyze**: Click submit and wait for the analysis to complete

4. **View Results**:
   - Detected accent with confidence score
   - Full transcript of the audio
   - Detailed scores for all accent types
   - Explanation of the classification
   - Download results as JSON

### Using the Jupyter Notebook

Open [`Classifier.ipynb`](Classifier.ipynb) to explore the step-by-step implementation:

```bash
jupyter notebook Classifier.ipynb
```

The notebook includes:
- Package installation
- Audio extraction module
- Speech analysis engine
- Accent classification algorithm
- Interactive testing functions

## 🔧 How It Works

### 1. Audio Extraction
- Downloads audio from Loom URLs using [`yt-dlp`](https://github.com/yt-dlp/yt-dlp)
- Processes uploaded audio/video files
- Converts to WAV format (16kHz) for optimal speech recognition

### 2. Speech Analysis
- **Transcription**: Uses OpenAI Whisper (base model) for accurate speech-to-text
- **Feature Extraction**:
  - MFCCs (Mel-frequency cepstral coefficients)
  - Spectral centroid
  - Zero-crossing rate
  - Tempo estimation
  - Pitch range analysis
  - Formant frequency estimation

### 3. Accent Classification
- **Text Pattern Analysis**: Examines rhotic features, key words, and text characteristics
- **Audio Pattern Analysis**: Analyzes tempo, pitch variance, and intonation patterns
- **Scoring Algorithm**: Combines text (40%) and audio (60%) features for final classification
- **Confidence Calculation**: Normalizes scores to 0-100% confidence range

## 📁 Project Structure

```
english-accent-classifier/
│
├── app.py                    # Main Gradio application
├── Classifier.ipynb          # Jupyter notebook with detailed implementation
├── requirements.txt.txt      # Python dependencies
└── README.md                # This file
```

## 🛠️ Technical Stack

- **[Gradio](https://gradio.app/)**: Web interface framework
- **[OpenAI Whisper](https://github.com/openai/whisper)**: Speech recognition model
- **[Librosa](https://librosa.org/)**: Audio analysis library
- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)**: Video/audio download tool
- **[Pydub](https://github.com/jiaaro/pydub)**: Audio manipulation library
- **NumPy & Pandas**: Data processing

## 📊 Classification Algorithm

The system uses a multi-factor approach:

1. **Rhotic Analysis** (30%): Presence of 'r' sounds in speech
2. **Vowel Patterns** (20%): Accent-specific vowel shifts
3. **Tempo Analysis** (20%): Speech rate characteristics
4. **Pitch & Intonation** (20%): Pitch variance and patterns
5. **Keyword Detection** (10%): Accent-specific word pronunciations

## 🎯 Accuracy & Limitations

### Strengths
- ✅ Works well with clear audio (>10 seconds)
- ✅ Handles multiple English accent varieties
- ✅ Provides confidence scores for transparency
- ✅ Real-time processing with visual feedback

### Limitations
- ⚠️ Accuracy depends on audio quality and length
- ⚠️ Short samples (<10 seconds) may reduce reliability
- ⚠️ Background noise can affect results
- ⚠️ Regional sub-accents may not be distinguished
- ⚠️ Currently only supports Loom URLs (not YouTube or other platforms)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement
- Add support for more video platforms (YouTube, Vimeo, etc.)
- Expand accent database with more regional variants
- Improve classification accuracy with larger training datasets
- Add real-time microphone input support
- Implement batch processing for multiple files

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Sahil Ranmbail**

- GitHub: [@sahilranmbail](https://github.com/SahiL911999)
- Project Link: [https://github.com/SahiL911999/English-Accent-Classifier-OpenAI-Whisper]

## 🙏 Acknowledgments

- OpenAI for the Whisper speech recognition model
- Gradio team for the excellent UI framework
- Librosa developers for audio analysis tools
- REM Waste for the hiring assessment opportunity

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/SahiL911999/English-Accent-Classifier-OpenAI-Whisper/issues) page
2. Create a new issue with detailed information
3. Contact the author directly

## 🔮 Future Enhancements

- [ ] Support for more video platforms (YouTube, Vimeo)
- [ ] Real-time microphone input
- [ ] Batch processing capabilities
- [ ] Enhanced accent sub-classification
- [ ] Mobile app version
- [ ] API endpoint for integration
- [ ] Multi-language support
- [ ] Improved accuracy with deep learning models

---

**Built with ❤️ for accurate English accent detection**
