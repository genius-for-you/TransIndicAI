-----

# AI Language Assistant

An offline, real-time desktop application designed for advanced speech analysis. It provides live transcription, translation, simplification, and deep sentiment/emotion analysis, all powered by local AI models.

The application features a modern, responsive user interface built with `ttkbootstrap`, a splash screen for a professional loading experience, and a sophisticated streaming pipeline using Voice Activity Detection (VAD) for true real-time performance.

-----

## ✨ Features

  * **🎙️ Real-Time Transcription:** Live speech-to-text using a high-accuracy Whisper model.
  * **🌍 Multilingual Translation:** Translates English speech into 22 official Indian languages using Meta AI's NLLB model.
  * **✏️ Text Simplification:** Simplifies complex English sentences into easy-to-understand text using a specialized Pegasus model.
  * **😊 Text-based Sentiment Analysis:** Automatically analyzes the transcribed text to determine if the sentiment is POSITIVE or NEGATIVE.
  * **😠 Audio-based Emotion Recognition:** Analyzes the *tone* of the speaker's voice to detect emotions like anger, sadness, happiness, or neutrality.
  * **🖥️ Modern & Responsive UI:** A beautiful, themeable interface that scales gracefully with window size, including dynamic font adjustments.
  * **✈️ Fully Offline:** After an initial setup, the entire application runs without an internet connection, ensuring privacy and accessibility.

-----

## 🛠️ Tech Stack

This project is built entirely in Python and leverages a suite of powerful open-source libraries.

  * **Core Application:**

      * **Python 3.9+**
      * **Tkinter** & **ttkbootstrap**: For the modern, themeable graphical user interface.
      * **Threading**: For concurrent processing to keep the UI responsive while AI models are running.

  * **AI & Machine Learning:**

      * **PyTorch**: The core machine learning framework.
      * **Transformers (by Hugging Face)**: Runs the translation, simplification, sentiment, and emotion models.
      * **Faster-Whisper**: A high-performance implementation of OpenAI's Whisper for transcription.

  * **Audio Processing:**

      * **SoundDevice**: Captures live audio from the microphone.
      * **webrtcvad-wheels**: A high-performance Voice Activity Detection (VAD) library.
      * **pydub**: For robust audio format conversion.
      * **FFmpeg**: An essential system dependency for advanced audio processing.
      * **librosa**: For audio feature extraction required by the emotion model.

-----

## 🚀 Installation & Setup Guide

Follow these four steps to get the application running on your local machine.

### Step 1: Install FFmpeg Prerequisite

This is a crucial, one-time setup for an audio processing tool.

  * **Windows (Recommended):**

    1.  Open PowerShell **as an Administrator**.
    2.  Install the package manager **Chocolatey** from [their official website](https://chocolatey.org/install).
    3.  Run: `choco install ffmpeg`

  * **macOS (using Homebrew):**

    ```bash
    brew install ffmpeg
    ```

  * **Linux (Debian/Ubuntu):**

    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

### Step 2: Set Up the Python Environment

1.  **Download Project Files:** Place `app.py`, `cache_models.py`, and `requirements.txt` into a new folder.

2.  **Create and Activate a Virtual Environment:** Open a terminal in your project folder and run:

    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

    Your terminal prompt should now start with `(venv)`.

3.  **Install All Python Libraries:** Run this single command:

    ```bash
    pip install -r requirements.txt
    ```

### Step 3: Pre-Download AI Models (One-Time)

This step downloads all the necessary AI models (several gigabytes) to your computer. This will take a long time but prevents slow downloads when you start the app.

In your terminal (with the virtual environment active), run the caching script:

```bash
python cache_models.py
```

### Step 4: Run the Application

Once the models are cached, you are ready to launch the app.

1.  Run the main application script from your terminal:
    ```bash
    python app.py
    ```
2.  A "Loading..." splash screen will appear instantly.
3.  After a short wait, the main application window will open, fully functional.

-----

## 📖 How to Use

1.  Click the **"🎤 Start"** button. The application will begin listening. The button will pulsate with a red glow to indicate it's active.
2.  Speak clearly into your microphone. The app uses VAD to detect when you start and stop speaking.
3.  When you pause, the transcribed text, sentiment, and detected emotion will appear automatically. The translation will follow shortly after.
4.  To simplify the last sentence you spoke, click the **"✨ Simplify"** button.
5.  Click the **"🛑 Stop"** button to end the session.