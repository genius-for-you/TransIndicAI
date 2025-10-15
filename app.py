#with emotions #final
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
import threading
import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from transformers import pipeline
import torch
import webrtcvad
from collections import deque

# --- Model Configurations ---
TRANSCRIPTION_MODEL_ID = "medium.en"
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"
SIMPLIFICATION_MODEL_NAME = "tuner007/pegasus_paraphrase"
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL_NAME = "superb/wav2vec2-base-superb-er"

# --- Other Configurations ---
DEVICE, COMPUTE_TYPE = "cpu", "int8"
SAMPLE_RATE = 16000
OUTPUT_LANGUAGE_MAPPING = { "English": "eng_Latn", "Assamese": "asm_Beng", "Bengali": "ben_Beng", "Bodo": "brx_Deva", "Dogri": "dgo_Deva", "Gujarati": "guj_Gujr", "Hindi": "hin_Deva", "Kannada": "kan_Knda", "Kashmiri": "kas_Arab", "Konkani": "gom_Deva", "Maithili": "mai_Deva", "Malayalam": "mal_Mlym", "Manipuri (Meitei)": "mni_Beng", "Marathi": "mar_Deva", "Nepali": "npi_Deva", "Odia": "ory_Orya", "Punjabi": "pan_Guru", "Sanskrit": "san_Deva", "Santali": "sat_Olck", "Sindhi": "snd_Arab", "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Urdu": "urd_Arab" }

def load_all_models(loading_queue):
    print("--- Background loading of all models started. ---")
    try:
        models = {}
        loading_queue.put("Loading transcription model...")
        models["transcription"] = WhisperModel(TRANSCRIPTION_MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("✅ Transcription model loaded.")
        loading_queue.put("Loading sentiment model...")
        models["sentiment"] = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)
        print("✅ Sentiment model loaded.")
        loading_queue.put("Loading translation model...")
        models["translation"] = pipeline("translation", model=TRANSLATION_MODEL_NAME)
        print("✅ Translation model loaded.")
        loading_queue.put("Loading simplification model...")
        models["simplification"] = pipeline("text2text-generation", model=SIMPLIFICATION_MODEL_NAME)
        print("✅ Simplification model loaded.")
        loading_queue.put("Loading emotion model...")
        models["emotion"] = pipeline("audio-classification", model=EMOTION_MODEL_NAME)
        print("✅ Emotion model loaded.")
        loading_queue.put(models)
    except Exception as e:
        loading_queue.put(f"Error: {e}")

audio_queue, transcript_queue, text_to_translate_queue = queue.Queue(), queue.Queue(), queue.Queue()
stop_event, last_english_text = threading.Event(), ""

def recording_thread(status_bar):
    print("🎙️ Real-time recording thread started.")
    vad, frame_duration_ms, frame_size = webrtcvad.Vad(3), 30, int(SAMPLE_RATE * 30 / 1000)
    speech_buffer, triggered, speech_frames = bytearray(), False, deque(maxlen=20)
    status_bar.config(text="Listening...")
    def audio_callback(indata, frames, time, status):
        nonlocal triggered, speech_buffer, speech_frames
        if status: print(status, flush=True)
        audio_data = (indata * 32767).astype(np.int16).tobytes()
        try:
            is_speech = vad.is_speech(audio_data, SAMPLE_RATE)
        except Exception:
            is_speech = False
        if not triggered:
            speech_frames.append((audio_data, is_speech))
            if len([f for f, s in speech_frames if s]) > 0.9 * speech_frames.maxlen:
                triggered = True; status_bar.config(text="Speech detected...")
                for f, s in speech_frames: speech_buffer.extend(f)
                speech_frames.clear()
        else:
            speech_buffer.extend(audio_data); speech_frames.append((audio_data, is_speech))
            if len([f for f, s in speech_frames if not s]) > 0.9 * speech_frames.maxlen:
                triggered = False; status_bar.config(text="Processing speech...")
                audio_to_process = np.frombuffer(speech_buffer, dtype=np.int16).astype(np.float32) / 32767.0
                audio_queue.put(audio_to_process)
                speech_buffer, speech_frames = bytearray(), deque(maxlen=20)
                status_bar.config(text="Listening...")
    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=frame_size, channels=1, dtype='float32', callback=audio_callback):
        while not stop_event.is_set(): sd.sleep(100)
    print("🎙️ Recording thread finished.")

def transcription_thread(app_instance):
    global last_english_text
    emotion_emoji_map = {"ang": "😠", "sad": "😢", "hap": "😄", "neu": "😐"}
    while not stop_event.is_set():
        try:
            audio_segment = audio_queue.get(timeout=1)
            segments, _ = app_instance.models["transcription"].transcribe(audio_segment, beam_size=5)
            full_text = "".join(segment.text for segment in segments).strip()
            if full_text:
                last_english_text = full_text
                transcript_queue.put(f"You: {full_text}\n")
                text_to_translate_queue.put(full_text)
                
                sentiment_result = app_instance.models["sentiment"](full_text)[0]
                label, score, emoji = sentiment_result['label'], sentiment_result['score'], "😊" if sentiment_result['label'] == 'POSITIVE' else "😠"
                app_instance.root.after(0, lambda: app_instance.sentiment_label.config(text=f"Sentiment: {label} ({score:.2f}) {emoji}"))

                # --- THIS IS THE FIX: Normalize audio for emotion model ---
                normalized_audio = audio_segment / np.max(np.abs(audio_segment))
                
                emotion_result = app_instance.models["emotion"](normalized_audio, sampling_rate=SAMPLE_RATE)[0]
                emotion_label = emotion_result['label'].capitalize()
                emotion_score = emotion_result['score']
                emotion_emoji = emotion_emoji_map.get(emotion_label[:3].lower(), "")
                app_instance.root.after(0, lambda: app_instance.emotion_label.config(text=f"Emotion: {emotion_label} ({emotion_score:.2f}) {emotion_emoji}"))

        except (queue.Empty, ValueError): # Added ValueError for potential empty audio chunks
            continue

def translation_thread(app_instance):
    while not stop_event.is_set():
        try:
            text_to_translate = text_to_translate_queue.get(timeout=1)
            selected_output_language = app_instance.output_language_var.get()
            target_lang_code = OUTPUT_LANGUAGE_MAPPING[selected_output_language]
            if target_lang_code == "eng_Latn": continue
            translated_text = app_instance.models["translation"](text_to_translate, src_lang="eng_Latn", tgt_lang=target_lang_code)[0]['translation_text']
            transcript_queue.put(f"[{selected_output_language}]: {translated_text}\n\n")
        except queue.Empty: continue

def simplification_thread(app_instance, status_bar):
    global last_english_text
    if not last_english_text:
        transcript_queue.put("[System]: No text to simplify.\n"); app_instance.root.after(0, app_instance.enable_simplify_button)
        return
    try:
        status_bar.config(text="Simplifying text...")
        simplified_text = app_instance.models["simplification"](last_english_text, max_length=128, clean_up_tokenization_spaces=True)[0]['generated_text']
        transcript_queue.put(f"[Simplified]: {simplified_text}\n\n")
        status_bar.config(text="Simplification complete.")
    except Exception as e: print(f"Simplification error: {e}")
    app_instance.root.after(0, app_instance.enable_simplify_button)

class TranslationApp:
    def __init__(self, root, loaded_models):
        self.root, self.models = root, loaded_models
        self.is_pulsing, self.is_pulsing_outline = False, False
        self.build_ui()

    def build_ui(self):
        self.root.title("AI Language Assistant")
        self.root.geometry("900x750")
        self.root.resizable(True, True) 

        self.root.grid_rowconfigure(0, weight=1); self.root.grid_columnconfigure(0, weight=1)
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(2, weight=1); main_frame.grid_columnconfigure(0, weight=1)
        
        actions_frame = ttk.LabelFrame(main_frame, text="Actions", padding="15")
        actions_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        actions_frame.grid_columnconfigure(3, weight=1)

        self.start_button = ttk.Button(actions_frame, text="🎤 Start", command=self.start_session, bootstyle="success", width=12); self.start_button.grid(row=0, column=0, padx=5, pady=5)
        self.stop_button = ttk.Button(actions_frame, text="🛑 Stop", command=self.stop_session, bootstyle="danger", state=DISABLED, width=12); self.stop_button.grid(row=0, column=1, padx=5, pady=5)
        self.simplify_button = ttk.Button(actions_frame, text="✨ Simplify", command=self.simplify_last_text, bootstyle="info", state=DISABLED, width=15); self.simplify_button.grid(row=0, column=2, padx=20, pady=5)
        
        analysis_display_frame = ttk.Frame(actions_frame)
        analysis_display_frame.grid(row=0, column=4, padx=10, pady=5, sticky='e')
        self.sentiment_label = ttk.Label(analysis_display_frame, text="Sentiment: -", font=("Segoe UI", 10, "italic")); self.sentiment_label.pack(anchor='w')
        self.emotion_label = ttk.Label(analysis_display_frame, text="Emotion: -", font=("Segoe UI", 10, "italic")); self.emotion_label.pack(anchor='w')

        translate_frame = ttk.LabelFrame(main_frame, text="Translation", padding="15"); translate_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.output_language_var = tk.StringVar(value="Hindi")
        output_lang_options = sorted(list(OUTPUT_LANGUAGE_MAPPING.keys()))
        ttk.Label(translate_frame, text="Translate English to:").pack(side=LEFT, padx=(0, 10))
        self.output_lang_menu = ttk.OptionMenu(translate_frame, self.output_language_var, "Hindi", *output_lang_options); self.output_lang_menu.pack(side=LEFT, padx=(0, 20))
        
        text_frame = ttk.LabelFrame(main_frame, text="Live Transcript", padding="10"); text_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=10)
        text_frame.grid_rowconfigure(0, weight=1); text_frame.grid_columnconfigure(0, weight=1)
        self.text_area = ScrolledText(text_frame, wrap=tk.WORD, height=20, font=("Segoe UI", 11), bootstyle="dark"); self.text_area.grid(row=0, column=0, sticky="nsew"); self.text_area.insert(tk.END, "Click 'Start' to begin...\n"); self.text_area.text.config(state=DISABLED)
        
        self.status_bar = ttk.Label(self.root, text="Ready", relief=SUNKEN, anchor=W, padding="5"); self.status_bar.grid(row=1, column=0, sticky="ew")
        
        self.threads = []; self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.widgets_to_scale = [self.output_lang_menu, self.start_button, self.stop_button, self.simplify_button, self.sentiment_label, self.emotion_label, self.text_area.text, self.status_bar]
        self._resize_job = None; self.root.bind("<Configure>", self.on_resize)

    def pulsate_record_button(self):
        if not self.is_pulsing: return
        style = "danger-outline" if self.is_pulsing_outline else "danger"
        self.start_button.config(bootstyle=style)
        self.is_pulsing_outline = not self.is_pulsing_outline
        self.root.after(700, self.pulsate_record_button)
    
    def on_resize(self, event):
        if self._resize_job: self.root.after_cancel(self._resize_job);
        self._resize_job = self.root.after(250, self.resize_fonts)
    def resize_fonts(self):
        new_size = 9 + (self.root.winfo_width() - 600) // 200
        if new_size < 9: new_size = 9
        if new_size > 18: new_size = 18
        new_font = ("Segoe UI", new_size)
        for widget in self.widgets_to_scale:
            try: widget.config(font=new_font)
            except tk.TclError: pass
        self._resize_job = None
    
    def start_session(self):
        self.start_button.config(state=DISABLED, text="🎤 Listening...")
        self.stop_button.config(state=NORMAL); self.simplify_button.config(state=NORMAL); self.output_lang_menu.config(state=DISABLED)
        self.text_area.text.config(state=NORMAL); self.text_area.delete('1.0', tk.END); self.text_area.insert(tk.END, "--- Session Started ---\n"); self.text_area.text.config(state=DISABLED)
        self.status_bar.config(text="Starting session...")
        self.is_pulsing = True; self.pulsate_record_button()
        stop_event.clear()
        self.threads = []
        self.threads.append(threading.Thread(target=recording_thread, args=(self.status_bar,), daemon=True))
        self.threads.append(threading.Thread(target=transcription_thread, args=(self,), daemon=True))
        self.threads.append(threading.Thread(target=translation_thread, args=(self,), daemon=True))
        [t.start() for t in self.threads]
        self.check_transcript_queue()

    def simplify_last_text(self):
        self.simplify_button.config(state=DISABLED)
        threading.Thread(target=simplification_thread, args=(self, self.status_bar), daemon=True).start()
    def enable_simplify_button(self):
        if self.stop_button['state'] == tk.NORMAL: self.simplify_button.config(state=tk.NORMAL)
    
    def stop_session(self):
        if not stop_event.is_set(): stop_event.set(); [t.join() for t in self.threads]
        self.is_pulsing = False; self.start_button.config(state=NORMAL, bootstyle="success", text="🎤 Start")
        self.stop_button.config(state=DISABLED); self.simplify_button.config(state=DISABLED); self.output_lang_menu.config(state=NORMAL)
        self.text_area.text.config(state=NORMAL); self.text_area.insert(tk.END, "\n--- Session Stopped ---\n"); self.text_area.text.config(state=DISABLED)
        self.status_bar.config(text="Ready"); self.sentiment_label.config(text="Sentiment: -"); self.emotion_label.config(text="Emotion: -")

    def check_transcript_queue(self):
        try:
            while not transcript_queue.empty():
                message = transcript_queue.get_nowait()
                self.text_area.text.config(state=NORMAL); self.text_area.insert(tk.END, message); self.text_area.see(tk.END); self.text_area.text.config(state=DISABLED)
        except queue.Empty: pass
        if not stop_event.is_set(): self.root.after(100, self.check_transcript_queue)

    def on_closing(self):
        self.stop_session(); self.root.destroy()

if __name__ == "__main__":
    root = ttk.Window(themename="cyborg")
    root.title("Loading...")
    root.geometry("400x200")
    root.resizable(False, False)
    ttk.Label(root, text="AI Language Assistant", font=("Segoe UI", 18, "bold")).pack(pady=20)
    progress_bar = ttk.Progressbar(root, mode='indeterminate', length=300); progress_bar.pack(pady=10); progress_bar.start()
    status_label = ttk.Label(root, text="Initializing...", font=("Segoe UI", 10)); status_label.pack(pady=10)
    loading_queue = queue.Queue()
    threading.Thread(target=load_all_models, args=(loading_queue,), daemon=True).start()
    def check_loading_queue():
        try:
            message = loading_queue.get_nowait()
            if isinstance(message, dict):
                progress_bar.stop()
                for widget in root.winfo_children(): widget.destroy()
                app = TranslationApp(root, message)
            elif "Error" in message:
                 status_label.config(text=message); progress_bar.stop()
            else:
                status_label.config(text=message); root.after(100, check_loading_queue)
        except queue.Empty:
            root.after(100, check_loading_queue)
    root.after(100, check_loading_queue)
    root.mainloop()