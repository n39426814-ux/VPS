import gradio as gr
from TTS.api import TTS
import torch, os
from moviepy.editor import *
import spaces

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

VOICES = {
    "هادئ - ناشيونال": "natgeo_calm.wav",
    "حماسي - ناشيونال": "natgeo_epic.wav"
}

@spaces.GPU
def create_video(text, voice_name, scene_desc):
    if not text:
        return None, "اكتب النص أولاً!"
    voice_file = VOICES[voice_name]
    if not os.path.exists(voice_file):
        return None, f"باقي ترفع {voice_file}"
    
    tts.tts_to_file(text=text, speaker_wav=voice_file, language="ar", file_path="/tmp/voice.wav")
    audio = AudioFileClip("/tmp/voice.wav")
    bg = ColorClip(size=(1280,720), color=(15,30,80), duration=audio.duration)
    final = bg.set_audio(audio)
    final.write_videofile("/tmp/vangogh.mp4", fps=24, codec="libx264", audio_codec="aac")
    return "/tmp/vangogh.mp4", "✅ تم!"

demo = gr.Interface(
    fn=create_video,
    inputs=[
        gr.Textbox(label="نص القصة", lines=4),
        gr.Radio(["هادئ - ناشيونال", "حماسي - ناشيونال"], label="نوع الصوت", value="هادئ - ناشيونال"),
        gr.Textbox(label="وصف مشهد فان جوخ")
    ],
    outputs=[gr.Video(label="فيديو فان جوخ"), gr.Textbox(label="الحالة")],
    title="🔥 مصنع VPS - ناشيونال + فان جوخ"
)
demo.launch(share=True)
