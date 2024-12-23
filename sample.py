# from TTS.api import TTS
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# # generate speech by cloning a voice using default settings
# tts.tts_to_file(text="제 이름은 최종민입니다. 목소리가 참 좋죠? 오늘 하루도 고생많았어요.",
#                 file_path="output.wav",
#                 speaker_wav=["./speaker_Choi.wav"],
#                 language="ko",
#                 split_sentences=True)

import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("./config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./", use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["speaker_Choi.wav"])

print("Inference...")
t0 = time.time()
chunks = model.inference_stream(
    "테스트 모델입니다.",
    "ko",
    gpt_cond_latent,
    speaker_embedding
)

wav_chuncks = []
for i, chunk in enumerate(chunks):
    if i == 0:
        print(f"Time to first chunck: {time.time() - t0}")
    print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
    wav_chuncks.append(chunk)
wav = torch.cat(wav_chuncks, dim=0)
torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)