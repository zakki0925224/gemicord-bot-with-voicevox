from dotenv import load_dotenv
from pathlib import Path
import discord
import google.generativeai as genai
import os
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

# load .env
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CUSTOM_INSTRUCTIONS = os.getenv("CUSTOM_INSTRUCTIONS")
VOICEVOX_STYLE_ID = int(os.getenv("VOICEVOX_STYLE_ID"))

# initialize voicevox core
wav_path = Path("output.wav")
synthesizer = Synthesizer(
    Onnxruntime.load_once(
        filename=f"voicevox_core/onnxruntime/lib/{Onnxruntime.LIB_VERSIONED_FILENAME}"
    ),
    OpenJtalk("voicevox_core/dict/open_jtalk_dic_utf_8-1.11"),
)

with VoiceModelFile.open("voicevox_core/models/vvms/0.vvm") as model:
    synthesizer.load_voice_model(model)

def synthesize(text):
    wav = synthesizer.tts(text, VOICEVOX_STYLE_ID)
    wav_path.write_bytes(wav)

# initialize gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(system_instruction=CUSTOM_INSTRUCTIONS)
chat = model.start_chat(history=[])

# initialize discord client
discord_client = discord.Client(intents=discord.Intents.default())

@discord_client.event
async def on_message(message):
    if message.author.bot or discord_client.user not in message.mentions:
        return

    response = chat.send_message(message.content)
    response_text = response.text

    # voice channel
    if message.author.voice:
        voice_client = await message.author.voice.channel.connect()
        await message.channel.send("Synthesizing voice...")
        synthesize(response_text)
        voice_client.play(discord.FFmpegPCMAudio(wav_path))

    await message.channel.send(response_text)

discord_client.run(DISCORD_BOT_TOKEN)