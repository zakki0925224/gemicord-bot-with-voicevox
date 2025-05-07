from dotenv import load_dotenv
from pathlib import Path
import discord
import google.genai as genai
from google.genai import types
import os
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

# load .env
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
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
genai_client = genai.Client(api_key=GOOGLE_API_KEY)
search_tool = types.Tool(google_search=types.GoogleSearch())
chat = genai_client.chats.create(
    model=GEMINI_MODEL,
    history=[],
    config=types.GenerateContentConfig(
        system_instruction=CUSTOM_INSTRUCTIONS,
        tools=[search_tool],
    ),
)

# initialize discord client
discord_client = discord.Client(intents=discord.Intents.default())


def extract_sources(response):
    sources = []
    if not hasattr(response, "candidates"):
        return sources

    for candidate in response.candidates:
        if not hasattr(candidate, "grounding_metadata"):
            continue
        metadata = candidate.grounding_metadata
        if (
            not metadata
            or not hasattr(metadata, "grounding_chunks")
            or not metadata.grounding_chunks
        ):
            continue
        for chunk in metadata.grounding_chunks:
            web = chunk.web
            if web and web.title and web.uri:
                sources.append((web.title, web.uri))
    return sources


@discord_client.event
async def on_message(message):
    if message.author.bot or discord_client.user not in message.mentions:
        return

    try:
        response = chat.send_message(message=[message.content])
    except Exception as e:
        await message.channel.send(f"Error: {e}")
        return

    sources = extract_sources(response)
    response_text = response.text

    # voice channel
    if message.author.voice:
        if discord.utils.get(discord_client.voice_clients, guild=message.guild):
            voice_client = discord.utils.get(
                discord_client.voice_clients, guild=message.guild
            )
        else:
            voice_client = await message.author.voice.channel.connect()

        await message.channel.send("Synthesizing voice...")
        try:
            synthesize(response_text)
            voice_client.play(discord.FFmpegPCMAudio(wav_path))
        except Exception as e:
            await message.channel.send(f"Error: {e}")

    await message.channel.send(response_text)

    if len(sources) > 0:
        await message.channel.send("Sources:")
        for title, uri in sources:
            await message.channel.send(f"{title}: {uri}")


discord_client.run(DISCORD_BOT_TOKEN)
