import re
import g4f
import json
import time
import shutil
import asyncio
import requests
import string
from utils import *
from cache import *
from config import *
from status import *
from uuid import uuid4
from constants import *
from typing import List
import assemblyai as aai
from moviepy.editor import *
from uploader import Uploader
from datetime import datetime
from termcolor import colored
from moviepy.video.fx.all import crop
from moviepy.audio.fx.all import volumex
from moviepy.config import change_settings
from moviepy.audio.fx.all import volumex as afx
from moviepy.video.tools.subtitles import SubtitlesClip
import re
import g4f
import json
import time
import requests
import assemblyai as aai

from utils import *
from cache import *
from config import *
from status import *
from uuid import uuid4
from constants import *
from typing import List
from moviepy.editor import *
from termcolor import colored
from moviepy.video.fx.all import crop
from moviepy.config import change_settings
from moviepy.video.tools.subtitles import SubtitlesClip
from datetime import datetime

# Set ImageMagick Path
change_settings({"IMAGEMAGICK_BINARY": get_imagemagick_path()})

class YouTube:
    def __init__(self, account_uuid: str, account_name: str, profile_path: str, niche: str, language: str) -> None:
        info(f"Initializing YouTube class for account: {account_name}")
        self._account_uuid: str = account_uuid
        self._account_name: str = account_name
        self._profile_path: str = profile_path
        self._niche: str = niche
        self._language: str = language
        self.images = []
        self.uploader = Uploader(profile_path)
        info(f"Niche: {niche}, Language: {language}")

    @property
    def niche(self) -> str:
        return self._niche
    
    @property
    def language(self) -> str:
        return self._language
    
    def generate_response(self, prompt: str, model: str = None) -> str:
        """
        Generates a response based on the given prompt using the specified model.
        If no model is provided, it uses the default model configured in the settings.
        """
        info(f"Generating response for prompt: {prompt[:50]}...")
        text_gen = get_text_gen()
        text_gen_model = get_text_gen_model()

        # Set default models for each provider
        if text_gen == "gemini":
            model = model or "gemini-pro"
            info(f"Using Google's Gemini model: {model}")
            import google.generativeai as genai
            genai.configure(api_key=get_gemini_api_key())
            response: str = genai.GenerativeModel(model).generate_content(prompt).text

        elif text_gen == "openai":
            model = model or "gpt-3.5-turbo"
            info(f"Using OpenAI model: {model}")
            import openai
            openai.api_key = get_openai_api_key()
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=150
            )["choices"][0]["text"]

        elif text_gen == "cluade":
            model = model or "claude-v1"
            info(f"Using Claude model: {model}")
            import anthropic
            anthropic_client = anthropic.Client(api_key=get_cluade_api_key())
            response = anthropic_client.completion(
                prompt=f"Human: {prompt}\nAssistant:",
                model=model,
                max_tokens_to_sample=150
            )["completion"]

        else:
            model = model or "gpt-3.5-turbo"
            info(f"Using model: {model}")
            response = g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
        info(f"Response generated successfully, length: {len(response)} characters")
        return response

    def generate_topic(self) -> str:
        """
        Generates a topic for the YouTube video based on the niche.
        """
        info("Generating topic for YouTube video")
        prompt = f"Please generate a specific video idea that takes about the following topic: {self.niche}. Make it exactly one sentence. Only return the topic, nothing else."
        completion = self.generate_response(prompt)

        if not completion:
            error("Failed to generate Topic.")
        else:
            self.subject = completion
            success(f"Generated topic: {completion}")

        return completion

    def generate_script(self) -> str:
        """
        Generates a script for the YouTube video based on the generated topic.
        """
        info("Generating script for YouTube video")
        prompt = f"""
        Generate a script for a video in 4 sentences, depending on the subject of the video.

        The script is to be returned as a string with the specified number of paragraphs.

        Do not under any circumstance reference this prompt in your response.

        Get straight to the point, don't start with unnecessary things like, "welcome to this video".

        Obviously, the script should be related to the subject of the video.
        
        YOU MUST NOT EXCEED THE 4 SENTENCES LIMIT. MAKE SURE THE 4 SENTENCES ARE SHORT.
        YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE.
        YOU MUST WRITE THE SCRIPT IN THE LANGUAGE SPECIFIED IN [LANGUAGE].
        ONLY RETURN THE RAW CONTENT OF THE SCRIPT. DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS OF WHAT SHOULD BE SPOKEN AT THE BEGINNING OF EACH PARAGRAPH OR LINE. YOU MUST NOT MENTION THE PROMPT, OR ANYTHING ABOUT THE SCRIPT ITSELF. ALSO, NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS OR LINES. JUST WRITE THE SCRIPT
        
        Subject: {self.subject}
        Language: {self.language}
        """
        completion = self.generate_response(prompt)

        # Apply regex to remove *
        completion = re.sub(r"\*", "", completion)
        
        if not completion:
            error("The generated script is empty.")
            return
        
        if len(completion) > 5000:
            warning("Generated Script is too long. Retrying...")
            return self.generate_script()
        
        self.script = completion
        success(f"Generated script: {completion[:100]}... Length: {len(completion)} characters")
    
        return completion

    def generate_metadata(self) -> dict:
        """
        Generates metadata (title and description) for the YouTube video.
        """
        info("Generating metadata for YouTube video")
        title_prompt = f"Please generate a YouTube Video Title for the following subject, including hashtags: {self.subject}. Only return the title, nothing else. Limit the title under 100 characters."
        title = self.generate_response(title_prompt)

        if len(title) > 100:
            warning("Generated Title is too long. Retrying...")
            return self.generate_metadata()

        description_prompt = f"Please generate a YouTube Video Description for the following script: {self.script}. Only return the description, nothing else."
        description = self.generate_response(description_prompt)
        
        self.metadata = {
            "title": title,
            "description": description
        }
        success(f"Generated metadata: {self.metadata}, Title length: {len(title)} characters, Description length: {len(description)} characters")

        return self.metadata
    
    def generate_prompts(self) -> List[str]:
        """
        Generates image prompts for AI image generation based on the video subject.
        """
        info("Generating image prompts for YouTube video")
        n_prompts = 3
        info(f"Number of prompts requested: {n_prompts}")

        prompt = f"""
        Generate {n_prompts} Image Prompts for AI Image Generation,
        depending on the subject of a video.
        Subject: {self.subject}

        The image prompts are to be returned as
        a JSON-Array of strings.

        Each search term should consist of a full sentence,
        always add the main subject of the video.

        Be emotional and use interesting adjectives to make the
        Image Prompt as detailed as possible.
        
        YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
        YOU MUST NOT RETURN ANYTHING ELSE. 
        YOU MUST NOT RETURN THE SCRIPT.
        
        The search terms must be related to the subject of the video.
        Here is an example of a JSON-Array of strings:
        ["image prompt 1", "image prompt 2", "image prompt 3"]

        For context, here is the full text:
        {self.script}
        """
        completion = self.generate_response(prompt)

        try:
            image_prompts = json.loads(completion)
        except json.JSONDecodeError:
            warning("Failed to parse JSON response for image prompts. Retrying...")
            return self.generate_prompts()

        if not isinstance(image_prompts, list):
            warning("Invalid JSON format for image prompts. Retrying...")
            return self.generate_prompts()

        self.image_prompts = image_prompts[:n_prompts]

        success(f"Generated {len(self.image_prompts)} Image Prompts.")

        return self.image_prompts

    def generate_images(self) -> List[str]:
        """
        Generates images based on the generated image prompts one by one.
        """
        info("Generating images for YouTube video")
        image_paths = []

        for i, prompt in enumerate(self.image_prompts, 1):
            info(f"Generating image {i}/{len(self.image_prompts)}")
            image_path = self.generate_image(prompt)
            if image_path:
                image_paths.append(image_path)

        self.images.extend(image_paths)
        success(f"Generated {len(image_paths)} images.")
        return image_paths

    def generate_image(self, prompt: str) -> str:
        """
        Generates an image based on the provided prompt using the configured image generation provider.
        """
        image_gen = get_image_gen()
        image_model = get_image_gen_model()
        
        info(f"Using: {image_gen} With Model: {image_model}")
        
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                if image_gen == "prodia":
                    info("Using Prodia provider for image generation")
                    s = requests.Session()
                    headers = {
                        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                    
                    # Generate job
                    info("Sending generation request to Prodia API")
                    resp = s.get(
                        "https://api.prodia.com/generate",
                        params={
                            "new": "true",
                            "prompt": prompt,
                            "model": image_model,
                            "negative_prompt": "verybadimagenegative_v1.3",
                            "steps": "20",
                            "cfg": "7",
                            "seed": random.randint(1, 10000),
                            "sample": "DPM++ 2M Karras",
                            "aspect_ratio": "square"
                        },
                        headers=headers
                    )
                    
                    job_id = resp.json()['job']
                    info(f"Job created with ID: {job_id}")
                    
                    while True:
                        time.sleep(5)
                        status = s.get(f"https://api.prodia.com/job/{job_id}", headers=headers).json()
                        if status["status"] == "succeeded":
                            info("Image generation successful, downloading result")
                            img_data = s.get(f"https://images.prodia.xyz/{job_id}.png?download=1", headers=headers).content
                            image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                            with open(image_path, "wb") as f:
                                f.write(img_data)
                            self.images.append(image_path)
                            success(f"Image saved to: {image_path}")
                            return image_path

                elif image_gen == "hercai":
                    info("Using Hercai provider for image generation")
                    url = f"https://hercai.onrender.com/{image_model}/text2image?prompt={prompt}"
                    r = requests.get(url)
                    parsed = r.json()

                    if "url" in parsed and parsed["url"]:
                        info("Image URL received from Hercai")
                        image_url = parsed["url"]
                        image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                        with open(image_path, "wb") as f:
                            image_data = requests.get(image_url).content
                            f.write(image_data)
                        self.images.append(image_path)
                        success(f"Image saved to: {image_path}")
                        return image_path
                    else:
                        warning("No image URL in Hercai response")

                elif image_gen == "pollinations":
                    info("Using Pollinations provider for image generation")
                    response = requests.get(f"https://image.pollinations.ai/prompt/{prompt}{random.randint(1,10000)}")
                    if response.status_code == 200:
                        info("Image received from Pollinations")
                        image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                        with open(image_path, "wb") as f:
                            f.write(response.content)
                        self.images.append(image_path)
                        success(f"Image saved to: {image_path}")
                        return image_path
                    else:
                        warning(f"Pollinations request failed with status code: {response.status_code}")

                elif image_gen == "segmind":
                    info("Using Segmind provider for image generation")
                    url = f"https://api.segmind.com/v1/generate"
                    payload = {
                        "prompt": prompt,
                        "model": image_model,
                        "negative_prompt": "low quality, blurry, dark, bad anatomy",
                        "steps": 20,
                        "cfg_scale": 7,
                        "width": 512,
                        "height": 512,
                        "samples": 1,
                        "scheduler": "ddim"
                    }
                    headers = {
                        "Authorization": f"Bearer {get_segmind_api_key()}",
                        "Content-Type": "application/json"
                    }
                    response = requests.post(url, json=payload, headers=headers)
                    if response.status_code == 200:
                        img_data = response.json()['images'][0]
                        image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                        with open(image_path, "wb") as f:
                            f.write(base64.b64decode(img_data))
                        self.images.append(image_path)
                        success(f"Image saved to: {image_path}")
                        return image_path
                    else:
                        warning(f"Segmind request failed with status code: {response.status_code}")

                elif image_gen == "openai":
                    info("Using OpenAI provider for image generation")
                    import openai
                    openai.api_key = get_openai_api_key()
                    response = openai.Image.create(
                        prompt=prompt,
                        model=image_model,
                        n=1,
                        size="1024x1024"
                    )
                    if response:
                        image_url = response['data'][0]['url']
                        image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                        with open(image_path, "wb") as f:
                            image_data = requests.get(image_url).content
                            f.write(image_data)
                        self.images.append(image_path)
                        success(f"Image saved to: {image_path}")
                        return image_path
                    else:
                        warning("No image URL in OpenAI response")

                elif image_gen == "stable_diffusion_cpu":
                    info("Using Local Stable Diffusion CPU provider for image generation")
                    from diffusers import StableDiffusionPipeline
                    import torch

                    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
                    pipe = pipe.to("cpu")

                    image = pipe(prompt).images[0]
                    image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                    image.save(image_path)
                    self.images.append(image_path)
                    success(f"Image saved to: {image_path}")
                    return image_path

                elif image_gen == "stability_ai":
                    info("Using Stability AI provider for image generation")
                    import stability_sdk.client
                    from stability_sdk.utils import convert_bytes_to_image

                    stability_host = "grpc.stability.ai:443"
                    stability_key = get_stability_api_key()

                    stability_client = stability_sdk.client.StabilityInference(
                        key=stability_key,
                        host=stability_host,
                        engine="stable-diffusion-xl-1024-v1-0"
                    )

                    answers = stability_client.generate(
                        prompt=prompt,
                        steps=20,
                        cfg_scale=7.0,
                        width=512,
                        height=512,
                        samples=1,
                        sampler=stability_sdk.client.SAMPLER_K_EULER_A
                    )

                    for resp in answers:
                        for artifact in resp.artifacts:
                            if artifact.finish_reason == stability_sdk.client.FILTER:
                                warning(f"Your request activated the API's safety filters and could not be processed.")
                                return None
                            if artifact.type == stability_sdk.client.ARTIFACT_IMAGE:
                                image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                                img = convert_bytes_to_image(artifact.binary)
                                img.save(image_path)
                                self.images.append(image_path)
                                success(f"Image saved to: {image_path}")
                                return image_path

            except Exception as e:
                warning(f"Image generation failed: {str(e)}")

            retries += 1
            warning(f"Retry {retries}/{max_retries} for image generation with {image_gen} failed.")

        error(f"Failed to generate image after {max_retries} retries.")
        return None

    def generate_speech(self, text: str, output_format: str = 'mp3') -> str:
        """
        Generates speech from the provided text using the configured TTS engine.
        """
        info("Generating speech from text")
        
        # Clean text
        text = re.sub(r'[^\w\s.?!]', '', text)
        
        tts_engine = get_speech_gen()
        tts_voice = get_speech_gen_voice()

        info(f"Using TTS Engine: {tts_engine}, Voice: {tts_voice}")
        
        audio_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.{output_format}")
        
        if tts_engine == "elevenlabs":
            # Latest ElevenLabs API implementation
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": get_elevenlabs_api_key()
            }
            
            payload = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{tts_voice}",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                success(f"Speech generated successfully using ElevenLabs at {audio_path}")
                self.tts_path = audio_path
                return audio_path
            else:
                error(f"ElevenLabs API error: {response.text}")
                return None
                
        elif tts_engine == 'bark':
            from bark import SAMPLE_RATE, generate_audio, preload_models
            preload_models()
            audio_array = generate_audio(text)
            import soundfile as sf
            sf.write(audio_path, audio_array, SAMPLE_RATE)
            
        elif tts_engine == "gtts":
            info("Using Google TTS provider for speech generation")
            from gtts import gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(audio_path)
            
        elif tts_engine == "openai":
            info("Using OpenAI provider for speech generation")
            import openai
            openai.api_key = get_openai_api_key()
            response = openai.Audio.speech.create(
                model="tts-1",
                voice=tts_voice,
                input=text
            )
            response.stream_to_file(audio_path)
            
        elif tts_engine == "edge":
            info("Using Edge TTS provider for speech generation")
            import edge_tts
            import asyncio
            async def generate():
                communicate = edge_tts.Communicate(text, tts_voice)
                await communicate.save(audio_path)
            asyncio.run(generate())
            
        elif tts_engine == "local_tts":
            info("Using Local TTS provider for speech generation")
            import requests
            
            url = "https://imseldrith-tts-openai-free.hf.space/v1/audio/speech"
            
            payload = {
                "model": "tts-1",
                "input": text,
                "voice": tts_voice,
                "response_format": "mp3",
                "speed": 0.60
            }
            
            headers = {
                "accept": "*/*",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                with open(audio_path, "wb") as f:
                    f.write(response.content)
                success(f"Speech generated successfully at {audio_path}")
            else:
                error(f"Failed to generate speech: {response.text}")
                return None
                
        elif tts_engine == "xtts":
            info("Using XTTS-v2 provider for speech generation")
            from TTS.api import TTS
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            tts.tts_to_file(
                text=text,
                file_path=audio_path,
                speaker=tts_voice,
                language="en"
            )
            
        elif tts_engine == "rvc":
            info("Using RVC provider for speech generation")
            from rvc_engine import RVCEngine
            
            # First generate base audio using GTTS
            temp_path = os.path.join(ROOT_DIR, ".mp", f"temp_{uuid4()}.wav")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_path)
            
            # Convert using RVC
            rvc = RVCEngine(model_path=get_rvc_model_path())
            rvc.convert(
                input_path=temp_path,
                output_path=audio_path,
                f0_method='dio'  # CPU-friendly method
            )
            
            # Cleanup temp file
            os.remove(temp_path)
            
        else:
            error(f"Unsupported TTS engine: {tts_engine}")
            return None
            
        success(f"Speech generated and saved to: {audio_path}")
        self.tts_path = audio_path
        return audio_path

    def generate_subtitles(self, audio_path: str) -> str:
        """
        Generates subtitles for the audio using AssemblyAI.
        """
        info("Starting subtitle generation process...")
        info(f"Using audio file: {audio_path}")
        
        info("Initializing AssemblyAI configuration...")
        aai.settings.api_key = get_assemblyai_api_key()
        config = aai.TranscriptionConfig()
        transcriber = aai.Transcriber(config=config)
        
        info("Transcribing audio to text... This may take a few minutes...")
        transcript = transcriber.transcribe(audio_path)
        info("Audio transcription completed successfully")
        
        info("Converting transcript to SRT format...")
        subtitles = transcript.export_subtitles_srt()
        
        info("Saving subtitles to file...")
        srt_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.srt")
        with open(srt_path, "w") as file:
            file.write(subtitles)
        info(f"Subtitles saved to: {srt_path}")
        
        info("Equalizing subtitles for better timing...")
        equalize_subtitles(srt_path, 10)
        success(f"Subtitle generation complete! File location: {srt_path}")
        return srt_path

    def combine(self) -> str:
        """
        Combines all elements into final video with proper subtitles and font.
        """
        info("Starting video combination process") 
        combined_image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.mp4")
        
        info("Loading TTS audio clip...")
        tts_clip = AudioFileClip(self.tts_path)
        max_duration = tts_clip.duration
        req_dur = max_duration / len(self.images)
        info(f"Total video duration will be: {max_duration:.2f} seconds")
        info(f"Each image will be shown for: {req_dur:.2f} seconds")

        font_path = os.path.join(get_fonts_dir(), get_font())
        info(f"Using font for subtitles: {font_path}")
        
        info("Setting up text generator for subtitles...")
        generator = lambda txt: TextClip(
            txt,
            font=font_path,
            fontsize=100,
            color="#FFFF00",
            stroke_color="black", 
            stroke_width=5,
            size=(1080, 1920),
            method="caption"
        )

        info("Processing images and creating video clips...")
        clips = []
        tot_dur = 0
        
        while tot_dur < max_duration:
            for idx, image_path in enumerate(self.images):
                info(f"Processing image {idx + 1}/{len(self.images)}: {image_path}")
                clip = ImageClip(image_path)
                clip.duration = req_dur
                clip = clip.set_fps(30)

                if round((clip.w/clip.h), 4) < 0.5625:
                    info(f"Resizing image {idx + 1} to vertical format (1080x1920)")
                    clip = crop(clip, width=clip.w, height=round(clip.w/0.5625),
                              x_center=clip.w/2, y_center=clip.h/2)
                else:
                    info(f"Resizing image {idx + 1} to horizontal format (1920x1080)")
                    clip = crop(clip, width=round(0.5625*clip.h), height=clip.h,
                              x_center=clip.w/2, y_center=clip.h/2)
                
                clip = clip.resize((1080, 1920))
                clips.append(clip)
                tot_dur += clip.duration
                info(f"Current total duration: {tot_dur:.2f} seconds")

        info("Concatenating all video clips...")
        final_clip = concatenate_videoclips(clips)
        final_clip = final_clip.set_fps(30)
        
        info("Loading background music...")
        random_song = choose_random_music()
        info(f"Selected background track: {random_song}")
        random_song_clip = AudioFileClip(random_song).set_fps(44100)
        info("Adjusting background music volume to 10%")
        random_song_clip = random_song_clip.fx(volumex, 0.1)
        
        info("Generating subtitles from audio...")
        subtitles_path = self.generate_subtitles(self.tts_path)
        info(f"Generated subtitles at: {subtitles_path}")
        
        info("Equalizing subtitles timing...")
        equalize_subtitles(subtitles_path, 10)
        info("Subtitles equalized successfully")
        
        info("Creating subtitle clips...")
        subtitles = SubtitlesClip(subtitles_path, generator)
        info("Setting subtitles position to center")
        subtitles = subtitles.set_pos(("center", "center"))
        
        info("Combining audio tracks (TTS + background music)...")
        comp_audio = CompositeAudioClip([
            tts_clip.set_fps(44100),
            random_song_clip
        ])

        info("Setting final video properties...")
        final_clip = final_clip.set_audio(comp_audio)
        final_clip = final_clip.set_duration(tts_clip.duration)
        
        info("Burning subtitles into video...")
        final_clip = CompositeVideoClip([
            final_clip,
            subtitles
        ])

        info(f"Writing final video file to: {combined_image_path}")
        info(f"Using {get_threads()} threads for video processing")
        final_clip.write_videofile(combined_image_path, threads=get_threads())
        
        success(f"Video generation complete! Final video saved at: {combined_image_path}")
        info(f"Final video duration: {final_clip.duration:.2f} seconds")
        info(f"Final video resolution: {final_clip.size[0]}x{final_clip.size[1]}")
        
        return combined_image_path



    def save_metadata(self):
        """
        Saves the metadata and copies the final video to the output directory.
        """
        info("Creating active_folder and saving metadata")
        videos_dir = os.path.join(ROOT_DIR, "Videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        existing_folders = [f for f in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir, f))]
        last_number = max([int(f.split('.')[0]) for f in existing_folders if f.split('.')[0].isdigit()] or [0])
        
        new_folder_number = last_number + 1
        sanitized_subject = ''.join(c for c in self.subject if c.isalnum() or c.isspace())
        folder_name = f"{new_folder_number}. {sanitized_subject}"
        
        active_folder = os.path.join(videos_dir, folder_name)
        os.makedirs(active_folder, exist_ok=True)

        metadata_file = os.path.join(active_folder, "metadata.txt")
        with open(metadata_file, "w") as f:
            f.write(f"Title: {self.metadata['title']}\n")
            f.write(f"Description: {self.metadata['description']}")

        shutil.copy2(self.video_path, os.path.join(active_folder, os.path.basename(self.video_path)))
        success(f"Metadata and video saved to: {active_folder}")

    def generate_video(self) -> str:
        """
        Generates the complete video with all components: topic, script, metadata, images, speech, subtitles, and combines them.
        """
        info("Starting video generation process")
        
        info("Generating topic")
        self.generate_topic()
        
        info("Generating script")
        self.generate_script()
        
        info("Generating metadata")
        self.generate_metadata()
        
        info("Generating image prompts")
        self.generate_prompts()
        
        info("Generating images")
        self.generate_images()
        
        info("Generating speech")
        self.generate_speech(self.script)
        
        info("Combining all elements into final video")
        path = self.combine()
        
        info(f"Video generation complete. File saved at: {path}")
        self.video_path = os.path.abspath(path)

        info("Saving metadata and video to active_folder")
        self.save_metadata()
        
        return path