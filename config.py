from dataclasses import dataclass
from environs import Env
from typing import Union, Optional
import os

@dataclass
class TgBot:
    token: str
    target_chat_id: int

@dataclass
class ModelConfig:
    path: str
    confidence_threshold: float

@dataclass
class Config:
    tg_bot: TgBot
    video_source: Union[int, str]
    model: ModelConfig
    skip_frames: int
    cooldown_minutes: float
    queue_size: int
    log_level: str

def load_config(path: Optional[str] = None) -> Config:
    env = Env()
    # Read environment variables from the file if provided and exists
    if path and os.path.exists(path):
        env.read_env(path)
    else:
        env.read_env()

    token = env.str('BOT_TOKEN', default=None)
    if not token:
        raise ValueError("BOT_TOKEN is required but was not provided.")

    target_chat_id = env.int('TARGET_CHAT_ID', default=None)
    if target_chat_id is None:
        raise ValueError("TARGET_CHAT_ID is required and must be an integer.")

    video_source_raw = env.str('VIDEO_SOURCE', default='0')
    # If the video source is a digit, parse it as an integer camera index
    if video_source_raw.isdigit():
        video_source: Union[int, str] = int(video_source_raw)
    else:
        video_source = video_source_raw

    model_path = env.str('MODEL_PATH', default='VandalGuardModel.pt')

    skip_frames = env.int('SKIP_FRAMES', default=3)
    if skip_frames <= 0:
        raise ValueError("SKIP_FRAMES must be a positive integer.")

    cooldown_minutes = env.float('COOLDOWN_MINUTES', default=3.0)
    if cooldown_minutes < 0:
        raise ValueError("COOLDOWN_MINUTES must be a non-negative number.")

    confidence_threshold = env.float('CONFIDENCE_THRESHOLD', default=0.5)
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0.")

    queue_size = env.int('QUEUE_SIZE', default=2)
    if queue_size <= 0:
        raise ValueError("QUEUE_SIZE must be a positive integer.")

    log_level = env.str('LOG_LEVEL', default='INFO').upper()

    return Config(
        tg_bot=TgBot(token=token, target_chat_id=target_chat_id),
        video_source=video_source,
        model=ModelConfig(path=model_path, confidence_threshold=confidence_threshold),
        skip_frames=skip_frames,
        cooldown_minutes=cooldown_minutes,
        queue_size=queue_size,
        log_level=log_level
    )