import asyncio
import logging
import sys
import signal
import cv2
import numpy as np
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, BufferedInputFile
from ultralytics import YOLO
from config import Config, load_config

# Setup placeholder logger until config is loaded
logger = logging.getLogger("VandalGuard")

dp = Dispatcher()

# Global variables for cooldown management
last_vandalism_time = None
vandalism_detection_lock = asyncio.Lock()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(
        "Добро пожаловать!\nЭтот бот будет отправлять уведомления о случаях вандализма, "
        "обнаруженных нашей системой при помощи камеры в лифте вашего подъезда."
    )

@dp.message()
async def echo_handler(message: Message) -> None:
    try:
        await message.send_copy(chat_id=message.chat.id)
    except TypeError:
        await message.answer("Nice try!")

def analyze_frame(frame: np.ndarray, model: YOLO, confidence_threshold: float) -> tuple[bool, float, int, str]:
    """
    Performs inference and checks if vandalism is detected.
    Returns:
        (is_vandalism, confidence, class_index, class_name)
    """
    prediction_results = model.predict(source=frame, verbose=False)
    if not prediction_results:
        return False, 0.0, -1, ""

    result = prediction_results[0]
    if result.probs is None:
        return False, 0.0, -1, ""

    probs = result.probs
    predicted_class_index = probs.top1
    predicted_confidence = probs.top1conf.item()

    class_names = getattr(result, "names", None)
    predicted_class_name = ""
    is_vandalism = False

    if class_names and predicted_class_index in class_names:
        predicted_class_name = class_names[predicted_class_index]
        if str(predicted_class_name).strip().lower() == "vandalism":
            is_vandalism = True
    else:
        # Safe index-1 fallback only when names unavailable
        if predicted_class_index == 1:
            is_vandalism = True
            predicted_class_name = "vandalism"

    # Check confidence threshold
    if is_vandalism and predicted_confidence >= confidence_threshold:
        return True, predicted_confidence, predicted_class_index, predicted_class_name

    return False, predicted_confidence, predicted_class_index, predicted_class_name

async def process_frame(frame: np.ndarray, bot: Bot, target_chat_id: int, model: YOLO, config: Config):
    global last_vandalism_time
    try:
        is_vandalism, confidence, class_idx, class_name = analyze_frame(
            frame, model, config.model.confidence_threshold
        )
        if not is_vandalism:
            return

        current_time = datetime.now()
        async with vandalism_detection_lock:
            # Re-check cooldown under the lock
            if last_vandalism_time is not None and (current_time - last_vandalism_time) < timedelta(minutes=config.cooldown_minutes):
                return
            last_vandalism_time = current_time

        logger.warning(f"Detection alert triggered! Vandalism detected with confidence: {confidence:.4f}")

        caption_text = (
            f"🚨🚨🚨 Обнаружен акт вандализма!!!!! 🚨🚨🚨\n"
            f"Время: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Точность: {confidence:.4f}"
        )

        try:
            success, encoded_img = cv2.imencode('.jpg', frame)
            if not success:
                raise ValueError("cv2.imencode returned False")

            photo_file = BufferedInputFile(encoded_img.tobytes(), filename="snapshot.jpg")
            await bot.send_photo(chat_id=target_chat_id, photo=photo_file, caption=caption_text)
            logger.info("Sent Telegram alert with snapshot.")
        except Exception as img_err:
            logger.error(f"Failed to encode or send snapshot: {img_err}. Sending text fallback.")
            await bot.send_message(chat_id=target_chat_id, text=caption_text)
            logger.info("Sent fallback text Telegram alert.")

    except Exception as e:
        logger.error(f"Error during frame processing: {e}", exc_info=True)

async def worker(queue: asyncio.Queue, bot: Bot, target_chat_id: int, model: YOLO, config: Config):
    logger.info("Worker task started.")
    try:
        while True:
            frame = await queue.get()
            try:
                await process_frame(frame, bot, target_chat_id, model, config)
            except Exception as e:
                logger.error(f"Error processing frame in worker: {e}", exc_info=True)
            finally:
                queue.task_done()
    except asyncio.CancelledError:
        logger.info("Worker task cancelled.")
        raise

async def main() -> None:
    # 1. Load configuration
    try:
        config: Config = load_config('.env')
    except Exception as e:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        logging.error(f"Configuration load error: {e}")
        sys.exit(1)

    # 2. Configure logging
    log_level = getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logger.info("VandalGuard initializing...")

    # 3. Load YOLO model
    logger.info(f"Loading YOLO model from: {config.model.path}")
    try:
        model = YOLO(config.model.path)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
        sys.exit(1)

    # 4. Initialize bot and queue
    bot = Bot(token=config.tg_bot.token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    queue = asyncio.Queue(maxsize=config.queue_size)

    stop_event = asyncio.Event()

    # 5. Capture loop
    async def capture_loop():
        logger.info(f"Opening video source: {config.video_source}")
        cap = await asyncio.to_thread(cv2.VideoCapture, config.video_source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {config.video_source}")
            raise RuntimeError(f"Could not open video source: {config.video_source}")

        frame_count = 0
        try:
            while not stop_event.is_set():
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret:
                    logger.info("End of video stream or error reading frame.")
                    stop_event.set()
                    break

                frame_count += 1
                if frame_count % config.skip_frames == 0:
                    if queue.full():
                        try:
                            # Drop the oldest frame to process the newest frame
                            _ = queue.get_nowait()
                            queue.task_done()
                            logger.warning("Queue full. Dropped oldest stale frame.")
                        except asyncio.QueueEmpty:
                            pass
                    await queue.put(frame)

                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("Capture loop cancelled.")
            raise
        finally:
            logger.info("Releasing video capture...")
            await asyncio.to_thread(cap.release)

    # Register SIGINT and SIGTERM handlers to set stop_event
    loop = asyncio.get_running_loop()

    def handle_signal(sig):
        logger.info(f"Received signal {sig.name}. Initiating graceful shutdown...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
        except NotImplementedError:
            pass  # add_signal_handler might not be supported on some platforms (e.g. Windows)

    # Start tasks
    worker_task = asyncio.create_task(worker(queue, bot, config.tg_bot.target_chat_id, model, config))
    capture_task = asyncio.create_task(capture_loop())
    polling_task = asyncio.create_task(dp.start_polling(bot, handle_signals=False))

    async def wait_for_stop():
        await stop_event.wait()

    stop_wait_task = asyncio.create_task(wait_for_stop())

    try:
        # Wait for either stop event or one of the tasks to complete/error
        done, pending = await asyncio.wait(
            [stop_wait_task, worker_task, capture_task, polling_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        logger.info("Shutdown initiated.")
    except Exception as e:
        logger.error(f"Error in main wait loop: {e}", exc_info=True)
    finally:
        # Ensure stop event is set to trigger other tasks shutdown
        stop_event.set()

        # Stop bot polling explicitly to ensure clean shutdown
        try:
            logger.info("Stopping bot polling...")
            await dp.stop_polling()
        except Exception as e:
            logger.error(f"Error stopping polling: {e}")

        # Cancel all running tasks
        for task in [worker_task, capture_task, polling_task, stop_wait_task]:
            if not task.done():
                task.cancel()

        # Wait for all tasks to finalize
        results = await asyncio.gather(
            worker_task, capture_task, polling_task, stop_wait_task,
            return_exceptions=True
        )

        # Log any errors from tasks other than CancelledError
        for name, res in zip(["worker", "capture", "polling", "stop_wait"], results):
            if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                logger.error(f"Error in {name} task: {res}", exc_info=res)

        logger.info("Closing bot session...")
        await bot.session.close()
        logger.info("Shutdown sequence finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received SIGINT/KeyboardInterrupt. Exiting...")
