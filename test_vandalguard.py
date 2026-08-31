import sys
from unittest.mock import MagicMock

# Mock cv2 and ultralytics modules before they are imported by main
sys.modules['cv2'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['ultralytics'].YOLO = MagicMock()

import unittest
import numpy as np
import asyncio
import os
from config import load_config
from main import analyze_frame

class TestVandalGuardConfig(unittest.TestCase):
    def setUp(self):
        # Save existing environment variables to restore them later
        self.env_backup = dict(os.environ)

    def tearDown(self):
        # Restore environment variables
        os.environ.clear()
        os.environ.update(self.env_backup)

    def test_config_missing_bot_token(self):
        if "BOT_TOKEN" in os.environ:
            del os.environ["BOT_TOKEN"]
        os.environ["TARGET_CHAT_ID"] = "12345"
        with self.assertRaises(ValueError) as ctx:
            load_config()
        self.assertIn("BOT_TOKEN is required", str(ctx.exception))

    def test_config_missing_target_chat_id(self):
        os.environ["BOT_TOKEN"] = "fake_token"
        if "TARGET_CHAT_ID" in os.environ:
            del os.environ["TARGET_CHAT_ID"]
        with self.assertRaises(ValueError) as ctx:
            load_config()
        self.assertIn("TARGET_CHAT_ID is required", str(ctx.exception))

    def test_config_invalid_target_chat_id(self):
        os.environ["BOT_TOKEN"] = "fake_token"
        os.environ["TARGET_CHAT_ID"] = "not_an_int"
        with self.assertRaises(ValueError):
            load_config()

    def test_config_defaults(self):
        os.environ["BOT_TOKEN"] = "fake_token"
        os.environ["TARGET_CHAT_ID"] = "1234567"
        for key in ["VIDEO_SOURCE", "MODEL_PATH", "SKIP_FRAMES", "COOLDOWN_MINUTES", "CONFIDENCE_THRESHOLD", "QUEUE_SIZE", "LOG_LEVEL"]:
            if key in os.environ:
                del os.environ[key]

        cfg = load_config()
        self.assertEqual(cfg.tg_bot.token, "fake_token")
        self.assertEqual(cfg.tg_bot.target_chat_id, 1234567)
        self.assertEqual(cfg.video_source, 0)
        self.assertEqual(cfg.model.path, "VandalGuardModel.pt")
        self.assertEqual(cfg.skip_frames, 3)
        self.assertEqual(cfg.cooldown_minutes, 3.0)
        self.assertEqual(cfg.model.confidence_threshold, 0.5)
        self.assertEqual(cfg.queue_size, 2)
        self.assertEqual(cfg.log_level, "INFO")

    def test_config_custom_values(self):
        os.environ["BOT_TOKEN"] = "another_token"
        os.environ["TARGET_CHAT_ID"] = "-100123456789"
        os.environ["VIDEO_SOURCE"] = "http://example.com/stream"
        os.environ["MODEL_PATH"] = "custom.pt"
        os.environ["SKIP_FRAMES"] = "5"
        os.environ["COOLDOWN_MINUTES"] = "1.5"
        os.environ["CONFIDENCE_THRESHOLD"] = "0.85"
        os.environ["QUEUE_SIZE"] = "10"
        os.environ["LOG_LEVEL"] = "DEBUG"

        cfg = load_config()
        self.assertEqual(cfg.tg_bot.token, "another_token")
        self.assertEqual(cfg.tg_bot.target_chat_id, -100123456789)
        self.assertEqual(cfg.video_source, "http://example.com/stream")
        self.assertEqual(cfg.model.path, "custom.pt")
        self.assertEqual(cfg.skip_frames, 5)
        self.assertEqual(cfg.cooldown_minutes, 1.5)
        self.assertEqual(cfg.model.confidence_threshold, 0.85)
        self.assertEqual(cfg.queue_size, 10)
        self.assertEqual(cfg.log_level, "DEBUG")

    def test_config_invalid_numeric_values(self):
        os.environ["BOT_TOKEN"] = "token"
        os.environ["TARGET_CHAT_ID"] = "123"

        os.environ["SKIP_FRAMES"] = "0"
        with self.assertRaises(ValueError) as ctx:
            load_config()
        self.assertIn("SKIP_FRAMES must be a positive integer", str(ctx.exception))
        os.environ["SKIP_FRAMES"] = "3"

        os.environ["COOLDOWN_MINUTES"] = "-1"
        with self.assertRaises(ValueError) as ctx:
            load_config()
        self.assertIn("COOLDOWN_MINUTES must be a non-negative number", str(ctx.exception))
        os.environ["COOLDOWN_MINUTES"] = "3"

        os.environ["CONFIDENCE_THRESHOLD"] = "1.5"
        with self.assertRaises(ValueError) as ctx:
            load_config()
        self.assertIn("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0", str(ctx.exception))
        os.environ["CONFIDENCE_THRESHOLD"] = "0.5"

        os.environ["QUEUE_SIZE"] = "-2"
        with self.assertRaises(ValueError) as ctx:
            load_config()
        self.assertIn("QUEUE_SIZE must be a positive integer", str(ctx.exception))


class TestVandalGuardDetection(unittest.TestCase):
    def create_mock_yolo_prediction(self, class_index: int, confidence: float, names_dict=None):
        mock_result = MagicMock()
        mock_probs = MagicMock()
        mock_probs.top1 = class_index

        mock_top1conf = MagicMock()
        mock_top1conf.item.return_value = confidence
        mock_probs.top1conf = mock_top1conf

        mock_result.probs = mock_probs
        if names_dict is not None:
            mock_result.names = names_dict
        else:
            if hasattr(mock_result, 'names'):
                del mock_result.names

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        return mock_model

    def test_analyze_frame_vandalism_detected(self):
        mock_model = self.create_mock_yolo_prediction(1, 0.8, {0: "nonVandalism", 1: "vAnDaLiSm"})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        is_vandalism, confidence, idx, name = analyze_frame(frame, mock_model, 0.5)
        self.assertTrue(is_vandalism)
        self.assertEqual(confidence, 0.8)
        self.assertEqual(idx, 1)
        self.assertEqual(name, "vAnDaLiSm")

    def test_analyze_frame_low_confidence(self):
        mock_model = self.create_mock_yolo_prediction(1, 0.4, {0: "nonVandalism", 1: "vandalism"})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        is_vandalism, confidence, idx, name = analyze_frame(frame, mock_model, 0.5)
        self.assertFalse(is_vandalism)
        self.assertEqual(confidence, 0.4)

    def test_analyze_frame_non_vandalism(self):
        mock_model = self.create_mock_yolo_prediction(0, 0.9, {0: "nonVandalism", 1: "vandalism"})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        is_vandalism, confidence, idx, name = analyze_frame(frame, mock_model, 0.5)
        self.assertFalse(is_vandalism)

    def test_analyze_frame_names_unavailable_fallback(self):
        mock_model = self.create_mock_yolo_prediction(1, 0.7, names_dict=None)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        is_vandalism, confidence, idx, name = analyze_frame(frame, mock_model, 0.5)
        self.assertTrue(is_vandalism)
        self.assertEqual(idx, 1)
        self.assertEqual(name, "vandalism")

    def test_analyze_frame_names_unavailable_fallback_non_vandalism(self):
        mock_model = self.create_mock_yolo_prediction(0, 0.7, names_dict=None)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        is_vandalism, confidence, idx, name = analyze_frame(frame, mock_model, 0.5)
        self.assertFalse(is_vandalism)


class TestVandalGuardQueue(unittest.IsolatedAsyncioTestCase):
    async def test_queue_drop_oldest_frame(self):
        queue = asyncio.Queue(maxsize=2)

        await queue.put("frame_1")
        await queue.put("frame_2")
        self.assertTrue(queue.full())

        if queue.full():
            dropped = queue.get_nowait()
            queue.task_done()
            self.assertEqual(dropped, "frame_1")

        await queue.put("frame_3")

        self.assertEqual(queue.qsize(), 2)
        self.assertEqual(await queue.get(), "frame_2")
        self.assertEqual(await queue.get(), "frame_3")

if __name__ == "__main__":
    unittest.main()
