"""
Emotion detection module using DeepFace (pretrained models).

This module detects facial emotions in real-time from images and video frames.
DeepFace uses state-of-the-art pretrained models and doesn't require any API keys.

Emotions detected: angry, fear, neutral, sad, disgust, happy, surprise
"""

import os
import cv2
import numpy as np
from typing import Optional, Dict
from dotenv import load_dotenv

try:
    from emotion_behavior.core import detect_emotion_from_bgr, load_emotion_predictor

    LOCAL_MODEL_AVAILABLE = True
except Exception as exc:
    LOCAL_MODEL_AVAILABLE = False
    print(f"[emotion_detector] Local emotion model import error: {exc}")

try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass

try:
    from hume import AsyncHumeClient  # noqa: F401

    HUME_SDK_AVAILABLE = True
except ImportError as exc:
    HUME_SDK_AVAILABLE = False
    print(f"[emotion_detector] Hume SDK import error: {exc}")

load_dotenv()

HUME_API_KEY = os.getenv("HUME_API_KEY", "").strip().strip('"').strip("'")
HUME_PROB_THRESHOLD = float(os.getenv("HUME_PROB_THRESHOLD", "0.2"))
USE_HUME = (
    os.getenv("USE_HUME", "0") == "1"
    and bool(HUME_API_KEY)
    and HUME_SDK_AVAILABLE
)

MAIN_EMOTIONS = {
    "Joy",
    "Sadness",
    "Anger",
    "Fear",
    "Surprise (positive)",
    "Surprise (negative)",
    "Disgust",
    "Calmness",
    "Excitement",
    "Contentment",
}

EMOTION_DETECTION_AVAILABLE = True
_LAST_ERROR: Optional[str] = None

print(
    f"[emotion_detector] Config: USE_HUME={USE_HUME}, "
    f"API_KEY={'SET' if HUME_API_KEY else 'MISSING'}, "
    f"SDK={'OK' if HUME_SDK_AVAILABLE else 'MISSING'}, "
    f"THRESHOLD={HUME_PROB_THRESHOLD}"
)


def get_last_detection_error() -> Optional[str]:
    """Return the most recent detection failure message for UI display."""
    return _LAST_ERROR


def analyze_frame(frame_data: np.ndarray) -> Optional[str]:
    """
    Analyze a video frame for emotion detection.

  Priority:
      1. Hume SDK streaming API
      2. Hume SDK batch API (snapshot-friendly fallback)
      3. OpenCV Haar-cascade detector (only when Hume is disabled)
    """
    global _LAST_ERROR
    _LAST_ERROR = None

    if frame_data is None or frame_data.size == 0:
        _LAST_ERROR = "Empty image frame."
        return None

    if LOCAL_MODEL_AVAILABLE:
        try:
            prediction = detect_emotion_from_bgr(frame_data)
            if prediction:
                _LAST_ERROR = None
                print(
                    f"[emotion_detector] Local emotion model: {prediction.label} "
                    f"({prediction.confidence:.2f})"
                )
                return prediction.label
        except Exception as exc:
            print(f"[emotion_detector] Local emotion model error: {exc}")

    if USE_HUME and HUME_API_KEY and HUME_SDK_AVAILABLE:
        try:
            emotion = _analyze_via_hume(frame_data)
            if emotion:
                return emotion
            if not _LAST_ERROR:
                _LAST_ERROR = "Hume could not determine an emotion from this frame."
            print(f"[emotion_detector] Hume failed: {_LAST_ERROR}")
        except Exception as exc:
            _LAST_ERROR = f"Hume error: {exc}"
            print(f"[emotion_detector] {_LAST_ERROR}")

        # If Hume is unavailable or discontinued, fall back so the app still works.
        fallback_emotion = _opencv_detector(frame_data)
        if fallback_emotion:
            _set_error(
                "Hume emotion detection is unavailable, so the app used the local OpenCV fallback. "
                f"Last Hume error: {_LAST_ERROR}"
            )
            return fallback_emotion

        return None

    return _opencv_detector(frame_data)


def _analyze_via_hume(frame_data: np.ndarray) -> Optional[str]:
    frame_bytes = _convert_to_jpeg_bytes(frame_data)
    if not frame_bytes:
        _set_error("Could not encode frame as JPEG.")
        return None

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_async_in_thread, frame_bytes)
        try:
            return future.result(timeout=20)
        except Exception as exc:
            _set_error(f"Hume request timed out: {exc}")
            return None


def _run_async_in_thread(frame_bytes: bytes) -> Optional[str]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_async_analyze_hume(frame_bytes))
    finally:
        loop.close()


async def _async_analyze_hume(frame_bytes: bytes) -> Optional[str]:
    stream_result = await _async_analyze_stream_sdk(frame_bytes)
    if stream_result:
        return stream_result

    return await _async_analyze_batch_sdk(frame_bytes)


async def _async_analyze_stream_sdk(frame_bytes: bytes) -> Optional[str]:
    from hume import AsyncHumeClient
    from hume.expression_measurement.stream import Config

    import base64

    frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
    config = Config(face={})

    try:
        client = AsyncHumeClient(api_key=HUME_API_KEY)
        async with client.expression_measurement.stream.connect() as socket:
            result = await socket.send_file(frame_b64, config=config)
            mood = _parse_sdk_result(result)
            if mood:
                return mood
    except Exception as exc:
        import traceback

        message = str(exc)
        tb = traceback.format_exc()
        # Provide actionable hint for common Windows AV/driver permission issues
        if "403" in message or "401" in message:
            _set_error(
                "Hume API key was rejected (HTTP 403). "
                "Use a valid Personal API key from https://platform.hume.ai/settings/keys"
            )
        elif "Permission denied" in message or "aswMonFltProxy" in message:
            _set_error(
                "Hume streaming failed: Permission denied. "
                "This often indicates an antivirus or network filter (for example, Avast) is blocking websocket/network calls to Python. "
                "Try adding an exception for Python or disabling the web shield, or run the app with elevated privileges. "
                f"Full error: {message}"
            )
        elif "discontinued" in message.lower() or "no longer available" in message.lower():
            _set_error(
                "Hume Expression Measurement API is discontinued and no longer available. "
                "The app will fall back to local OpenCV detection unless you migrate to a current Hume API. "
                f"Full error: {message}"
            )
        else:
            _set_error(f"Hume streaming failed: {message}")

        print(f"[emotion_detector] Stream SDK error: {message}\n{tb}")

    # Raw websocket fallback for older connection stacks.
    mood = await _async_analyze_raw_websocket(frame_b64)
    return mood


async def _async_analyze_raw_websocket(frame_b64: str) -> Optional[str]:
    import websockets

    payload = json.dumps(
        {"data": frame_b64, "models": {"face": {}}, "raw_text": False}
    )
    headers = {"X-Hume-Api-Key": HUME_API_KEY}
    uris = (
        "wss://api.hume.ai/v0/stream/models",
        f"wss://api.hume.ai/v0/stream/models?api_key={HUME_API_KEY}",
    )

    for uri in uris:
        try:
            async with websockets.connect(uri, additional_headers=headers) as ws:
                await ws.send(payload)
                async for raw in ws:
                    mood = _parse_hume_response(raw)
                    if mood:
                        return mood
                    break
        except Exception as exc:
            print(f"[emotion_detector] Raw websocket error ({uri}): {exc}")
            continue

    return None


async def _async_analyze_batch_sdk(frame_bytes: bytes) -> Optional[str]:
    from hume import AsyncHumeClient
    from hume.expression_measurement.batch import Models

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(frame_bytes)
            temp_path = handle.name

        client = AsyncHumeClient(api_key=HUME_API_KEY)
        models = Models(face={})
        job_id = await client.expression_measurement.batch.start_inference_job_from_local_file(
            file=[temp_path],
            json=models,
        )

        for _ in range(20):
            await asyncio.sleep(0.5)
            job = await client.expression_measurement.batch.get_job_details(id=job_id)
            status = job.state.status if job.state else None
            if status == "COMPLETED":
                predictions = await client.expression_measurement.batch.get_job_predictions(
                    id=job_id
                )
                return _parse_batch_predictions(predictions)
            if status == "FAILED":
                _set_error("Hume batch job failed.")
                return None
    except Exception as exc:
        message = str(exc)
        if "discontinued" in message.lower() or "no longer available" in message.lower():
            _set_error(
                "Hume Expression Measurement API is discontinued and no longer available. "
                "The app will fall back to local OpenCV detection unless you migrate to a current Hume API."
            )
        else:
            _set_error(f"Hume batch SDK error: {message}")
        print(f"[emotion_detector] Batch SDK error: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    return None


def _parse_sdk_result(result: Any) -> Optional[str]:
    if result is None:
        return None

    error = getattr(result, "error", None) or getattr(result, "message", None)
    if error and not getattr(result, "face", None):
        _set_error(f"Hume API error: {error}")
        return None

    face = getattr(result, "face", None)
    if not face or not face.predictions:
        return None

    emotions_raw = [
        {"name": emotion.name, "score": emotion.score}
        for emotion in face.predictions[0].emotions
    ]
    return _pick_mood_from_emotions(emotions_raw)


def _parse_batch_predictions(predictions: Any) -> Optional[str]:
    try:
        if not predictions:
            return None

        first_result = predictions[0]
        results = getattr(first_result, "results", None) or first_result.get("results", {})
        predictions_list = getattr(results, "predictions", None) or results.get(
            "predictions", []
        )
        if not predictions_list:
            return None

        first_prediction = predictions_list[0]
        models = getattr(first_prediction, "models", None) or first_prediction.get(
            "models", {}
        )
        face_data = getattr(models, "face", None) or models.get("face", {})
        grouped = getattr(face_data, "grouped_predictions", None) or face_data.get(
            "grouped_predictions", []
        )
        if not grouped:
            return None

        face_predictions = getattr(grouped[0], "predictions", None) or grouped[0].get(
            "predictions", []
        )
        if not face_predictions:
            return None

        emotions = getattr(face_predictions[0], "emotions", None) or face_predictions[
            0
        ].get("emotions", [])
        emotions_raw = [
            {
                "name": getattr(emotion, "name", emotion.get("name")),
                "score": getattr(emotion, "score", emotion.get("score", 0)),
            }
            for emotion in emotions
        ]
        return _pick_mood_from_emotions(emotions_raw)
    except Exception as exc:
        print(f"[emotion_detector] Batch parse error: {exc}")
        return None


def _parse_hume_response(raw: str) -> Optional[str]:
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if "error" in result:
        _set_error(f"Hume API error: {result['error']}")
        return None

    face_block = result.get("face") or {}
    predictions = face_block.get("predictions") or []
    if not predictions:
        return None

    emotions_raw = predictions[0].get("emotions") or []
    if not emotions_raw:
        return None

    return _pick_mood_from_emotions(emotions_raw)


def _pick_mood_from_emotions(emotions_raw: List[Dict[str, Any]]) -> Optional[str]:
    """
    Cluster all Hume emotion scores by mapped mood and pick the strongest cluster.

    Sad expressions often score higher on Disappointment / Distress than on Sadness,
    while Calmness can still register on a neutral baseline.
    """
    all_sorted = sorted(emotions_raw, key=lambda e: e["score"], reverse=True)

    print("[emotion_detector] Hume top 5:")
    for i, emotion in enumerate(all_sorted[:5]):
        mark = "*" if emotion["name"] in MAIN_EMOTIONS else " "
        print(f"  {mark} {i+1}. {emotion['name']}: {emotion['score']:.3f}")

    mood_scores: Dict[str, float] = {}
    for emotion in emotions_raw:
        mood = _map_hume_emotion_to_mood(emotion["name"])
        if not mood:
            continue
        mood_scores[mood] = mood_scores.get(mood, 0.0) + float(emotion["score"])

    if mood_scores:
        best_mood, best_score = max(mood_scores.items(), key=lambda item: item[1])
        if best_score >= HUME_PROB_THRESHOLD:
            contributors = [
                f"{emotion['name']}({emotion['score']:.2f})"
                for emotion in all_sorted
                if _map_hume_emotion_to_mood(emotion["name"]) == best_mood
            ][:3]
            print(
                f"[emotion_detector] Hume cluster: {best_mood} "
                f"(score={best_score:.2f}, from {', '.join(contributors)})"
            )
            return best_mood

    if all_sorted:
        top = all_sorted[0]
        mapped = _map_hume_emotion_to_mood(top["name"])
        if mapped:
            print(
                f"[emotion_detector] Hume top emotion: {top['name']} "
                f"({top['score']:.2f}) -> {mapped}"
            )
            return mapped

    return None


def _opencv_detector(frame_data: np.ndarray) -> Optional[str]:
    """Haar-cascade fallback only when Hume is disabled."""
    try:
        gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            _set_error("No face detected in the image.")
            return None

        x, y, w, h = faces[0]
        face_roi_gray = gray[y : y + h, x : x + w]
        smiles = smile_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.7, minNeighbors=15, minSize=(25, 25)
        )

        if len(smiles) > 0:
            print("[emotion_detector] OpenCV: smile -> happy")
            return "happy"

        _set_error("OpenCV cannot reliably detect sadness. Enable Hume AI or use manual input.")
        return None
    except Exception as exc:
        _set_error(f"OpenCV error: {exc}")
        return None


def _convert_to_jpeg_bytes(frame_data: np.ndarray, quality: int = 85) -> Optional[bytes]:
    try:
        frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        if img.width > 1024 or img.height > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except Exception as exc:
        _set_error(f"Frame conversion error: {exc}")
        return None


def _map_hume_emotion_to_mood(hume_emotion: str) -> Optional[str]:
    primary = {
        "Joy": "happy",
        "Sadness": "sad",
        "Anger": "angry",
        "Fear": "fearful",
        "Surprise (positive)": "surprised",
        "Surprise (negative)": "surprised",
        "Disgust": "angry",
        "Calmness": "calm",
        "Excitement": "energized",
        "Contentment": "relaxed",
    }
    if hume_emotion in primary:
        return primary[hume_emotion]

    extended = {
        "Amusement": "happy",
        "Satisfaction": "happy",
        "Triumph": "happy",
        "Pride": "happy",
        "Relief": "happy",
        "Gratitude": "happy",
        "Admiration": "happy",
        "Adoration": "happy",
        "Love": "loving",
        "Romance": "loving",
        "Disappointment": "sad",
        "Empathic Pain": "sad",
        "Sympathy": "sad",
        "Tiredness": "sad",
        "Boredom": "sad",
        "Guilt": "sad",
        "Shame": "sad",
        "Embarrassment": "sad",
        "Pain": "sad",
        "Distress": "sad",
        "Contempt": "angry",
        "Annoyance": "angry",
        "Envy": "angry",
        "Anxiety": "anxious",
        "Horror": "fearful",
        "Doubt": "fearful",
        "Confusion": "fearful",
        "Awkwardness": "fearful",
        "Concentration": "focused",
        "Contemplation": "focused",
        "Determination": "focused",
        "Interest": "focused",
        "Enthusiasm": "energized",
        "Craving": "energized",
        "Realization": "surprised",
        "Awe": "surprised",
        "Nostalgia": "relaxed",
        "Entrancement": "focused",
    }
    return extended.get(hume_emotion)


def _set_error(message: str) -> None:
    global _LAST_ERROR
    _LAST_ERROR = message


def normalize_emotion(emotion: str) -> str:
    if not emotion:
        return "calm"
    value = emotion.lower().strip()
    valid = {
        "happy",
        "sad",
        "angry",
        "fearful",
        "surprised",
        "calm",
        "energized",
        "relaxed",
        "focused",
        "loving",
        "anxious",
    }
    if value in valid:
        return value
    aliases = {
        "neutral": "calm",
        "content": "calm",
        "peaceful": "calm",
        "surprise": "surprised",
        "excited": "energized",
        "hyper": "energized",
        "tired": "relaxed",
        "sleepy": "relaxed",
        "scared": "fearful",
        "afraid": "fearful",
        "worried": "anxious",
        "mad": "angry",
        "frustrated": "angry",
        "annoyed": "angry",
        "joyful": "happy",
        "cheerful": "happy",
        "glad": "happy",
        "depressed": "sad",
        "unhappy": "sad",
        "down": "sad",
        "amazed": "surprised",
        "shocked": "surprised",
        "astonished": "surprised",
        "romantic": "loving",
    }
    return aliases.get(value, "calm")
