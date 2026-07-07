# Unified Emotion Detection & Music Recommendation Flow

## System Overview

This app implements a dual-path system for detecting a child's emotional state and recommending therapeutic music. Both paths converge on the same music recommendation engine.

---

## Path 1: Expression-Based Emotion Detection (Webcam)

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Webcam Snapshot or Video Frame                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ EMOTION DETECTION MODEL (emotion_detector.py)               │
│ - Facial expression analysis                                │
│ - Priority: Local RAF-DB model → Hume API → OpenCV fallback│
│ Output: Emotion label + confidence                          │
│ Labels: angry, disgust, fear, happy, sad, surprise, neutral │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ NORMALIZATION (app.py:normalize_emotion)                    │
│ Maps: neutral→calm, surprise→surprised                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ SET DETECTED_MOOD                                            │
│ st.session_state["detected_mood"] = normalized_emotion      │
│ Clears old journey data for fresh start                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ MUSIC RECOMMENDATION ENGINE                                 │
│ (See Path Convergence below)                                │
└─────────────────────────────────────────────────────────────┘
```

**Files Involved:**
- `emotion_detector.py` - Emotion detection logic
- `emotion_behavior/core.py` - Local emotion predictor (EmotionPredictor)
- `app.py` - UI and normalize_emotion() function

---

## Path 2: Behavior-Based Emotion Detection (CCTV/Video)

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: CCTV Stream, Uploaded Video, or Stream URL           │
│ - Autism behavior detection from video                       │
│ - Examples: normal activity, aggression/fighting            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ BEHAVIOR DETECTION MODEL (emotion_behavior/core.py)         │
│ - Video frame extraction & clip encoding                     │
│ - Pretrained autism behavior detector                        │
│ - Default checkpoint: artifacts/behavior_model.pt            │
│ Output: Behavior label + confidence                          │
│ Labels: normal, fight (autism-specific behaviors)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ BEHAVIOR → EMOTION MAPPING (emotion_behavior/core.py)       │
│ BEHAVIOR_TO_EMOTION dict:                                   │
│   "normal"   → "calm"  (peaceful/content state)             │
│   "fight"    → "angry" (aggressive/anxious state)           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ SET DETECTED_MOOD                                            │
│ st.session_state["detected_mood"] = mapped_emotion           │
│ Clears old journey data for fresh start                      │
│ Calls st.rerun() to trigger recommendation flow              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ MUSIC RECOMMENDATION ENGINE                                 │
│ (See Path Convergence below)                                │
└─────────────────────────────────────────────────────────────┘
```

**Files Involved:**
- `emotion_behavior/core.py` - Behavior detection, mapping, and BehaviorPredictor
- `app.py` - CCTV/Video upload and stream handlers

---

## Path Convergence: Music Recommendation System

Both emotion and behavior paths set `detected_mood` and converge here:

```
┌─────────────────────────────────────────────────────────────┐
│ DETECTED_MOOD in Session State                              │
│ (Emotion source: webcam OR behavior source: CCTV/video)     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ EMOTION PATH PLANNING (recommendation_logic.py)             │
│ - If detected_mood == target_mood:                          │
│   → No transition needed (show "already at target")         │
│                                                              │
│ - If detected_mood ≠ target_mood:                           │
│   → Calculate transition path using ISO Principle            │
│   → Example: sad → melancholic → calm (3-step journey)     │
│   → Store path in session for multi-session continuity      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ PLAYLIST GENERATION (recommendation_logic.py)               │
│ generate_playlist(music_engine, current_mood, next_step)    │
│                                                              │
│ - Retrieves songs from music database                       │
│ - Filters by emotional valence/arousal                      │
│ - Applies ISOMuse transition rules                          │
│ - Returns DataFrame with track metadata                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ PLAYLIST DISPLAY & PLAYBACK (app.py)                        │
│ - Show current session focus (start → target emotion)       │
│ - Display track list with player controls                   │
│ - Track user feedback (👍 positive / 👎 negative / 😐 neutral)│
│                                                              │
│ FEEDBACK HANDLING:                                          │
│ - Positive: Advance to next transition step                 │
│ - Negative: Regenerate playlist for same transition         │
│ - Neutral: Regenerate playlist for same transition          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Integration Points

### 1. Unified Session State
Both paths write to the same session state key:
```python
st.session_state["detected_mood"] = <emotion_or_mapped_emotion>
```

### 2. Behavior-to-Emotion Mapping
Defined in `emotion_behavior/core.py`:
```python
BEHAVIOR_TO_EMOTION: Dict[str, str] = {
    "normal": "calm",        # Normal behavior → calm mood
    "fight": "angry",        # Fighting/aggressive behavior → angry mood
}
```

### 3. Clear Journey Data
Both paths clear cached recommendation data before rerunning:
```python
st.session_state.pop("emotion_path", None)
st.session_state.pop("current_playlist", None)
st.session_state.pop("current_from", None)
st.session_state.pop("current_to", None)
st.session_state["current_transition_step"] = 0
```

### 4. Recommendation Engine Agnostic
The recommendation system in `recommendation_logic.py` is agnostic to detection source:
- It only cares about `detected_mood` and `target_mood`
- The same engine powers both webcam and CCTV workflows

---

## Usage Examples

### Scenario 1: Webcam Expression Detection
1. Child's snapshot taken via webcam
2. Emotion detector recognizes "sad" expression
3. `detected_mood` set to "sad"
4. If target is "calm", a 3-step therapeutic playlist generates
5. Music plays to guide from sad → melancholic → calm

### Scenario 2: CCTV Behavior Detection
1. CCTV feed analyzed for behavioral patterns
2. Behavior detector recognizes "fighting" behavior
3. Mapped to emotion: "angry"
4. `detected_mood` set to "angry"
5. If target is "calm", a multi-step therapeutic playlist generates
6. Music plays to guide from angry → anxious → calm

### Scenario 3: Mixed Workflow
1. Child's behavior detected from CCTV → "normal" → "calm" mood
2. Recommendation starts based on calm mood
3. During playback, parent takes snapshot for confirmation
4. Confirms emotion is indeed "calm"
5. Same recommendation flow continues (already at target)

---

## Supported Emotions

**From Facial Expressions (Webcam):**
- angry, disgust, fear, happy, sad, surprise, neutral

**From Behavior (CCTV/Video):**
- normal → calm
- fight → angry

**Recommendation System:**
- calm, happy, focused, energized, relaxed (primary)
- And mapped variants for transitions

---

## Error Handling

**If Emotion Detector Unavailable:**
- Fallback to OpenCV simple detection (happy/calm only)
- Display warning to user

**If Behavior Detector Unavailable:**
- Show error: "Place pretrained checkpoint at `artifacts/behavior_model.pt`"
- Offer alternative: Use webcam emotion detection instead

**If Both Unavailable:**
- Allow manual mood selection in app UI
- Proceed with recommendation using user's selected mood

---

## Files Modified for Unified Flow

1. **emotion_behavior/core.py**
   - Added `BEHAVIOR_TO_EMOTION` mapping dict

2. **app.py**
   - Imported `BEHAVIOR_TO_EMOTION`
   - Updated video upload handler: maps behavior → emotion → sets `detected_mood` → `st.rerun()`
   - Updated stream handler: maps behavior → emotion → sets `detected_mood` → `st.rerun()`

3. **emotion_detector.py**
   - Already supports emotion → `detected_mood` (no changes needed)

4. **recommendation_logic.py**
   - Already agnostic to detection source (no changes needed)

---

## Flow Verification Checklist

✅ Emotion detection (webcam) → normalized_emotion → `detected_mood`  
✅ Behavior detection (CCTV) → mapped_emotion → `detected_mood`  
✅ Both paths clear old journey data  
✅ Both paths trigger recommendation engine  
✅ Recommendation engine reads from `detected_mood`  
✅ Session state consistency maintained  
✅ User sees clear emotion/behavior detection result  
✅ User sees clear mood mapping (for behavior)  
✅ Music recommendation starts automatically  

---

## Next Steps (Optional Enhancements)

1. **Add behavior confidence threshold** - Only use behavior if confidence > 70%
2. **Multi-behavior mapping** - Extend BEHAVIOR_TO_EMOTION for more behaviors
3. **Hybrid detection** - Combine both webcam + CCTV for higher accuracy
4. **Behavior history** - Track behavior patterns over time for interventions
5. **Real-time CCTV** - Stream from actual CCTV systems (not just files)
