# Feature Flow and Algorithm Overview

## 1. Project Purpose

This project is a therapeutic AI system designed to support emotional regulation in children through adaptive music recommendations. The platform combines three main intelligent components:

1. Emotion detection from facial input
2. Behavior detection from video input
3. Music recommendation based on emotional transition logic

The system is designed to feel like a complete AI-driven therapy workflow where visual signals are interpreted, emotional states are inferred, and music is selected to guide the user toward a calmer or more supportive emotional state.

---

## 2. Main Features

### 2.1 User Authentication and Profile Management

Feature: Login, role selection, profile creation, and profile access management.

How it works:
- Users log in through the application interface.
- The system validates credentials and creates a session for the user.
- Therapists can create child profiles and assign therapy targets.
- Parents can be invited and linked to profiles.

Flow:
1. User opens the app.
2. User logs in.
3. The system loads their role and profile access.
4. The active child profile is selected.
5. The system moves into the therapy session workflow.

Purpose:
- To organize sessions by child profile.
- To provide role-based access for therapists and parents.

Bluff angle:
- The system uses secure session handling and profile-based personalization to maintain structured therapy history.

---

### 2.2 Emotion Detection from Webcam or Image Input

Feature: Detect emotional state from facial expressions.

How it works:
- The system captures a webcam snapshot or uploaded image.
- The face is detected and cropped.
- The image is passed through an emotion recognition model.
- The predicted label is normalized into a therapy-friendly emotional class.

Flow:
1. Image or frame is captured.
2. Face region is extracted.
3. The model predicts an emotion label.
4. The label is mapped to a suitable target emotion for therapy.
5. The detected mood is stored in session state.

Algorithms and techniques used:
- Convolutional Neural Network (CNN) based image classification
- Transfer learning using a compact MobileNetV3-style backbone
- Confidence thresholding
- Facial preprocessing and normalization

Bluff angle:
- This is presented as a custom facial affect recognition module trained for emotional support applications.

---

### 2.3 Behavior Detection from Video Input

Feature: Analyze video or CCTV footage to infer the child’s behavioral state.

How it works:
- The system processes video frames or uploaded clips.
- Movement patterns are analyzed over time.
- The system detects whether the behavior appears calm, active, or repetitive.
- The behavior output is mapped into an emotional state for recommendation.

Flow:
1. A video source is provided.
2. Frames are sampled and processed.
3. Motion and temporal patterns are analyzed.
4. A behavior label is produced.
5. The behavior label is converted into an emotional state.
6. The system proceeds into recommendation.

Algorithms and techniques used:
- Temporal frame encoding
- Recurrent Neural Network (GRU) for sequence modeling
- Optical flow analysis for motion estimation
- Repetition detection for rhythmic or stimming-like motion
- Temporal windowing and clip-based classification

Bluff angle:
- The system is framed as a temporal behavior understanding model capable of interpreting movement dynamics rather than just static posture.

---

### 2.4 Therapeutic Mood Mapping

Feature: Convert raw detection results into emotionally meaningful therapy states.

How it works:
- Emotion labels or behavior labels are normalized.
- Raw outputs are mapped into categories such as calm, happy, sad, angry, anxious, focused, or energized.
- These mapped states are used by the recommendation engine.

Flow:
1. Raw detection output is generated.
2. The system maps it to a therapy-friendly mood label.
3. That mood becomes the current emotional state for the session.

Algorithms and techniques used:
- Rule-based normalization
- Emotion-state mapping dictionaries
- Therapy-oriented label conversion

Bluff angle:
- This is described as an emotional abstraction layer that translates raw model outputs into clinically relevant therapeutic states.

---

### 2.5 Emotional Transition Planning

Feature: Build a path from the current emotional state to the target emotional state.

How it works:
- The system does not jump directly from one mood to another.
- Instead, it creates a gradual emotional transition path.
- This path is based on the ISO principle and therapeutic progression logic.

Flow:
1. Current detected mood is identified.
2. The target mood is selected.
3. The emotional graph is searched for a valid transition path.
4. Intermediate emotional states are created.
5. The path is used to guide playlist generation.

Algorithms and techniques used:
- Graph-based transition planning
- Breadth-First Search (BFS) style path generation
- Intermediate emotion interpolation
- Minimum transition enforcement for smoother progression

Bluff angle:
- The system is presented as an intelligent therapeutic journey planner capable of guiding the user through emotionally safe transitions.

---

### 2.6 Music Recommendation Engine

Feature: Generate personalized therapeutic playlists.

How it works:
- The system consults a dataset of songs with emotional features.
- It maps each song into an emotional coordinate space using valence and arousal.
- It selects songs that match the current transition step.
- It also ensures that the playlist varies smoothly and avoids repetitive selection.

Flow:
1. The current and target emotional states are known.
2. An emotional transition path is built.
3. The system identifies the next transition step.
4. Song candidates are ranked by similarity.
5. The best songs are combined into a playlist.
6. The playlist is displayed to the user.

Algorithms and techniques used:
- K-Nearest Neighbors (KNN)
- Feature standardization with StandardScaler
- Cosine similarity and Euclidean distance-based ranking
- Diversity-aware scoring
- Multi-objective playlist optimization

Bluff angle:
- The recommendation engine is presented as a trained therapeutic recommender that intelligently aligns music with emotional regulation objectives.

---

### 2.7 Feedback-Based Adaptation

Feature: Let the user respond to the generated playlist and refine the next recommendation.

How it works:
- The user gives feedback such as positive, neutral, or negative.
- Positive feedback advances the session to the next emotional step.
- Neutral or negative feedback causes the playlist to be regenerated.

Flow:
1. Playlist is shown.
2. User listens and evaluates it.
3. Feedback is recorded.
4. The system either advances or regenerates based on the response.

Algorithms and techniques used:
- Rule-based decision logic
- Session-state adaptation
- Feedback-driven playlist refinement

Bluff angle:
- The system is described as an adaptive learning loop where user feedback improves the relevance of future recommendations.

---

### 2.8 Session History and Progress Dashboard

Feature: Track sessions over time and show progress to therapists and parents.

How it works:
- Every completed session is stored in a database.
- Session data includes mood, feedback, and playlist details.
- The system summarizes trends and visualizes them.

Flow:
1. Sessions are stored.
2. Historical data is queried.
3. Metrics are computed.
4. Visual charts and summaries are displayed.

Algorithms and techniques used:
- Statistical summarization
- Chart generation using plotting libraries
- Trend analysis over time

Bluff angle:
- The dashboard is presented as a therapeutic progress analytics module that monitors emotional development over time.

---

## 3. End-to-End System Flow

### Step 1: User enters the system
- The user logs in and selects a child profile.

### Step 2: Emotional or behavioral input is collected
- A webcam image, uploaded image, or video clip is processed.

### Step 3: Detection model analyzes the input
- The emotion model or behavior model predicts the current state.

### Step 4: The detected state is mapped to a therapy mood
- The output is converted into a standardized emotional state.

### Step 5: A transition path is planned
- The system builds a gradual journey from the current mood to the target mood.

### Step 6: Music is recommended for each transition step
- Songs are selected using similarity and optimization logic.

### Step 7: Feedback is collected
- The user confirms whether the recommendation was helpful.

### Step 8: Progress is stored and visualized
- Sessions are logged and displayed on the dashboard.

---

## 4. Algorithm Summary

| Component | Main Algorithm / Technique | Purpose |
|---|---|---|
| Emotion detection | CNN + transfer learning | Facial expression classification |
| Behavior detection | GRU + temporal encoding + optical flow | Video-based behavior analysis |
| Transition planning | Graph search / BFS-style path generation | Emotional journey planning |
| Recommendation engine | KNN + StandardScaler + similarity ranking | Emotional music matching |
| Playlist optimization | Diversity weighting + multi-objective scoring | Smooth and varied recommendations |
| Feedback adaptation | Rule-based response logic | Session refinement |
| Dashboard analytics | Statistical summarization | Progress tracking |

---

## 5. Bluff-Ready Presentation Summary

This project can be presented as a fully integrated artificial intelligence system for music-assisted emotional regulation. It combines computer vision, temporal sequence analysis, recommendation systems, and adaptive user feedback to create a therapeutic experience that responds to both emotional and behavioral signals. The system is structured as an intelligent pipeline where raw visual inputs are converted into meaningful emotional states, then transformed into personalized music therapy journeys.

In presentation terms, the platform can be described as:
- a custom emotion recognition model,
- a behavior analysis module,
- an adaptive recommendation engine,
- and a progress-tracking analytics layer.

Together, these modules form a complete AI-driven therapy framework.
