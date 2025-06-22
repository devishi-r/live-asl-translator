from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import psycopg2
import faiss
import numpy as np
import json
import mediapipe as mp  
import cv2
import os

load_dotenv()

conn = psycopg2.connect(
    database="postgres",
    host="localhost",
    user="postgres",
    port=5432,
    password=os.getenv('POSTGRES_PASSWORD')
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

word_list = json.load(open("words.json"))  # The words in your FAISS index
embedding_matrix = np.load("word_embeddings.npy")
faiss_index = faiss.read_index("word_index.faiss")

def get_landmark_points_faiss_aware(word, cursor):
    # direct lookup
    cursor.execute("SELECT points FROM signs WHERE word = %s", (word,))
    result = cursor.fetchone()
    if result:
        return result[0], word  # success

    # Fallback: FAISS 
    vec = embedding_model.encode([word]).astype("float32")
    D, I = faiss_index.search(vec, k=1)
    matched_word = word_list[I[0][0]]

    cursor.execute("SELECT points FROM signs WHERE word = %s", (matched_word,))
    result = cursor.fetchone()
    if result:
        print(f"[FAISS fallback] '{word}' not found. Using closest match: '{matched_word}' (distance: {D[0][0]:.4f})")
        return result[0], matched_word

    print(f"[FAISS fallback] '{word}' and its nearest neighbor '{matched_word}' both failed.")
    return None, None

#  helper function to parse the raw points (from your points column) into per-frame dicts of landmarks
def parse_frames(raw_frames):
    parsed_frames = []
    for frame_data in raw_frames:
        # frame_data example: [frame_number, pose_landmarks, hand_landmarks]
        frame_number, pose_landmarks_raw, hand_landmarks_raw = frame_data

        # Convert landmarks into dicts for pose and hands
        pose_landmarks = {}
        if pose_landmarks_raw:
            for landmark in pose_landmarks_raw:
                id, x, y, z = landmark
                pose_landmarks[id] = (x, y, z)

        hand_landmarks = []
        if hand_landmarks_raw:
            for hand in hand_landmarks_raw:
                hand_dict = {}
                for landmark in hand:
                    id, x, y, z = landmark
                    hand_dict[id] = (x, y, z)
                hand_landmarks.append(hand_dict)

        parsed_frames.append({
            "frame_number": frame_number,
            "pose": pose_landmarks,
            "hands": hand_landmarks
        })
    return parsed_frames


# interpolation function between two frames (same structure), to generate intermediate frames for smooth transition

def interpolate_landmarks(lm1, lm2, steps=5):
    # lm1 and lm2: dicts {id: (x,y,z)}, interpolate each coordinate
    interpolated_frames = []
    for step in range(1, steps + 1):
        t = step / (steps + 1)
        interp_frame = {}
        for key in lm1:
            if key in lm2:
                x1, y1, z1 = lm1[key]
                x2, y2, z2 = lm2[key]
                x = x1 * (1 - t) + x2 * t
                y = y1 * (1 - t) + y2 * t
                z = z1 * (1 - t) + z2 * t
                interp_frame[key] = (x, y, z)
        interpolated_frames.append(interp_frame)
    return interpolated_frames


# Interpolate between two frames that have pose and hands
def interpolate_frames(frame1, frame2, steps=5):
    interp_frames = []
    pose_interp = interpolate_landmarks(frame1["pose"], frame2["pose"], steps)
    
    # For hands, assume same number of hands (usually 2)
    hands_interp = []
    for h_idx in range(len(frame1["hands"])):
        interp_hands = interpolate_landmarks(
            frame1["hands"][h_idx], frame2["hands"][h_idx], steps
        )
        hands_interp.append(interp_hands)
    
    # Now, combine pose + hands for each intermediate frame
    for i in range(steps):
        combined_hands = []
        for h in hands_interp:
            combined_hands.append(h[i])
        interp_frames.append({
            "frame_number": None,
            "pose": pose_interp[i],
            "hands": combined_hands
        })
    return interp_frames


def on_connect(sentence):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    words = [w.strip() for w in sentence.split()]

    cursor = conn.cursor()
    animations = []

    # Fetch all word animations
    for word in words:
        points, matched_word = get_landmark_points_faiss_aware(word, conn.cursor())
        if points:
            parsed = parse_frames(points)
            for frame in parsed:
                frame["word"] = matched_word  # attach the word to each frame
            animations.append(parsed)

    # Now concatenate all word animations with interpolation between words
    full_animation = []
    for i in range(len(animations)):
        full_animation.extend(animations[i])
        if i < len(animations) - 1:
            interp = interpolate_frames(animations[i][-1], animations[i + 1][0], steps=5)
            for frame in interp:
                frame["word"] = ""  # optional: blank or "transition"
            full_animation.extend(interp)

    # Setup OpenCV window and writer
    frame_width, frame_height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

    pose_connections = mp_pose.POSE_CONNECTIONS
    hand_connections = mp_hands.HAND_CONNECTIONS

    # Render frames
    for frame_data in full_animation:
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        # Display current word
        word_text = frame_data.get("word", "")
        cv2.putText(frame, f"{word_text}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)


        # Draw pose landmarks
        pose_landmarks = frame_data["pose"]
        pose_landmarks_dict = {}
        for id, (x, y, z) in pose_landmarks.items():
            pose_landmarks_dict[id] = (int(x), int(y))
            if 0 <= x < frame_width and 0 <= y < frame_height:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        for connection in pose_connections:
            if connection[0] in pose_landmarks_dict and connection[1] in pose_landmarks_dict:
                start = pose_landmarks_dict[connection[0]]
                end = pose_landmarks_dict[connection[1]]
                cv2.line(frame, start, end, (0, 255, 0), 2)

        # Draw hand landmarks
        for hand_landmarks in frame_data["hands"]:
            hand_landmarks_dict = {}
            for id, (x, y, z) in hand_landmarks.items():
                hand_landmarks_dict[id] = (int(x), int(y))
                if 0 <= x < frame_width and 0 <= y < frame_height:
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
            for connection in hand_connections:
                if connection[0] in hand_landmarks_dict and connection[1] in hand_landmarks_dict:
                    start = hand_landmarks_dict[connection[0]]
                    end = hand_landmarks_dict[connection[1]]
                    cv2.line(frame, start, end, (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow('ASL Animation', frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    cursor.close()

on_connect("jacket tail suit tree")