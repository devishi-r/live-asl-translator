# # from sentence_transformers import SentenceTransformer

# # try:
# #     model = SentenceTransformer("all-MiniLM-L6-v2")
# #     print("Model loaded successfully")
# # except Exception as e:
# #     print(f"Error loading model: {e}")

# # import ssl
# # print(ssl.get_default_verify_paths())


from sentence_transformers import SentenceTransformer
import psycopg2
conn = psycopg2.connect(
    database="postgres",
    host="localhost",
    user="postgres",
    port=5432,
    password="passw0rd"
)
cur = conn.cursor()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# def on_request_animation(words: str):
#     """Triggered when client requests an expressive animation for a word or sentence"""

#     animations = []
#     words = words.strip()

#     if not words:
#         return

#     # Gloss the words
#     # words = llm.gloss(words)
#     # words = words.split()

#     cursor = conn.cursor()
#     for word in words:
#         word = word.strip()
#         if not word:
#             continue

#         embedding = embedding_model.encode(word)
#         cursor.execute(
#             "SELECT word, points, (embedding <=> %s) AS cosine_similarity FROM signs ORDER BY cosine_similarity ASC LIMIT 1",
#             (embedding,),
#         )
#         result = cursor.fetchone()

#         # Add sign to animation
#         if result and 1 - result[2] > 0.70:
#             animations.append((word, result[1]))
#         else:  # Add fingerspell to animation
#             animation = []
#             for letter in word:
#                 animation.extend(alphabet_frames.get(letter.upper(), []))

#             for i in range(len(animation)):
#                 animation[i][0] = i
#             animations.append((f"fs-{word.upper()}", animation))

#         if "." in word:
#             space = []
#             last_frame = animations[-1][1][-1]
#             for i in range(50):
#                 space.append(last_frame)
#                 space[-1][0] = i
#             animations.append(("", space))

#     emit("E-ANIMATION", animations)
#     cursor.close()


import mediapipe as mp  
import cv2
import numpy as np
import json


def on_connect(sentence):

    # print("Connected to client")
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    words = sentence.split(" ")
    for i in words:
        i.strip()

    # Send hello sign
    cursor = conn.cursor()
    animations = []
    # embedding = embedding_model.encode("apple").tolist()
    for word in words:
        cursor.execute( '''
                    SELECT word, points, embedding FROM signs WHERE word = %s
                        ORDER BY embedding ASC LIMIT 1''',(word,)
            # ,(json.dumps(embedding),)
        )
        result = cursor.fetchone()
        print(result[1][0][0])
        print("Length of result[1] is: ",len(result[1]))

        # for i in result[1]:
        #     print("Length: ", len(i))
        #     print("Type: ", type(i))
        #     for j in i:
        #         print("Nested type: ", type(j))

        if result:
            animations.append(("APPLE", result[1]))
            points_data = (result[1])

        # Create a blank video
            frame_width, frame_height = 640, 480
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

            pose_connections = mp.solutions.pose.POSE_CONNECTIONS
            hand_connections = mp.solutions.hands.HAND_CONNECTIONS

            for frame_data in points_data:
                frame_number, pose_landmarks, hand_landmarks = frame_data
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            # Draw pose landmarks
                if pose_landmarks:
                    pose_landmarks_dict={}
                    for landmark in pose_landmarks:
                        landmark_number, x, y, z = landmark
                        print("(x, y): (For pose)= (",x,", ", y,")")
                        # pose_landmarks_list.append(mp_pose.PoseLandmark(x/frame_width, y/frame_height, z/frame_width))
                        # mp_drawing.draw_landmarks(frame, mp_pose.PoseLandmark(landmark_list=pose_landmarks_list), mp_pose.POSE_CONNECTIONS)           
                        pose_landmarks_dict[landmark_number] = (int(x), int(y))
                        if 0 <= x < frame_width and 0 <= y < frame_height:
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    for connection in pose_connections:
                        if connection[0] in pose_landmarks_dict and connection[1] in pose_landmarks_dict:
                            start = pose_landmarks_dict[connection[0]]
                            end = pose_landmarks_dict[connection[1]]
                            cv2.line(frame, start, end, (0, 255, 0), 2)
                        

            # Draw hand landmarks
                for hand_landmarks_set in hand_landmarks:
                    hand_landmarks_dict = {}
                    for landmark in hand_landmarks_set:
                        landmark_number, x, y, z = landmark
                        hand_landmarks_dict[landmark_number] = (int(x), int(y))
                        if 0 <= x < frame_width and 0 <= y < frame_height:
                            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)     
                    for connection in hand_connections:
                        if connection[0] in hand_landmarks_dict and connection[1] in hand_landmarks_dict:
                            start = hand_landmarks_dict[connection[0]]
                            end = hand_landmarks_dict[connection[1]]
                            cv2.line(frame, start, end, (255, 0, 0), 2)    

                out.write(frame)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            out.release()
            cv2.destroyAllWindows()

    cursor.close()

on_connect("SUIT TAIL TREE 9_OCLOCK")