import pygame
import sys
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import math
import random
import speech_recognition as sr
import google.generativeai as genai
import threading
import textwrap

GEMINI_API_KEY = 'YOUR_API_KEY_HERE'

def main():
    pygame.init()

    screen_width = 1280
    screen_height = 720
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Project Chimera: Oracle Protocol")
    clock = pygame.time.Clock()
    font_ui = pygame.font.Font("C:/Windows/Fonts/consola.ttf", 22)
    font_oracle = pygame.font.Font("C:/Windows/Fonts/bahnschrift.ttf", 28)

    cap = cv2.VideoCapture(0)
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    if GEMINI_API_KEY == 'YOUR_API_KEY_HERE':
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE PUT YOUR GEMINI API KEY IN THE SCRIPT !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    stars = [[random.randint(-screen_width, screen_width), random.randint(-screen_height, screen_height), random.randint(1, screen_width)] for _ in range(200)]

    state = "IDLE"
    oracle_response = ""
    oracle_thread = None
    
    def listen_and_query():
        nonlocal state, oracle_response
        state = "LISTENING"
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.pause_threshold = 1.0
            r.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                oracle_response = "ORACLE: Listening timed out. No input detected."
                state = "IDLE"
                return

        try:
            query = r.recognize_google(audio)
            state = "QUERYING"
            oracle_response = ""
            response = model.generate_content(f"Answer concisely, as a futuristic AI assistant: {query}")
            oracle_response = response.text
        except sr.UnknownValueError:
            oracle_response = "ORACLE: Audio incomprehensible. Please try again."
        except sr.RequestError as e:
            oracle_response = f"ORACLE: Network error. Could not reach service. {e}"
        except Exception as e:
            oracle_response = f"ORACLE: An unexpected error occurred. {e}"
        state = "IDLE"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False

        success, frame = cap.read()
        if not success: continue
        
        h, w, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results_face = face_mesh.process(rgb_frame)
        results_hands = hands.process(rgb_frame)

        screen.fill((0, 0, 10))
        center_x, center_y = screen_width // 2, screen_height // 2
        yaw, pitch = 0, 0

        if results_face.multi_face_landmarks:
            face_lms = results_face.multi_face_landmarks[0].landmark
            landmarks = np.array([(lm.x, lm.y) for lm in face_lms])
            
            face_3d = np.array([(face_lms[i].x * w, face_lms[i].y * h, face_lms[i].z) for i in [1, 199, 33, 263, 61, 291]], dtype=np.float64)
            face_2d = np.array([(p[0], p[1]) for p in face_3d], dtype=np.float64)
            
            cam_matrix = np.array([[w, 0, h/2], [0, w, w/2], [0, 0, 1]])
            success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4,1), dtype=np.float64))
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch, yaw = angles[0] * 360, angles[1] * 360

            for i in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                p1 = (int(landmarks[i[0]][0] * w), int(landmarks[i[0]][1] * h))
                p2 = (int(landmarks[i[1]][0] * w), int(landmarks[i[1]][1] * h))
                pygame.draw.line(screen, (0, 100, 80, 50), p1, p2, 1)

        for star in stars:
            star[2] -= 1
            if star[2] <= 0: star[2] = random.randint(center_x, screen_width)
            k = 128.0 / star[2]
            star_x = int(star[0] * k + center_x + yaw * 1.5)
            star_y = int(star[1] * k + center_y - pitch * 1.5)
            if 0 < star_x < screen_width and 0 < star_y < screen_height:
                size = (1 - star[2] / screen_width) * 3
                shade = int((1 - star[2] / screen_width) * 150)
                pygame.draw.rect(screen, (shade, shade, shade), (star_x, star_y, size, size))
        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                lms = hand_landmarks.landmark
                thumb_tip = lms[4]
                index_mcp = lms[5]
                is_thumbs_up = thumb_tip.y < index_mcp.y
                
                if is_thumbs_up and state == "IDLE":
                    oracle_response = ""
                    oracle_thread = threading.Thread(target=listen_and_query)
                    oracle_thread.start()
                    state = "ACTIVATED"

        status_color = (255, 255, 0)
        status_text = "SYSTEM: IDLE [THUMBS UP TO ACTIVATE]"
        if state == "ACTIVATED": status_text = "ORACLE: ACTIVATED"
        if state == "LISTENING":
            status_text = "ORACLE: LISTENING..."
            status_color = (255, 0, 0)
        if state == "QUERYING": status_text = "ORACLE: ...TRANSMITTING..."

        hud_text = font_ui.render(status_text, True, status_color)
        screen.blit(hud_text, (20, 20))
        
        if oracle_response:
            wrapped_text = textwrap.wrap(oracle_response, width=60)
            for i, line in enumerate(wrapped_text):
                line_surface = font_oracle.render(line, True, (0, 255, 180))
                screen.blit(line_surface, (screen_width / 2 - line_surface.get_width() / 2, screen_height / 2 - 100 + i * 40))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()