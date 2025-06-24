import pygame
import sys
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

def main():
    pygame.init()

    screen_width = 1280
    screen_height = 720
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Project Chimera: Operator Symbiote")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    cap = cv2.VideoCapture(0)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    
    def calculate_ear(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def calculate_mar(mouth):
        A = dist.euclidean(mouth[1], mouth[7])
        B = dist.euclidean(mouth[2], mouth[6])
        C = dist.euclidean(mouth[3], mouth[5])
        D = dist.euclidean(mouth[0], mouth[4])
        mar = (A + B + C) / (2.0 * D)
        return mar
    
    LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
    MOUTH_IDXS = [61, 76, 291, 405, 314, 14, 84, 17]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        success, frame = cap.read()
        if not success:
            continue

        h, w, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        screen.fill((0, 0, 10))
        
        if results.multi_face_landmarks:
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in results.multi_face_landmarks[0].landmark])

            left_eye = np.array([landmarks[i] for i in LEFT_EYE_IDXS])
            right_eye = np.array([landmarks[i] for i in RIGHT_EYE_IDXS])
            mouth = np.array([landmarks[i] for i in MOUTH_IDXS])

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(mouth)

            for lm in landmarks:
                pygame.draw.circle(screen, (0, 255, 150), (int(lm[0]), int(lm[1])), 1)

            ear_text = font.render(f"EAR: {avg_ear:.2f}", True, (255, 255, 0))
            mar_text = font.render(f"MAR: {mar:.2f}", True, (255, 255, 0))
            screen.blit(ear_text, (10, 10))
            screen.blit(mar_text, (10, 50))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    face_mesh.close()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()