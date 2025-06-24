import pygame
import sys
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import math
import random

def main():
    pygame.init()

    screen_width = 1280
    screen_height = 720
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Project Chimera: Operator Symbiote")
    clock = pygame.time.Clock()
    font_small = pygame.font.Font("C:/Windows/Fonts/consola.ttf", 20)
    font_large = pygame.font.Font("C:/Windows/Fonts/consola.ttf", 28)

    cap = cv2.VideoCapture(0)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
    
    stars = []
    for _ in range(200):
        x = random.randint(-screen_width, screen_width)
        y = random.randint(-screen_height, screen_height)
        z = random.randint(1, screen_width)
        stars.append([x, y, z])

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False

        success, frame = cap.read()
        if not success:
            continue
        
        h, w, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        screen.fill((0, 0, 10))

        center_x, center_y = screen_width // 2, screen_height // 2
        
        yaw = 0
        pitch = 0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
            
            nose_2d = (landmarks[1][0] * w, landmarks[1][1] * h)
            face_3d = np.array([
                (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h, face_landmarks.landmark[1].z),
                (face_landmarks.landmark[199].x * w, face_landmarks.landmark[199].y * h, face_landmarks.landmark[199].z),
                (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h, face_landmarks.landmark[33].z),
                (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h, face_landmarks.landmark[263].z),
                (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h, face_landmarks.landmark[61].z),
                (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h, face_landmarks.landmark[291].z)
            ], dtype=np.float64)
            face_2d = np.array([(p[0], p[1]) for p in face_3d], dtype=np.float64)

            focal_length = 1 * w
            cam_matrix = np.array([[focal_length, 0, h / 2], [0, focal_length, w / 2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            pitch, yaw, roll = angles[0] * 360, angles[1] * 360, angles[2] * 360

            for i in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                p1 = (int(landmarks[i[0]][0] * w), int(landmarks[i[0]][1] * h))
                p2 = (int(landmarks[i[1]][0] * w), int(landmarks[i[1]][1] * h))
                pygame.draw.line(screen, (0, 150, 100, 50), p1, p2, 1)

            left_eye_pts = np.array([(landmarks[i][0] * w, landmarks[i][1] * h) for i in LEFT_EYE_IDXS])
            right_eye_pts = np.array([(landmarks[i][0] * w, landmarks[i][1] * h) for i in RIGHT_EYE_IDXS])
            
            A = dist.euclidean(left_eye_pts[1], left_eye_pts[5])
            B = dist.euclidean(left_eye_pts[2], left_eye_pts[4])
            C = dist.euclidean(left_eye_pts[0], left_eye_pts[3])
            ear = (A + B) / (2.0 * C)
            
            ear_bar = int(np.interp(ear, [0.2, 0.4], [0, 150]))
            pygame.draw.rect(screen, (0, 255, 0), (20, 50, ear_bar, 20))
            pygame.draw.rect(screen, (255, 255, 255), (20, 50, 150, 20), 1)
            ear_text = font_large.render(f"BLINK LVL", True, (255, 255, 0))
            screen.blit(ear_text, (20, 15))

            pose_text = font_small.render(f"P: {pitch:.0f} Y: {yaw:.0f} R: {roll:.0f}", True, (255,255,0))
            screen.blit(pose_text, (screen_width - 200, 20))


        for star in stars:
            star[2] -= 2
            if star[2] <= 0:
                star[0] = random.randint(-screen_width, screen_width)
                star[1] = random.randint(-screen_height, screen_height)
                star[2] = random.randint(center_x, screen_width)

            k = 128.0 / star[2]
            star_x = int(star[0] * k + center_x + yaw * 2)
            star_y = int(star[1] * k + center_y - pitch * 2)

            if 0 < star_x < screen_width and 0 < star_y < screen_height:
                size = (1 - star[2] / screen_width) * 4
                shade = int((1 - star[2] / screen_width) * 200)
                pygame.draw.rect(screen, (shade, shade, shade), (star_x, star_y, size, size))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    face_mesh.close()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()