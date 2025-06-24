import pygame
import sys
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import math

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

    def calculate_ear(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def calculate_mar(mouth):
        A = dist.euclidean(mouth[1], mouth[7])
        B = dist.euclidean(mouth[2], mouth[6])
        C = dist.euclidean(mouth[3], mouth[5])
        D = dist.euclidean(mouth[0], mouth[4])
        return (A + B + C) / (2.0 * D)
    
    LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
    MOUTH_IDXS = [61, 76, 291, 405, 314, 14, 84, 17]

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
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks])

            left_eye = np.array([landmarks[i] for i in LEFT_EYE_IDXS])
            right_eye = np.array([landmarks[i] for i in RIGHT_EYE_IDXS])
            mouth = np.array([landmarks[i] for i in MOUTH_IDXS])

            avg_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)
            
            nose_2d = (landmarks[1][0], landmarks[1][1])
            face_3d = np.array([
                (face_landmarks[1].x * w, face_landmarks[1].y * h, face_landmarks[1].z), # Nose tip
                (face_landmarks[199].x * w, face_landmarks[199].y * h, face_landmarks[199].z), # Chin
                (face_landmarks[33].x * w, face_landmarks[33].y * h, face_landmarks[33].z), # Left eye left corner
                (face_landmarks[263].x * w, face_landmarks[263].y * h, face_landmarks[263].z),# Right eye right corner
                (face_landmarks[61].x * w, face_landmarks[61].y * h, face_landmarks[61].z), # Left Mouth corner
                (face_landmarks[291].x * w, face_landmarks[291].y * h, face_landmarks[291].z) # Right mouth corner
            ], dtype=np.float64)

            face_2d = np.array([
                (lm[0], lm[1]) for lm in face_3d
            ], dtype=np.float64)

            focal_length = 1 * w
            cam_matrix = np.array([ [focal_length, 0, h / 2],
                                    [0, focal_length, w / 2],
                                    [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            
            nose_end_point2D, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
            
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            pygame.draw.line(screen, (255, 0, 0), p1, p2, 2)

            for lm in landmarks:
                pygame.draw.circle(screen, (0, 255, 150), (int(lm[0]), int(lm[1])), 1)

            parallax_x = int(y * 0.1)
            parallax_y = int(x * 0.1)

            ear_text = font.render(f"EAR: {avg_ear:.2f}", True, (255, 255, 0))
            mar_text = font.render(f"MAR: {mar:.2f}", True, (255, 255, 0))
            pos_text = font.render(f"YAW: {y:.0f}", True, (255, 255, 0))
            
            screen.blit(ear_text, (10 - parallax_x, 10 - parallax_y))
            screen.blit(mar_text, (10 - parallax_x, 50 - parallax_y))
            screen.blit(pos_text, (10 - parallax_x, 90 - parallax_y))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    face_mesh.close()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()