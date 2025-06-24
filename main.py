import pygame
import sys
import cv2
import numpy as np
import mediapipe as mp

def main():
    pygame.init()

    screen_width = 1280
    screen_height = 720
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Project Chimera: Operator Symbiote")
    clock = pygame.time.Clock()

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec_points = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
    drawing_spec_lines = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_hands = hands.process(rgb_frame)
        results_face = face_mesh.process(rgb_frame)
        
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec_points,
                    connection_drawing_spec=drawing_spec_lines)
        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=2))

        frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
        frame_surface = pygame.transform.flip(frame_surface, True, False)
        
        scaled_frame = pygame.transform.scale(frame_surface, (screen_width, screen_height))
        
        screen.blit(scaled_frame, (0, 0))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()