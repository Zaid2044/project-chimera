# Project Chimera: The Operator Symbiote

**Project Chimera is a state-of-the-art, real-time symbiotic interface that transforms a standard laptop and webcam into a powerful command and control center. This project merges computer vision, gesture control, voice recognition, and generative AI to create a visually spectacular and technically complex portfolio piece.**

The user becomes the "Operator," whose biometric data and commands are processed in real-time to interact with a futuristic, holographic-style interface.

---

### Core Features

-   **Real-time Biometric Tracking:** Utilizes `OpenCV` and `MediaPipe` to perform high-fidelity 3D tracking of the user's face mesh and hands.
-   **Dynamic Holo-Deck UI:** A custom `Pygame` interface renders the Operator's face as a live wireframe, floating in a 3D starfield that reacts to head movement for a convincing parallax effect.
-   **Gesture & Voice Control:** The system is activated via a "thumbs up" gesture. It then uses `SpeechRecognition` to listen for a spoken command from the Operator.
-   **The Oracle Protocol:** The captured voice command is sent to **Google's Gemini 1.5 Flash API**. The generative AI's response is streamed back and displayed on the Holo-Deck, creating a seamless conversation with a powerful AI.
-   **Robust & Performant:** Multi-threaded to handle voice recognition without freezing the UI, and built with specific error handling to manage API and audio input issues gracefully.

---

### Tech Stack

-   **Primary Language:** Python 3
-   **Computer Vision:** OpenCV, MediaPipe
-   **UI & Graphics:** Pygame
-   **Generative AI:** Google Gemini API (`google-generativeai`)
-   **Voice Input:** SpeechRecognition, PyAudio
-   **Math & Numerics:** NumPy, SciPy

---

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Zaid2044/project-chimera.git
    cd project-chimera
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    

4.  **Configure API Key:**
    -   Obtain a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    -   Open `main.py` and replace `'YOUR_API_KEY_HERE'` with your actual key.

5.  **Run the application:**
    ```bash
    python main.py
    ```

---

### How to Use

1.  Launch the application. The system will be in `IDLE` mode.
2.  Give a clear **thumbs up** gesture to your webcam. The status will change to `LISTENING`.
3.  Speak your question clearly into the microphone.
4.  The system will show `TRANSMITTING` as it queries the Gemini API.
5.  The Oracle's response will be displayed on the screen.

---

This project was built to push the boundaries of what is possible in a real-time, personal AI interface on consumer hardware.