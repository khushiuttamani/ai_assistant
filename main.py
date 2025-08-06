import os
import gradio as gr
import time
from speech_to_text import record_audio, transcribe_with_groq
from ai_agent import ask_agent
from text_to_speech import text_to_speech_with_elevenlabs
import cv2

# --- Gradio UI setup ---
# Global variables for continuous loop control
is_listening = False
audio_filepath = "audio_question.mp3"
# Initialize chatbot_history as an empty list to store message dictionaries
chatbot_history = []
camera = None
is_camera_running = False
last_frame = None

def start_listening():
    """Starts the continuous listening loop."""
    global is_listening
    if not is_listening:
        is_listening = True
        return "Continuous listening started. Say 'goodbye' to stop."
    return "Already listening."

def stop_listening():
    """Stops the continuous listening loop."""
    global is_listening
    is_listening = False
    return "Listening stopped."

def process_audio_and_chat_generator():
    """A generator function for continuous audio processing in Gradio."""
    global is_listening, chatbot_history

    while is_listening:
        try:
            # Check if listening is enabled
            if not is_listening:
                break

            # Record audio and transcribe
            print("Recording audio...")
            if not record_audio(file_path=audio_filepath, timeout=5, phrase_time_limit=10):
                print("No speech detected, continuing loop...")
                time.sleep(1)
                continue
            
            print("Transcribing audio...")
            user_input = transcribe_with_groq(audio_filepath)
            
            if "Error" in user_input or user_input is None or user_input.strip() == "":
                print("Transcription failed or no speech. Skipping...")
                time.sleep(1)
                continue
            
            print(f"User says: {user_input}")

            # Check for a stop command
            if "goodbye" in user_input.lower():
                stop_listening()
                # Append the final message and yield one last time
                chatbot_history.append({"role": "user", "content": user_input})
                chatbot_history.append({"role": "assistant", "content": "Goodbye! Have a great day."})
                yield chatbot_history
                break

            # Append user message in the correct dictionary format and yield to update the UI
            chatbot_history.append({"role": "user", "content": user_input})
            yield chatbot_history
            
            # Send query to the AI agent
            response = ask_agent(user_query=user_input)

            # Append the assistant's response in the correct dictionary format
            chatbot_history.append({"role": "assistant", "content": response})
            yield chatbot_history

            # Using ElevenLabs (placeholder)
            text_to_speech_with_elevenlabs(input_text=response, output_filepath="final.mp3")

        except Exception as e:
            print(f"Error in continuous recording loop: {e}")
            is_listening = False
            break

def clear_chat():
    """Clears the chat history."""
    global chatbot_history
    chatbot_history = []
    # Return the empty list to update the Gradio component
    return []

# --- Webcam Functions (no changes needed here) ---
def initialize_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera is not None and camera.isOpened()

def start_webcam():
    global is_camera_running, last_frame
    is_camera_running = True
    if not initialize_camera():
        return None
    ret, frame = camera.read()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        return frame
    return last_frame

def stop_webcam():
    global is_camera_running, camera
    is_camera_running = False
    if camera is not None:
        camera.release()
        camera = None
    return None

def get_webcam_frame():
    global camera, is_camera_running, last_frame
    if not is_camera_running or camera is None:
        return last_frame
    
    if camera.get(cv2.CAP_PROP_BUFFERSIZE) > 1:
        for _ in range(int(camera.get(cv2.CAP_PROP_BUFFERSIZE)) - 1):
            camera.read()
    
    ret, frame = camera.read()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        return frame
    return last_frame

# --- Gradio UI setup ---
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='color: orange; text-align: center;  font-size: 4em;'> 👧🏼 Dora – Your Personal AI Assistant</h1>")

    with gr.Row():
        # Left column - Webcam
        with gr.Column(scale=1):
            gr.Markdown("## Webcam Feed")
            
            with gr.Row():
                start_btn = gr.Button("Start Camera", variant="primary")
                stop_btn = gr.Button("Stop Camera", variant="secondary")
            
            webcam_output = gr.Image(
                label="Live Feed",
                streaming=True,
                show_label=False,
                width=640,
                height=480
            )
            
            webcam_timer = gr.Timer(0.033)
        
        # Right column - Chat
        with gr.Column(scale=1):
            gr.Markdown("## Chat Interface")
            
            # Corrected: Added type='messages' and now using the correct data format
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                show_label=False,
                type='messages'
            )
            
            gr.Markdown("*🎤 Continuous listening mode is active - speak anytime!*")
            
            with gr.Row():
                start_listening_btn = gr.Button("Start Listening", variant="primary")
                stop_listening_btn = gr.Button("Stop Listening", variant="secondary")
                clear_btn = gr.Button("Clear Chat", variant="secondary")

    # Event handlers
    start_btn.click(
        fn=start_webcam,
        outputs=webcam_output
    )
    
    stop_btn.click(
        fn=stop_webcam,
        outputs=webcam_output
    )
    
    webcam_timer.tick(
        fn=get_webcam_frame,
        outputs=webcam_output,
        show_progress=False
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=chatbot
    )
    
    # Event handlers for listening control
    start_listening_btn.click(
        fn=start_listening,
        outputs=None
    ).then(
        fn=process_audio_and_chat_generator,
        outputs=chatbot
    )

    stop_listening_btn.click(
        fn=stop_listening,
        outputs=None
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        debug=True
    )