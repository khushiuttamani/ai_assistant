import cv2
import base64
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
print("DEBUG - Loaded GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

def capture_image() -> str:
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            for _ in range(10):
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            cv2.imwrite("sample.jpg", frame)
            ret, buf = cv2.imencode('.jpg', frame)
            if ret:
                return base64.b64encode(buf).decode('utf-8')
    raise RuntimeError("Could not open any webcam (tried indices 0-3)")

def analyze_image_with_query(query: str) -> str:
    """
    Takes a user query that depends on visual input, captures a webcam image,
    and returns a natural-language answer using Groq's vision model.
    """
    img_b64 = capture_image()
    model = "meta-llama/llama-4-maverick-17b-128e-instruct"

    if not query or not img_b64:
        return "Error: both 'query' and 'image' fields required."

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                    },
                },
            ],
        }
    ]
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content