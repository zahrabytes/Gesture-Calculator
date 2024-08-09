import cvzone
import cv2
import numpy as np
import google.generativeai as genai
import os
import customtkinter
from cvzone.HandTrackingModule import HandDetector
from dotenv import load_dotenv
from PIL import Image, ImageTk
# get video feed on interface
# get output on interface
# toggle button
# tutorial
load_dotenv()

genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3, 600)
cap.set(4, 400)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def button_callback():
    print("button clicked")


def getHandInfo(img):
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas, img):
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None: 
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1,1,1,1,1]:
        canvas = np.zeros_like(img)
    
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [0,0,0,0,1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["solve this math question and give me concise step by step to do so", pil_image])
        text_output.configure(state="normal")  # Enable text widget
        text_output.delete("1.0", "end")  # Clear previous content
        text_output.insert("1.0", response.text)  # Insert new text
        text_output.configure(state="disabled") 


prev_pos = None
canvas = None
image_combined = None

app = customtkinter.CTk()
app.geometry("1280x720")

app.title("My AI Math Assistant")

title_label = customtkinter.CTkLabel(app, text="My AI Math Assistant", 
                                     font=("Arial", 48, "bold"),  # Font and size for the title
                                     text_color="lightblue")  # Set text color
title_label.pack(pady=20)

# Add a label to display the image
label = customtkinter.CTkLabel(app)
label.pack(side="left",padx=50, pady=20)

text_output = customtkinter.CTkTextbox(app, width=400, height=200)
text_output.pack(side="right",padx=20, pady=20)
text_output.configure(state="disabled")

def update_frame():
    global prev_pos, canvas, image_combined

    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas, img)
        sendToAI(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    # Convert the image_combined to a PIL image, then to a PhotoImage
    img_pil = Image.fromarray(cv2.cvtColor(image_combined, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(img_pil)

    # Update the label with the new image
    label.img_tk = img_tk  # Keep a reference to the image to avoid garbage collection
    label.configure(image=img_tk, text = "")

    # Repeat the function after 10 ms to create a video loop
    label.after(10, update_frame)

# Start the video loop
update_frame()

# Start the main loop of the app
app.mainloop()