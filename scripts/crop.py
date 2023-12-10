import os
import cv2
import mediapipe as mp
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def detect_and_crop_hand_landmarks(image_path, output_folder, log_file):
    image = cv2.imread(image_path)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)
    min_h = 1
    max_h = 0
    min_w = 1
    max_w = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                h, w, _ = image.shape
                if (landmark.x < min_w):
                    min_w = landmark.x
                if (landmark.x > max_w):
                    max_w = landmark.x
                if (landmark.y < min_h):
                    min_h = landmark.y
                if (landmark.y > max_h):
                    max_h = landmark.y

        # Convert normalized coordinates to pixel values
        min_w = int(min_w * w)
        max_w = int(max_w * w)
        min_h = int(min_h * h)
        max_h = int(max_h * h)

        # Add some margin (15% of image size) to the bounding box
        margin = 0.15
        min_w = max(0, min_w - int(margin * w))
        max_w = min(w, max_w + int(margin * w))
        min_h = max(0, min_h - int(margin * h))
        max_h = min(h, max_h + int(margin * h))

        cropped_image = image[min_h:max_h, min_w:max_w]

        # Check if cropped_image is not empty before writing
        if cropped_image.size > 0:
            if hands.process(cropped_image).multi_hand_landmarks:
                # Create the output folder if it doesn't exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                cropped_output_path = os.path.join(output_folder, os.path.basename(image_path))
                cv2.imwrite(cropped_output_path, cropped_image)
            else:
                print(f"Cropped image not detected for {image_path}", file=log_file)
        else:
            print(f"Empty cropped image for {image_path}", file=log_file)

    # Release resources
    hands.close()

# Example usage on a folder
input_folder_root = "./oneHandedGestures"
output_folder_root = "./oneHandedGesturesCropped"
log_file_path = "../data/log.txt"

with open(log_file_path, 'w') as log_file:
    for subfolder in os.listdir(input_folder_root):
        input_folder = os.path.join(input_folder_root, subfolder)
        output_folder = os.path.join(output_folder_root, subfolder)

        for filename in tqdm(os.listdir(input_folder), desc=subfolder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(input_folder, filename)
                detect_and_crop_hand_landmarks(image_path, output_folder, log_file)
