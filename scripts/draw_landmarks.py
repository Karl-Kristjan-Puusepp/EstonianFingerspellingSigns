import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke

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
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)

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

        # Add some margin (10% of image size) to the bounding box
        margin = 0.10
        min_w = max(0, min_w - int(margin * w))
        max_w = min(w, max_w + int(margin * w))
        min_h = max(0, min_h - int(margin * h))
        max_h = min(h, max_h + int(margin * h))

        # Crop the image to the bounding box
        cropped_image = image[min_h:max_h, min_w:max_w]

        # Process image and detect hand landmarks in the cropped image
        results_crop = hands.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        fig, ax = plt.subplots()
        ax.imshow(cropped_image[:, :, ::-1])  # Reverse the order of color channels (BGR to RGB)

        # Draw hand landmarks on the cropped image
        if results_crop.multi_hand_landmarks:
            hand_landmarks = results_crop.multi_hand_landmarks[0]
            x_coords = [int(landmark.x * cropped_image.shape[1]) for landmark in hand_landmarks.landmark]
            y_coords = [int(landmark.y * cropped_image.shape[0]) for landmark in hand_landmarks.landmark]

            ax.scatter(x_coords, y_coords, marker='o', color='blue', s=50)

            for connection in list(mp_hands.HAND_CONNECTIONS)[:21]:  # Only draw the first 21 connections
                idx_1, idx_2 = connection
                x1, y1 = x_coords[idx_1], y_coords[idx_1]
                x2, y2 = x_coords[idx_2], y_coords[idx_2]

                line = Line2D([x1, x2], [y1, y2], color='white', linewidth=2)
                ax.add_line(line)

            for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
                text = ax.text(x, y, str(idx), fontsize=12, color='white', ha='left', va='bottom')
                text.set_path_effects([withStroke(linewidth=3, foreground='black')])

        plt.show()

        legend_labels = [
            f"{idx}. {label}" for idx, label in enumerate([
                "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
                "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
                "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
                "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
                "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
            ])
        ]
        ax.legend(handles=[], labels=legend_labels, loc='lower right', bbox_to_anchor=(1.0, 0.5))

        plt.savefig(output_folder + "/cropped_image.svg", format='svg', bbox_inches='tight')
        cv2.imwrite(output_folder + "/cropped_image.jpg", cropped_image)

image_path = "../data/oneHandedGestures/B/B_070.jpg"
output_folder = "."
log_file = "log2.txt"

detect_and_crop_hand_landmarks(image_path, output_folder, log_file)
