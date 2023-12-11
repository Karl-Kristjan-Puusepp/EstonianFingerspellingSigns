import os
import csv
import cv2
import mediapipe as mp
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np
import csv
from matplotlib.colors import TABLEAU_COLORS

# Function to extract hand landmarks using mediapipe
def extract_hand_landmarks(image_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and extract hand landmarks
    results = hands.process(image_rgb)
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for point in hand_landmarks.landmark:
                landmarks.extend([point.x, point.y, point.z])

    hands.close()
    return landmarks


# Function to process images in a folder and save hand landmarks to a CSV file
def process_folder(folder_path, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ['label', 'landmark_1_x', 'landmark_1_y', 'landmark_1_z', ..., 'landmark_63_x', 'landmark_63_y',
             'landmark_63_z'])

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()

        for label in tqdm(os.listdir(folder_path), desc="Processing folders"):
            label_path = os.path.join(folder_path, label)
            if os.path.isdir(label_path):
                for image_name in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
                    image_path = os.path.join(label_path, image_name)

                    # Extract hand landmarks
                    landmarks = extract_hand_landmarks(image_path)

                    # Check if the number of landmarks is as expected
                    if len(landmarks) == 63:
                        # Write label and landmarks to CSV
                        csv_writer.writerow([label] + landmarks)

        hands.close()


# Function to perform PCA on the data
def perform_pca(input_csv, output_csv):
    # Read data from CSV file
    data = np.genfromtxt(input_csv, delimiter=',', skip_header=1, dtype='str')

    # Extract labels and features
    labels = data[:, 0]
    features = data[:, 1:].astype(float)

    # Perform PCA
    pca = PCA()
    principal_components = pca.fit_transform(features)

    # Save all PCA results to CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['label'] + [f'PC{i + 1}' for i in range(len(principal_components[0]))]
        csv_writer.writerow(header)
        for i in range(len(labels)):
            if np.isnan(principal_components[i]).any():
                continue  # Skip rows with 'nan' labels
            csv_writer.writerow([labels[i]] + principal_components[i].tolist())

    # Print explained variance of each component
    explained_variance = pca.explained_variance_ratio_
    for i, variance in enumerate(explained_variance, 1):
        print(f'Explained Variance of PC{i}: {variance:.4f}')

    # Print top 5 landmarks contributing to each PC
    loadings = pca.components_
    for i in range(len(loadings)):
        print(f'\nTop 5 Landmarks Contributing to PC{i + 1}:')
        top_landmarks_indices = np.abs(loadings[i]).argsort()[-5:][::-1]
        for j in top_landmarks_indices:
            print(f'Landmark_{j + 1}: {loadings[i, j]:.4f}')


# Function to scatter plot two specified landmarks colored by label
def plot_landmarks(input_csv, feature1, feature2):
    # Read data from CSV file
    data = np.genfromtxt(input_csv, delimiter=',', skip_header=1, dtype='str')

    if data.size == 0:
        print("Error: CSV file is empty.")
        return

    # Extract labels and features
    if feature1 not in data[0] or feature2 not in data[0]:
        print(f"Error: One or both of the specified features ({feature1}, {feature2}) not found in the CSV file.")
        return

    labels = data[:, 0]
    feature1_column_index = np.where(data[0] == feature1)[0][0]  # No need to subtract 1 for label column
    feature2_column_index = np.where(data[0] == feature2)[0][0]  # No need to subtract 1 for label column

    feature1_values = data[:, feature1_column_index].astype(float)
    feature2_values = data[:, feature2_column_index].astype(float)

    # Get unique labels
    unique_labels = np.unique(labels)

    # Set up a cycle of colors for plotting
    color_cycle = cycle(plt.cm.get_cmap('tab10').colors)

    # Scatter plot
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        color = next(color_cycle)
        plt.scatter(
            feature1_values[label_indices],
            feature2_values[label_indices],
            label=label,
            color=color
        )

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Scatter Plot of {feature1} vs {feature2} Colored by Label')
    plt.legend()
    plt.show()


def generate_unique_colors(num_colors):
    # Generate a list of unique colors
    colors = list(TABLEAU_COLORS.values()) + list(plt.cm.get_cmap('tab20c').colors)
    if num_colors <= len(colors):
        return colors[:num_colors]
    else:
        raise ValueError("Requested number of colors exceeds the available color palette.")

def plot_pca_3d(input_csv):
    # Read data from CSV file
    data = np.genfromtxt(input_csv, delimiter=',', skip_header=1, dtype='str')

    if data.size == 0:
        print("Error: CSV file is empty.")
        return

    # Extract labels and features
    labels = data[:, 0]
    features = data[:, 1:].astype(float)

    # Perform PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features)

    # Get unique labels
    unique_labels = np.unique(labels)

    # Generate unique colors for labels
    label_colors = generate_unique_colors(len(unique_labels))

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8), dpi=400)
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(unique_labels):
        label_indices = np.where(labels == label)[0]
        color = label_colors[i]

        ax.scatter(
            principal_components[label_indices, 0],
            principal_components[label_indices, 1],
            principal_components[label_indices, 2],
            label=label,
            color=color,
            s=25  # Adjust marker size as needed
        )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D Scatter Plot of Top 3 PCA Components Colored by Label')

    ax.set_xlim([-1.6, 1.5])  # Adjust as needed
    ax.set_ylim([0, 3.5])  # Adjust as needed
    ax.set_zlim([-1.5, 0.5])  # Adjust as needed

    ax.legend()
    plt.show()


def plot_pca_2d(input_csv, selected_label=None):
    # Read data from CSV file
    data = np.genfromtxt(input_csv, delimiter=',', skip_header=1, dtype='str')

    if data.size == 0:
        print("Error: CSV file is empty.")
        return

    # Extract labels and features
    labels = data[:, 0]
    features = data[:, 1:].astype(float)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features)

    # Get unique labels
    unique_labels = np.unique(labels)

    # Generate unique colors for labels
    label_colors = generate_unique_colors(len(unique_labels))

    # Create a 2D scatter plot
    plt.figure(figsize=(10, 8), dpi=400)

    for i, label in enumerate(unique_labels):
        if selected_label is not None and label != selected_label:
            continue  # Skip labels that are not selected

        label_indices = np.where(labels == label)[0]
        color = label_colors[i]

        # Differentiate the second half with squares
        marker = 's' if i < len(unique_labels) // 2 else 'o'

        plt.scatter(
            principal_components[label_indices, 0],
            principal_components[label_indices, 1],
            label=label,
            color=color,
            marker=marker,
            s=30  # Adjust marker size as needed
        )

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    if selected_label is not None:
        plt.title(f'2D Scatter Plot for Label: {selected_label}')
    else:
        plt.title('2D Scatter Plot of Top 2 PCA Components Colored by Label')

    plt.legend()
    plt.show()
# Example usage
input_folder = 'path/to/your/labeled/folder'
output_csv_hand_landmarks = 'hand_landmarks.csv'
output_csv_pca = 'pca_results.csv'
output_csv_synthetic = 'synthetic_landmarks.csv'

# Process images and save hand landmarks to CSV
#process_folder(input_folder, output_csv_hand_landmarks)

# Perform PCA on the data
#perform_pca(output_csv_hand_landmarks, output_csv_pca)

#plot_pca_3d(output_csv_pca)
plot_pca_2d(output_csv_pca, 'H')