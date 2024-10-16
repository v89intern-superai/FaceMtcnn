import streamlit as st
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from ultralytics import YOLO

# Function to load MTCNN model
def load_mtcnn_model(device):
    return MTCNN(keep_all=True, device=device)

# Function to load FaceNet model
def load_facenetwork_model(device):
    return InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to load YOLO model
def load_yolo_model(model_choice):
    return YOLO(model_choice)

# Function to augment the image
def augment_image(image, flip=False, rotate_angle=0, contrast_factor=1.0):
    # Convert to PIL Image for augmentation
    pil_image = Image.fromarray(image)

    # Flip the image horizontally
    if flip:
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotate the image
    if rotate_angle != 0:
        pil_image = pil_image.rotate(rotate_angle)

    # Adjust contrast
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)

    # Convert back to NumPy array
    return np.array(pil_image)

# Determine if an NVIDIA GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load known faces from a single .pt file
known_faces_file = r'D:\Friend\Super AI\V89\My Project\facenet-pytorch-master\examples\embeddings_with_names.pt'  # Update the path
known_faces_data = torch.load(known_faces_file)  # Load embeddings

# Extract embeddings and names
known_face_embeddings = []
known_face_names = []
for name, embedding in known_faces_data.items():
    known_face_names.append(name)
    known_face_embeddings.append(embedding)

# Convert known_face_embeddings to a tensor
known_face_embeddings = torch.stack(known_face_embeddings).to(device)

# Function to recognize faces
def recognize_face(embedding, known_face_embeddings, known_face_names, threshold=1.0):
    distances = [torch.norm(embedding - known_face_embedding).item() for known_face_embedding in known_face_embeddings]
    min_distance_index = np.argmin(distances)
    if distances[min_distance_index] < threshold:
        return known_face_names[min_distance_index]
    return "Unknown"

# Streamlit App
st.title("Face Recognition App")

# Model selection
model_choice = st.selectbox("Choose a face detection model:", ["MTCNN", "YOLOv8m", "YOLOv9m", "YOLOv10m"])

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Slider for threshold
threshold = st.slider("Recognition Threshold", 0.0, 5.0, 1.0)

# Options for data augmentation
# Set default values in session state if not already set
if 'flip' not in st.session_state:
    st.session_state.flip = False
if 'rotate_angle' not in st.session_state:
    st.session_state.rotate_angle = 0  # Default to 0
if 'contrast_factor' not in st.session_state:
    st.session_state.contrast_factor = 1.0

# UI for data augmentation options
st.session_state.flip = st.checkbox("Flip Image Horizontally", value=st.session_state.flip)

# Select box for rotation angle with discrete values
rotation_options = [-90,-180, 0, 90, 180]
st.session_state.rotate_angle = st.selectbox("Rotate Image", rotation_options, index=rotation_options.index(st.session_state.rotate_angle))

# Slider for contrast factor
st.session_state.contrast_factor = st.slider("Contrast Factor", 0.0, 5.0, st.session_state.contrast_factor)

# Reset function
def reset_parameters():
    st.session_state.flip = False
    st.session_state.rotate_angle = 0
    st.session_state.contrast_factor = 1.0

# Reset button
if st.button("Reset"):
    reset_parameters()

if uploaded_file is not None:
    # Convert the file to an opencv image.
    image = Image.open(uploaded_file)
    image_rgb = np.array(image)

    # Apply data augmentation
    augmented_image = augment_image(image_rgb, flip=st.session_state.flip, rotate_angle=st.session_state.rotate_angle, contrast_factor=st.session_state.contrast_factor)

    st.image(augmented_image, caption='Augmented Image', use_column_width=True)
    st.write("Processing...")

    boxes = None  # Initialize boxes for detection
    model = None  # Initialize model

    # Initialize the chosen model
    if model_choice == "MTCNN":
        model = load_mtcnn_model(device)  # Load MTCNN
        boxes, _ = model.detect(augmented_image)

    elif model_choice in ["YOLOv8m", "YOLOv9m", "YOLOv10m"]:
        model_path = f"{model_choice.lower()}.pt"  # Change this to your model path if necessary
        model = load_yolo_model(model_path)  # Load the selected YOLO model
        results = model(augmented_image)  # Perform detection
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get the bounding boxes

    detected_faces = []  # To store detected faces and their labels

    if boxes is not None:
        for box in boxes:
            # Convert box coordinates to integer
            box = box.astype(int)

            # Ensure the box coordinates are within the image boundaries
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(augmented_image.shape[1], box[2])
            box[3] = min(augmented_image.shape[0], box[3])

            # Draw rectangle on the original image
            cv2.rectangle(augmented_image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

            # Get the face region for recognition
            face_region = augmented_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            # Check if the face_region is valid before resizing
            if face_region.size == 0:
                st.write("Face region is empty, skipping this box.")
                continue

            face_region_resized = cv2.resize(face_region, (160, 160))  # Resize for FaceNet
            face_tensor = torch.tensor(face_region_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            # Load FaceNet model and get embedding
            with torch.no_grad():
                face_network_model = load_facenetwork_model(device)
                embedding = face_network_model(face_tensor).cpu()
                identity = recognize_face(embedding, known_face_embeddings, known_face_names, threshold)

            detected_faces.append((face_region_resized, identity))
            st.write(f"Identified: {identity}")

        # Display the processed image with bounding boxes
        st.image(augmented_image, caption='Processed Image', use_column_width=True)

        # Display detected faces
        st.write("Detected Faces:")
        num_faces = len(detected_faces)
        cols = st.columns(min(5, num_faces))

        for i, (face, identity) in enumerate(detected_faces):
            cols[i % 5].image(face, caption=identity, use_column_width=True)

    else:
        st.write("No faces detected in the image.")
