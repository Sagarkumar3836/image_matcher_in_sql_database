import cv2
import numpy as np
import mysql.connector
import faiss
from insightface.app import FaceAnalysis

# Initialize RetinaFace and ArcFace
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def extract_embedding(image_input):
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise Exception(f"Failed to load image from {image_input}")
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise Exception("Invalid image input. Provide a valid image path or NumPy array.")
    
    faces = face_app.get(image)
    if len(faces) == 0:
        raise Exception("No face detected")
    
    return faces[0].embedding.astype(np.float32)  # Ensure float32 for FAISS compatibility

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="sagar",
        database="imgdataset"
    )

def fetch_data_from_db():
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, name, image_data FROM Images")

            ids, names, images = [], [], []
            for row in cursor.fetchall():
                ids.append(row[0])
                names.append(row[1])
                images.append(np.frombuffer(row[2], dtype=np.uint8))
    
    return ids, names, images

def search_image(query_path):
    try:
        query_embedding = extract_embedding(query_path)
        dim = len(query_embedding)  # Get embedding dimension

        ids, names, images = fetch_data_from_db()
        embeddings = []
        
        for img_data in images:
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            try:
                embedding = extract_embedding(img)
                embeddings.append(embedding)
            except Exception:
                embeddings.append(np.zeros(dim, dtype=np.float32))  # Avoid failures

        if not embeddings:
            raise Exception("No valid embeddings found in database.")

        # Convert to NumPy array
        embeddings = np.array(embeddings, dtype=np.float32)

        # FAISS Index for similarity search
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # Search for the most similar image
        distances, indices = index.search(np.array([query_embedding], dtype=np.float32), 1)

        best_match_id = ids[indices[0][0]]
        best_match_name = names[indices[0][0]]
        
        print(f"Best Match: {best_match_name}, Distance: {distances[0][0]}")

    except Exception as e:
        print(f"Error: {e}")

# Example Usage
search_image("C:/Users/Ayush Sagar/Desktop/img to img/011.png")
