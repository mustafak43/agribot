# put all the files in a folder
# create a virtual environment
# download all the necessary libraries
# & start using

import os
import asyncio
import keyboard
from openai import OpenAI

import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key
# api_key = os.environ.get("OPENAI_API_KEY")  # Or replace with your actual API key
api_key = "your-api-key"

client = OpenAI(api_key=api_key)

# CLIP Tiny Modelini Yükle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32').to(device)
# Global variables to store embeddings and image files in memory
all_embeddings = None
all_image_files = None

# Function to load embeddings and image files from a .npy file
def load_embeddings(npy_file):
    embeddings_data = np.load(npy_file, allow_pickle=True).item()
    embeddings = embeddings_data['embeddings']
    image_files = embeddings_data['image_files']
    return embeddings, image_files

# Function to load all embeddings from the provided .npy files (called only once)
def load_all_embeddings(npy_files):
    global all_embeddings, all_image_files
    if all_embeddings is None or all_image_files is None:
        all_embeddings = []
        all_image_files = []

        for npy_file in npy_files:
            embeddings, image_files = load_embeddings(npy_file)
            all_embeddings.append(embeddings)
            all_image_files.extend(image_files)

        # Flatten all embeddings into one array for easier similarity computation
        all_embeddings = np.vstack(all_embeddings)

# Function to check similarity for a given query image against loaded embeddings
def check_similarity(query_image_path, npy_files):
    # Load all embeddings once if they are not already loaded
    load_all_embeddings(npy_files)
    # Sorgu resmi embedding
    query_image = Image.open(query_image_path).resize((224, 224))
    query_embedding = model.encode(query_image, device=device)

    # Calculate cosine similarity between the query image and all loaded embeddings
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    top_match_idx = similarities.argmax()

    # Return the best match
    best_similarity = similarities[top_match_idx]
    best_match_image = all_image_files[top_match_idx]
    return best_match_image, best_similarity

# Example usage
npy_files = ['./embeddings/image_embeddings1.npy', './embeddings/image_embeddings2.npy', './embeddings/image_embeddings3.npy',
             './embeddings/image_embeddings4.npy', './embeddings/image_embeddings5.npy', './embeddings/image_embeddings6.npy',
             './embeddings/image_embeddings7.npy', './embeddings/image_embeddings8.npy', './embeddings/image_embeddings9.npy']  # List of .npy files containing embeddings
#query_image_path = './query.jpg'  # Path to the query image
#best_match_image, best_similarity = check_similarity(query_image_path, npy_files)
#print(f"Best match image: {best_match_image} with similarity {best_similarity}")

most_resembling_photo_file_name = ""

def get_the_most_resembling_photo(query_image_path):
    global npy_files
    query_image_path = './Test isimsiz resimler/' + query_image_path
    
    print("query_image_path: ", query_image_path)
    
    best_match_image, best_similarity = check_similarity(query_image_path, npy_files)
    
    return best_match_image, best_similarity

async def chat_with_gpt():
    print("Type 'q' to quit the conversation.")

    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant that assists with tasks and answers questions in agricultural context only. If you are given image: image_name in the conversation, that means you take into account the image_name before answering because it includes necessary information about the image you are hypothetically given. The image_name may look like 'Hastalıklı_angular_leafspot162.jpg_çilek' that means it's a diseased image with angular leafspot and it's a strawberry. You never mention the full name of the image instead you take the necessary information within the file name then parse it and tell the user what he wants to know about it."}  # System message
        
        # also add necessary system message so that it also takes necessary information with pdf rag into account
    ]
    
    global most_resembling_photo_file_name
    while True:
        # Get user input
        user_input = input("You: ")

        # Exit the loop if the user types 'q'
        if user_input.lower() == 'q':
            print("Exiting conversation.")
            break
        # Check if the 'Esc' key is pressed to save the conversation
        if user_input.lower() == 'attach image':
            
            file_name = input("Enter the file name of the image you want to attach: ")
            
            try:
                most_resembling_photo_file_name, similarity_score = get_the_most_resembling_photo(file_name)
                # print("most resembling photo: ", most_resembling_photo_file_name)
                print("similarity_score: ", similarity_score)
            except:
                print("An error occured")
            
            continue
        
        if most_resembling_photo_file_name != "":
            user_input += f"image: {most_resembling_photo_file_name}"
        
        # Add user input to conversation history
        conversation_history.append({"role": "user", "content": user_input})

        # Send the conversation history to GPT for response using async API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Or other available model
            messages=conversation_history,
            stream=True,
        )

        print('GPT: ', end='')  # Print "GPT: " before the streamed response
        # Process the streamed response
        for chunk in response:
            # Safely check if content is available
            content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ''
            print(content, end='', flush=True)  # Print content as it streams, with no newlines
        print('\n')

        # Optionally, add GPT's response to conversation history
        conversation_history.append({"role": "assistant", "content": content})
        
        most_resembling_photo_file_name = ""

if __name__ == "__main__":
    asyncio.run(chat_with_gpt())
