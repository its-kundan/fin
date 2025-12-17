# chatbot_session.py - Stateful Gemini Chatbot
import os
import sys
from typing import List
from pathlib import Path
from PIL import Image
from google.genai import types

# --- 1. Import necessary components from insights.py ---
try:
    # Temporarily modify path to import local modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Import the setup function and the EnhancedInsightGenerator class for the chart directory
    from insights import setup_gemini, EnhancedInsightGenerator
    
    # Use the hardcoded path from EnhancedInsightGenerator for consistency
    CHART_DIRECTORY = Path(EnhancedInsightGenerator.CHART_DIR_1)
    
except ImportError as e:
    print(f"FATAL: Could not import from local files (state.py or insights.py).")
    print(f"Error: {e}")
    print("Ensure insights.py is in the same directory.")
    sys.exit(1)


def find_and_load_charts() -> List[Image.Image]:
    """
    Scans the primary chart directory for image files and loads them as PIL Image objects.
    """
    loaded_images = []
    
    if not CHART_DIRECTORY.is_dir():
        print(f"\n--- ❌ ERROR: Primary chart directory not found at: {CHART_DIRECTORY} ---")
        return []

    print(f"Scanning and loading images from: {CHART_DIRECTORY}")
    
    # Check for common image extensions
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for path in CHART_DIRECTORY.glob(ext):
            try:
                img = Image.open(path)
                loaded_images.append(img)
            except Exception as load_e:
                print(f"Warning: Could not load image {path}: {load_e}")

    if not loaded_images:
        print("--- ⚠️ WARNING: No usable charts found. ---")

    return loaded_images


def run_chatbot():
    """
    Main function to run the Gemini-only interactive chatbot using a persistent chat session.
    """
    print("--- 🤖 Starting Stateful Gemini Insight Chatbot Setup ---")

    # --- Setup API Key and Client ---
    gemini_client, gemini_model = setup_gemini()
    
    if not gemini_client:
        print("\nFATAL: Gemini client initialization failed. Exiting.")
        return
        
    # --- 1. Load Charts ONCE ---
    loaded_images = find_and_load_charts()
    
    if not loaded_images:
        print("Cannot run chatbot without charts to analyze. Exiting.")
        return

    # --- 2. Define System Context (Replaces old report_content) ---
    system_instruction = (
        "You are an expert business analyst and visual data interpreter. "
        "The attached charts are the primary and only source of information for this conversation. "
        "Your task is to answer the user's queries by analyzing the visual data in the images. "
        "Base your answers solely on the provided charts. Be concise, clear, and data-driven. "
        "Maintain the context of previous questions and answers in this chat session."
    )
    
    # --- 3. Start the Persistent Chat Session (The key step) ---
    print("\n--- 💬 Starting Persistent Chat Session (Sending initial context) ---")
    
    # The first message sets the context, including the images, the system prompt, and the first "user" turn.
    initial_prompt = [
        # System Instruction
        system_instruction, 
        
        # All the loaded images
        *loaded_images, 
        
        # A clear prompt indicating the images are the primary context
        "I have attached 14 business charts. Please confirm you have received them and are ready to answer questions based on the visual data in these charts."
    ]
    
    try:
        # Create a new chat session with the model
        chat = gemini_client.chats.create(model=gemini_model)
        
        # Send the large context ONCE. The response is just confirmation.
        print(f"Sending {len(loaded_images)} images and context to start the session...")
        response = chat.send_message(initial_prompt)
        
        print("\n--- ✅ Gemini Chat Initialized ---")
        print(f"Confirmation: {response.text.strip()}")
        
    except Exception as e:
        print(f"\nFATAL: Failed to start chat session or send initial context: {e}")
        return


    # --- 4. Start Interactive Chat Loop ---
    print("\n--- ❓ Start Asking Questions ---")
    print("The charts are now permanently loaded for this session.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            user_query = input("\nYour Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if user_query.lower() in ['exit', 'quit']:
            print("\n👋 Thank you for using the Gemini Insight Chatbot. Goodbye!")
            break
        
        if not user_query:
            continue

        print("\nThinking... (Using persistent chat session)...")
        
        try:
            # Use the chat object to send the message. This only sends the new text query!
            response = chat.send_message(user_query)
        
            print("\n--- 🧠 CHATBOT RESPONSE ---")
            print(response.text.strip())
            print("---------------------------")
            
        except Exception as e:
            print(f"\n--- ❌ ERROR DURING CHAT ---")
            print(f"An error occurred: {str(e)}")
            print("The session may need to be restarted.")


if __name__ == "__main__":
    run_chatbot()

