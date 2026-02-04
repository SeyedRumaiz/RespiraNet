import requests

API_URL = "http://localhost:5000/predict"

def predict_image(image_file):
    """
    Wraps the HTTP POST request.
    Sends the image to flask, get the prediction and return it.

    Args:
        image_file: File object received from streamlit (uploaded_file)
    """
    # Prepare files for POST
    files = {"file": image_file}

    # Send POST request. File travels over HTTP in bytes format
    response = requests.post(API_URL, files=files)

    # Handle errors
    response.raise_for_status()

    # Convert JSON response into Python dictionary
    return response.json()
