# app.py

import uuid  # For generating unique user IDs
from flask import Flask, request, Response

# Import the function that interacts with the RAG chatbot using provided URLs
from rag_chatbot import ask_rag_bot_from_urls

# Initialize a new Flask web application instance
app = Flask(__name__)

# Define an API endpoint '/ask' that only allows POST requests
@app.route('/ask', methods=['POST'])
def ask():
    # Get JSON data sent in the request body
    data = request.get_json()

    # Extract 'urls' and 'question' from the request payload
    urls = data.get("urls", [])
    question = data.get("question")
    
    # Generate a unique user ID if not provided
    user_id = data.get("user_id") or str(uuid.uuid4())

    # Validate input: if 'urls' or 'question' is missing, return an error response
    if not urls or not question:
        return Response("Please provide 'urls' and 'question'.", status=400)

    try:
        # Call the RAG bot function to get the raw text response
        response_text = ask_rag_bot_from_urls(question, urls, user_id)

        # Return plain text and the assigned/generated user ID in a header
        response = Response(response_text, mimetype='text/plain')
        response.headers["X-User-ID"] = user_id  # Optional: for client to track session
        return response

    except Exception as e:
        # If something goes wrong, return error message with status code 500
        return Response(f"Error: {str(e)}", status=500, mimetype='text/plain')

# Entry point of the Flask app. Runs the server in debug mode for easier troubleshooting during development.
if __name__ == '__main__':
    app.run(debug=True)
