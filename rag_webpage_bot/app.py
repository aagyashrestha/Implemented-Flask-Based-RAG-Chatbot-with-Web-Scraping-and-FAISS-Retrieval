# app.py
from flask import Flask, request, jsonify
from rag_chatbot import ask_rag_bot_from_urls

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    urls = data.get("urls", [])
    question = data.get("question")

    if not urls or not question:
        return jsonify({"error": "Please provide both 'urls' and 'question'."}), 400

    try:
        response = ask_rag_bot_from_urls(question, urls)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
