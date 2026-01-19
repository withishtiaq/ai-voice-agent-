import os
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# API Key কনফিগারেশন
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found!")
else:
    genai.configure(api_key=api_key)

# মডেল কনফিগারেশন
generation_config = {
  "temperature": 0.4,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

system_instruction = """
আপনি একজন বন্ধুসুলভ এবং দক্ষ গণিত শিক্ষক। আপনি ব্যবহারকারীর সাথে সাবলীল বাংলায় কথা বলবেন।
গণিতের সমস্যা ধাপে ধাপে সমাধান করবেন। সমীকরণগুলো পড়ার উপযোগী করে লিখবেন।
"""

# মডেল ইনিশিলাইজ (নাম আপডেট করা হয়েছে)
model = genai.GenerativeModel(
  model_name="gemini-1.5-flash-001", # <--- এই পরিবর্তনটি জরুরি
  generation_config=generation_config,
  system_instruction=system_instruction,
)

chat_session = model.start_chat(history=[])

@app.route('/', methods=['GET'])
def home():
    return "Math AI Agent is Running!"

@app.route('/chat', methods=['POST'])
def chat_with_math_agent():
    try:
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        response = chat_session.send_message(user_message)
        clean_text = response.text.replace("*", "").replace("#", "")

        return jsonify({
            "reply": clean_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
