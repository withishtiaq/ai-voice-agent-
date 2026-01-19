import os
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# --- API Key সেটআপ ---
# Render এর Environment Variable থেকে Key লোড হবে
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables!")
else:
    genai.configure(api_key=api_key)

# --- কনফিগারেশন ---
generation_config = {
  "temperature": 0.4,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

# System Instruction
system_instruction = """
আপনি একজন বন্ধুসুলভ এবং দক্ষ গণিত শিক্ষক (Math Tutor)।
১. আপনি ব্যবহারকারীর সাথে সাবলীল বাংলায় কথা বলবেন।
২. গণিতের জটিল বিষয়গুলো সহজে ধাপে ধাপে ব্যাখ্যা করবেন।
৩. যদি কোনো গাণিতিক সমীকরণ (Equation) থাকে, তবে সেটি এমনভাবে বর্ণনা করবেন যাতে ভয়েস অ্যাসিস্ট্যান্ট বা TTS (Text-to-Speech) সেটি সহজে পড়ে শোনাতে পারে। (যেমন: "x^2" না লিখে বলবেন "x স্কয়ার")।
৪. উত্তরগুলো খুব বেশি বড় করবেন না, কথোপকথন বা চ্যাটের মতো ছোট রাখুন।
"""

# --- মডেল সিলেকশন (আপনার অনুরোধ অনুযায়ী) ---
model_name = "gemini-1.5-flash-latest"

try:
    model = genai.GenerativeModel(
      model_name=model_name,
      generation_config=generation_config,
      system_instruction=system_instruction,
    )
    chat_session = model.start_chat(history=[])
    print(f"Success: Model loaded using {model_name}")
except Exception as e:
    print(f"Error loading model {model_name}: {e}")
    model = None

# --- API রাউট ---

@app.route('/', methods=['GET'])
def home():
    # হোম পেজে দেখাবে কোন মডেল ব্যবহার করা হচ্ছে
    return f"Math AI Agent running with: {model_name}. Use /chat to talk."

# ডিবাগিং রাউট: এটি চেক করবে আপনার API Key তে কোন কোন মডেল পারমিশন আছে
@app.route('/models', methods=['GET'])
def list_models():
    try:
        available_models = []
        if api_key:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        return jsonify({
            "current_target_model": model_name,
            "available_models_for_your_key": available_models
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/chat', methods=['POST'])
def chat_with_math_agent():
    global chat_session
    try:
        # মডেল লোড না হলে এরর দেবে
        if not model:
            return jsonify({
                "error": f"Model {model_name} failed to initialize. Check /models endpoint."
            }), 500

        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # মেসেজ পাঠানো
        response = chat_session.send_message(user_message)
        
        # ক্লিন টেক্সট
        clean_text = response.text.replace("*", "").replace("#", "")

        return jsonify({
            "reply": clean_text
        })

    except Exception as e:
        # সেশন এরর হলে নতুন সেশন দিয়ে ট্রাই করবে
        try:
             chat_session = model.start_chat(history=[])
             response = model.generate_content(user_message)
             clean_text = response.text.replace("*", "").replace("#", "")
             return jsonify({"reply": clean_text})
        except Exception as inner_e:
             return jsonify({"error": str(e), "detail": str(inner_e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
