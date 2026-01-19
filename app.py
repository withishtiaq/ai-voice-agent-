import os
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# --- API Key সেটআপ ---
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

# --- মডেল সিলেকশন (আপনার লিস্ট থেকে সেরাটি নেওয়া হয়েছে) ---
# আপনার দেওয়া লিস্ট অনুযায়ী 'gemini-2.0-flash' সবথেকে ভালো অপশন
model_name = "gemini-2.0-flash"

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
    return f"Math AI Agent is running perfectly with: {model_name}"

@app.route('/chat', methods=['POST'])
def chat_with_math_agent():
    global chat_session
    try:
        # মডেল লোড না হলে এরর দেবে
        if not model:
            return jsonify({
                "error": f"Model {model_name} failed to initialize."
            }), 500

        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # মেসেজ পাঠানো
        response = chat_session.send_message(user_message)
        
        # ক্লিন টেক্সট (ভয়েস বা TTS এর পড়ার সুবিধার জন্য)
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
