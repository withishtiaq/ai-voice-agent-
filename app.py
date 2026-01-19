import os
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# --- API Key সেটআপ ---
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found!")
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

# --- মডেল সিলেকশন (সবচেয়ে নিরাপদ অপশন) ---
# 'gemini-1.5-flash' হলো ফ্রি টিয়ারের জন্য সবচেয়ে স্টেবল মডেল
model_name = "gemini-1.5-flash"

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
    # যদি 1.5 Flash কাজ না করে, তবে 'gemini-pro' তে ফলব্যাক করবে
    fallback_model = "gemini-pro"
    print(f"Trying fallback model: {fallback_model}")
    try:
        model = genai.GenerativeModel(
            model_name=fallback_model,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        chat_session = model.start_chat(history=[])
        model_name = fallback_model # নাম আপডেট করা হলো
    except Exception as e2:
         model = None
         print(f"Critical Error: No models worked. {e2}")

# --- API রাউট ---

@app.route('/', methods=['GET'])
def home():
    return f"Math AI Agent is active using: {model_name}"

@app.route('/chat', methods=['POST'])
def chat_with_math_agent():
    global chat_session
    try:
        if not model:
            return jsonify({"error": "Model failed to initialize."}), 500

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
        # যদি কোনো কারণে সেশন বা টোকেন এরর দেয়, নতুন করে চেষ্টা করবে
        try:
             chat_session = model.start_chat(history=[])
             response = model.generate_content(user_message)
             clean_text = response.text.replace("*", "").replace("#", "")
             return jsonify({"reply": clean_text})
        except Exception as inner_e:
             # বিশেষ করে 429 এরর হ্যান্ডলিং
             error_msg = str(e)
             if "429" in error_msg:
                 return jsonify({"error": "Too many requests. Please wait a moment."}), 429
             return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
