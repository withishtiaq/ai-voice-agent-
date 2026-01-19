import os
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# --- কনফিগারেশন ---

# Render এর Environment Variable থেকে API KEY নেওয়া হচ্ছে
# (নিরাপত্তার জন্য এখানে সরাসরি কি (Key) বসাবেন না)
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    # যদি কোনো কারণে কি (Key) না পায়, তবে এরর দেখাবে (লগ-এ)
    print("Error: GOOGLE_API_KEY not found in environment variables!")
else:
    genai.configure(api_key=api_key)

# মডেল কনফিগারেশন (ম্যাথ এবং ভয়েস ফ্রেন্ডলি)
generation_config = {
  "temperature": 0.4, # ম্যাথের জন্য কম টেম্পারেচার ভালো (বেশি লজিক্যাল)
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

# System Instruction: AI-কে বলা হচ্ছে সে কেমন আচরণ করবে
system_instruction = """
আপনি একজন বন্ধুসুলভ এবং দক্ষ গণিত শিক্ষক (Math Tutor)।
১. আপনি ব্যবহারকারীর সাথে সাবলীল বাংলায় কথা বলবেন।
২. গণিতের জটিল বিষয়গুলো সহজে ধাপে ধাপে ব্যাখ্যা করবেন।
৩. যদি কোনো গাণিতিক সমীকরণ (Equation) থাকে, তবে সেটি এমনভাবে বর্ণনা করবেন যাতে ভয়েস অ্যাসিস্ট্যান্ট বা TTS (Text-to-Speech) সেটি সহজে পড়ে শোনাতে পারে। (যেমন: "x^2" না লিখে বলবেন "x স্কয়ার")।
৪. উত্তরগুলো খুব বেশি বড় করবেন না, কথোপকথন বা চ্যাটের মতো ছোট রাখুন।
৫. শুধু উত্তর দেবেন না, কিভাবে সমাধান করা হয়েছে তা বুঝিয়ে বলবেন।
"""

# মডেল ইনিশিলাইজ করা
model = genai.GenerativeModel(
  model_name="gemini-1.5-flash", # ফাস্ট এবং ভয়েস এজেন্টের জন্য পারফেক্ট
  generation_config=generation_config,
  system_instruction=system_instruction,
)

# চ্যাট সেশন শুরু (সিম্পল হিস্ট্রি ম্যানেজমেন্ট)
chat_session = model.start_chat(history=[])

# --- API রাউট ---

@app.route('/', methods=['GET'])
def home():
    return "Math AI Agent is Running! Use /chat endpoint."

@app.route('/chat', methods=['POST'])
def chat_with_math_agent():
    try:
        # রিকোয়েস্ট থেকে ডেটা নেওয়া
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # জেমিনির কাছে মেসেজ পাঠানো
        response = chat_session.send_message(user_message)
        
        # রেসপন্স টেক্সট ক্লিন করা (Markdown সিম্বল সরানো যাতে ভয়েস ভালো শোনায়)
        # যেমন: **Bold** বা ## Header ভয়েসে পড়ার দরকার নেই
        clean_text = response.text.replace("*", "").replace("#", "")

        return jsonify({
            "reply": clean_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # লোকাল পিসিতে টেস্ট করার জন্য
    app.run(debug=True, host='0.0.0.0', port=8080)
