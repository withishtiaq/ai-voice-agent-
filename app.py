import os
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# আপনার Google AI Studio থেকে পাওয়া API KEY এখানে দিন
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# মডেল কনফিগারেশন (Math Friendly & Bengali)
generation_config = {
  "temperature": 0.4, # ম্যাথের জন্য কম টেম্পারেচার ভালো (বেশি লজিক্যাল)
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

# System Instruction সেট করা
system_instruction = """
আপনি একজন বন্ধুসুলভ এবং দক্ষ গণিত শিক্ষক। আপনি ব্যবহারকারীর সাথে সাবলীল বাংলায় কথা বলবেন।
গণিতের সমস্যা ধাপে ধাপে সমাধান করবেন। সমীকরণগুলো পড়ার উপযোগী করে লিখবেন।
"""

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash", # Flash মডেলটি ফাস্ট এবং ভয়েস এজেন্টের জন্য ভালো
  generation_config=generation_config,
  system_instruction=system_instruction,
)

# চ্যাট হিস্ট্রি রাখার জন্য একটি সাধারণ ভেরিয়েবল (প্রোডাকশনে ডেটাবেস ব্যবহার করবেন)
chat_session = model.start_chat(history=[])

@app.route('/chat', methods=['POST'])
def chat_with_math_agent():
    try:
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # জেমিনির কাছে মেসেজ পাঠানো
        response = chat_session.send_message(user_message)
        
        # রেসপন্স টেক্সট ক্লিন করা (Markdown সিম্বল সরানো যাতে ভয়েস ভালো শোনায়)
        clean_text = response.text.replace("*", "").replace("#", "")

        return jsonify({
            "reply": clean_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
