import os
import google.genai as genai

def generate_personalized_guidance(student_text, emotion, confidence, api_key=None):
    """Generates empathetic guidance using Gemini AI, with a simulated fallback if no API key is set."""
    
    # Check if API key is provided, else fallback to environment variable
    key_to_use = api_key or os.environ.get("GEMINI_API_KEY")
    
    if not key_to_use:
        # Provide a beautiful simulated response as a fallback to make the app functional without a key
        return get_simulated_response(student_text, emotion, confidence)
        
    try:
        # Initialize Gemini client with the provided API key
        client = genai.Client(api_key=key_to_use)
        chat = client.chats.create(model="gemini-2.0-flash")
        
        prompt = f"""
You are an empathetic, highly skilled Academic Coach and Teaching Assistant. 
A student has shared a challenge they are facing with their studies:
"{student_text}"

Our emotion classification pipeline has detected that the student's current state is predominantly: **{emotion}** (Model Confidence: {confidence:.1%}).

Please write a personalized, highly supportive, and actionable response in Markdown format. 

Follow these design principles:
1. **Empathy & Validation (First Paragraph)**: Genuinely validate their emotion. Acknowledge what they are feeling (e.g., if Frustrated, validate that debugging or setups are exhausting; if Confused, assure them that feeling lost is an essential part of learning; if Bored, suggest why this topic is secretly fascinating).
2. **Concept & Troubleshooting Guidance (Second Paragraph)**: Briefly explain the core of their topic or suggest a troubleshooting method (e.g., if they mention recursion, explain base/recursive cases; if compiler errors, suggest reading line numbers).
3. **Actionable Next Steps (Bulleted List)**: Provide 2-3 concrete steps they can do right now to move forward.
4. **Encouragement**: End with a warm, positive closing remark.

Keep the response structured, clear, and around 150-250 words. Do not sound robotic.
"""
        response = chat.send_message(prompt)
        return response.candidates[0].content if response.candidates else get_simulated_response(student_text, emotion, confidence)
    except Exception as e:
        print(f"Gemini API invocation failed: {e}")
        return f"**[AI Assistant Error]** Could not communicate with Gemini AI. Error details: `{str(e)}`.\n\n*Below is a simulated response based on the detected emotion:*\n\n" + get_simulated_response(student_text, emotion, confidence)

def get_simulated_response(student_text, emotion, confidence):
    """Provides a realistic, high-quality response template based on the emotion if Gemini API key is unavailable."""
    
    responses = {
        "Bored": """
### 💡 Shake Things Up!
It's completely normal to feel **Bored** when working on repetitive exercises. Often, textbooks present concepts in a very dry way, making it hard to see the big picture.

**Let's look at this differently:**
Instead of just memorizing the syntax or formulas, think about how this is applied in the real world. For example, did you know that the algorithms you're studying are used to power Netflix recommendations or control flight patterns? Finding a cool project can immediately turn boredom into excitement!

**Here are your next steps:**
*   **Find a Real Application:** Google how the topic you're studying is used in gaming, finance, or space exploration.
*   **Build a Mini-Project:** Instead of solving abstract problems, write a small script that does something fun (e.g., a simple text game or a web scraper).
*   **Take a Quick Break:** Step away from the screen for 5 minutes, stretch, and come back with a fresh perspective.

*Remember: Every expert developer and scientist had to learn the basics. You can do this!*
""",
        "Confident": """
### 🚀 Keep Up the Excellent Work!
Fantastic job! You're feeling **Confident**, and it's clear you've got a solid grasp of this material. Celebrate this win—when things click, it means your hard work is paying off.

**Here is how you can push yourself further:**
To truly cement your knowledge, try to explain it to someone else or explore more advanced concepts. Teaching is the ultimate test of understanding!

**Here are your next steps:**
*   **Optimize Your Solution:** Can you make your code run faster or use less memory? Look into Big-O notation.
*   **Explain the Concept:** Try explaining what you just did to a peer, a TA, or even summarize it in writing.
*   **Try a Harder Challenge:** If you're coding, head over to LeetCode or HackerRank and try a medium/hard problem on this topic.

*You're on a roll! Keep feeding that momentum.*
""",
        "Confused": """
### 🔍 Take a Deep Breath – Confused is Where Learning Begins!
It is absolutely okay to feel **Confused**. In fact, confusion is the feeling of your brain actively building new neural pathways to understand something new!

**Let's break this down:**
When a concept (like recursion or database joins) feels overwhelming, it's usually because we're trying to digest too much at once. Let's simplify the problem by isolating the smallest piece that you *do* understand.

**Here are your next steps:**
*   **Isolate the Core:** Write down the exact line of code or step in the math equation where you get lost.
*   **Use Visuals:** Draw a diagram or trace variables on a piece of paper. Visualizing recursion trees or table relations helps immensely.
*   **Ask a Specific Question:** Frame your confusion as: *"I understand X, but when Y happens, I don't see why it equals Z."*

*Stay patient with yourself. You're closer to understanding this than you think!*
""",
        "Curious": """
### 🌟 Love the Curiosity!
It's wonderful that you're feeling **Curious**! Asking "why" and wanting to know what happens under the hood is what separates good developers and engineers from great ones.

**Let's explore this:**
When you're curious about how a system scales or why an algorithm works, the best approach is hands-on experimentation. Modify parameters, read source code, and run benchmarks.

**Here are your next steps:**
*   **Experiment:** Change parameters in your code or mathematical models and observe the exact outputs.
*   **Read Documentation:** Check the official developer docs or academic papers to see how standard libraries implement these functions.
*   **Deep Dive:** Write a blog post or markdown file summarizing your findings—it's a great addition to a portfolio!

*Never stop asking questions. Your curiosity will take you very far!*
""",
        "Frustrated": """
### 🛠️ We've All Been There – Let's Tackle This Together!
I hear you, and your **Frustration** is 100% valid. Getting stuck on environment setups, compilation bugs, or logic errors is incredibly draining and can make you want to throw your laptop.

**Let's debug this systematically:**
When frustration peaks, our ability to think logically declines. The best thing you can do right now is step back, clear your mind, and approach the bug from a clean slate.

**Here are your next steps:**
*   **Step Away:** Take a 10-minute break. Don't look at the screen. Walk, drink water, or stretch.
*   **Print Debugging:** Add simple print statements or use a debugger to check the exact state of variables right before the crash.
*   **Rubber Ducking:** Explain your code or problem out loud (to a roommate, a pet, or even a rubber duck) line-by-line. You'll often spot the error yourself!

*Don't give up. The feeling of relief when you finally fix this bug will be absolutely worth it!*
"""
    }
    
    sim_msg = f"""
> ⚠️ **[Demo Mode]** Generating simulated response because no Gemini API key is configured. You can input your API key in the sidebar or Settings tab to enable live AI responses.
    
{responses.get(emotion, responses['Confused'])}
"""
    return sim_msg
