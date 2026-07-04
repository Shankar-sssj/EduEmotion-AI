import os
import pandas as pd

def generate_synthetic_dataset():
    data = []
    
    # Emotional categories and common expressions
    patterns = {
        "Bored": [
            "This is so dry and boring, I can't stay focused.",
            "Why do we need to learn this? It has no real-world use.",
            "I am falling asleep reading these slides.",
            "This topic is completely uninteresting to me.",
            "I don't care about this formula, it's just memorization.",
            "This lecture feels like it's going on forever.",
            "I'm just staring at the screen, totally unengaged.",
            "None of this seems relevant to what I actually want to do.",
            "I find this subject incredibly dull.",
            "Why is this module so tedious?",
            "I am losing interest in this class rapidly.",
            "This is just repetitive busywork.",
            "I'm bored out of my mind with these programming exercises.",
            "This math proof is so dry, I can't bring myself to study it.",
            "I've lost all motivation to complete this reading list."
        ],
        "Confident": [
            "I completely understand how this algorithm works now!",
            "This makes perfect sense, it's very logical.",
            "I solved all the practice problems on my first try.",
            "I feel fully prepared for the upcoming exam.",
            "This concept is very easy for me to grasp.",
            "I can explain this topic to my classmates without any issues.",
            "I've got a solid handle on this material.",
            "The lab was straightforward and I finished it early.",
            "I'm confident I can write this script from scratch.",
            "This is clicking for me perfectly.",
            "I know exactly how to debug this issue now.",
            "I have mastered this chapter already.",
            "This assignment is going to be a breeze.",
            "I understand the theory and the application completely.",
            "I've got this, no extra help needed."
        ],
        "Confused": [
            "I am completely lost on how recursion works.",
            "What does this equation actually represent?",
            "I don't understand the difference between these two terms.",
            "Can you explain this logic step-by-step again?",
            "I don't get why we use a pointer here.",
            "The lecture explanation went completely over my head.",
            "I'm struggling to see how these two concepts connect.",
            "I don't know where to start with this coding challenge.",
            "The instructions for this project are very vague.",
            "I am baffled by this error message, what does it mean?",
            "Could someone clarify the difference between inheritance and polymorphism?",
            "I'm having trouble understanding how to calculate this limit.",
            "Why did the value change in this step of the calculation?",
            "I don't understand how to set up the base case here.",
            "I'm confused about when to use a list vs a dictionary."
        ],
        "Curious": [
            "How does this library optimize search queries under the hood?",
            "Is there a more efficient way to solve this problem?",
            "Can we apply this machine learning model to financial data?",
            "What happens if I change the learning rate to a very high value?",
            "I'd love to learn more about the history of this theorem.",
            "Are there any real-world applications of quantum computing in this field?",
            "I want to explore how this algorithm behaves with larger datasets.",
            "What is the mathematical proof behind this approximation?",
            "How does the browser parse this specific style attribute?",
            "Can we combine these two design patterns in a single project?",
            "I'm interested in reading the original research paper on this.",
            "How do top companies handle this scaling challenge?",
            "What is the difference between this implementation and the industry standard?",
            "I wonder if we can use recursion instead of iteration here to make it cleaner.",
            "How does memory management work for this data structure?"
        ],
        "Frustrated": [
            "I've been stuck on this bug for five hours and I want to quit.",
            "This code is throwing errors and nothing I do fixes it.",
            "This compiler error makes absolutely no sense to me!",
            "I'm so annoyed with this assignment, it's taking way too long.",
            "Nothing is working and I feel like pulling my hair out.",
            "Why is this tool so hard to configure and set up?",
            "I keep getting a syntax error but my syntax looks completely fine.",
            "This is incredibly frustrating, I've restarted three times.",
            "I'm about to give up on this project entirely.",
            "I've read the documentation ten times and I still don't get it.",
            "This environment setup is a nightmare, nothing is installing.",
            "My code works on my local machine but fails on the autograder!",
            "I am so sick of dealing with these merge conflicts.",
            "This database connection keeps timing out and it's driving me crazy.",
            "I've tried everything and it's still not working."
        ]
    }
    
    # Let's expand these statements with some sentence templates to create a larger dataset
    subjects = [
        "programming", "calculus", "machine learning", "data structures", 
        "python", "physics", "web development", "databases", "algorithms", 
        "statistics", "organic chemistry", "linear algebra"
    ]
    
    fillers = {
        "Bored": [
            "Honestly, {subject} is so boring.",
            "I can't pay attention to {subject} lectures at all.",
            "Why do we study {subject}? It's just pointless memorization.",
            "Struggling to stay awake while studying {subject}.",
            "This {subject} textbook is the most tedious thing I've ever read."
        ],
        "Confident": [
            "I feel really good about my understanding of {subject}.",
            "I solved all the {subject} homework problems easily.",
            "I'm ready to ace this {subject} test.",
            "Understanding {subject} has been super smooth for me.",
            "I can confidently implement the core concepts of {subject}."
        ],
        "Confused": [
            "I'm completely lost when it comes to {subject}.",
            "Can someone help explain {subject}? I don't understand it.",
            "I don't know what is going on in {subject} class.",
            "The concept of {subject} is very confusing to me.",
            "I'm stuck on {subject} basics, I need a tutor."
        ],
        "Curious": [
            "I really want to dive deeper into {subject} applications.",
            "I'm curious about how {subject} works in the industry.",
            "How does {subject} relate to real-world software systems?",
            "I want to know more about the advanced topics in {subject}.",
            "Are there any projects I can build to master {subject}?"
        ],
        "Frustrated": [
            "I'm so frustrated with this {subject} homework.",
            "I've been trying to solve this {subject} issue for hours.",
            "This {subject} lab is absolute torture.",
            "I keep failing the tests for {subject} and I'm losing my mind.",
            "Nothing makes sense in {subject} and it's extremely annoying."
        ]
    }
    
    # Add base sentences
    for emotion, texts in patterns.items():
        for text in texts:
            data.append({"text": text, "emotion": emotion})
            
    # Add template-generated sentences to make the dataset larger (around 500 records)
    for subject in subjects:
        for emotion, templates in fillers.items():
            for template in templates:
                text = template.format(subject=subject)
                # Add multiple variations
                data.append({"text": text, "emotion": emotion})
                data.append({"text": text.lower().replace(".", ""), "emotion": emotion})
                data.append({"text": text + "!", "emotion": emotion})
                data.append({"text": "Honestly, " + text[0].lower() + text[1:], "emotion": emotion})
                data.append({"text": "I feel like " + text[0].lower() + text[1:], "emotion": emotion})

    df = pd.DataFrame(data)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Ensure data directory exists
    from utils import DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "student_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated synthetic dataset with {len(df)} samples at {csv_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_dataset()
