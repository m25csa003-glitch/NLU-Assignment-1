# ==============================================================================
# File Name   : M25CSA003_prob1.py
# Author      : [Your Name Here]
# Roll Number : M25CSA003
# Description : Reggy++ Chatbot (Assignment Problem 1)
#               This chatbot extends a basic regex-based bot to:
#               1. Detect surnames from full names.
#               2. Parse various date formats (numeric & word-based) to calculate age.
#               3. Detect moods including handling typos/spelling mistakes.
# ==============================================================================

import re
from datetime import date

# ------------------------------------------------------------------------------
# Function: get_month_number
# Description: Converts month names (full or short) to their integer equivalent.
#              Returns None if the month name is invalid.
# ------------------------------------------------------------------------------
def get_month_number(month_text):
    if not month_text:
        return None
        
    # Standardize input to lowercase substring for easier matching
    m = month_text.lower()[:3]  # taking first 3 chars handles Jan/January
    
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    return month_map.get(m)

# ------------------------------------------------------------------------------
# Function: normalize_year
# Description: Handles 2-digit years.
#              Assumption: 00-26 -> 2000-2026, 27-99 -> 1927-1999.
# ------------------------------------------------------------------------------
def normalize_year(year):
    if year < 100:
        current_year = date.today().year % 100
        # If year is 00-26 (or current year), assume 2000s
        if year <= current_year + 1: # +1 buffer
            return 2000 + year
        else:
            return 1900 + year
    return year

# ------------------------------------------------------------------------------
# Function: extract_dob_auto
# Description: Iterates through a list of regex patterns to automatically detect
#              the date format without asking the user.
# Returns: (day, month, year) tuple or None if no match found.
# ------------------------------------------------------------------------------
def extract_dob_auto(text):
    text = text.strip().lower()

    # List of patterns to check. 
    # Tuple structure: (Regex Pattern, Order of parts)
    # Order codes: 'dmy' (Day-Month-Year), 'mdy', 'ymd'
    patterns = [
        # 1. Word-based formats (e.g., 15 Jan 2000, Jan 15 2000)
        # Matches: "15 jan 2000", "15-january-00", "15th jan 2000"
        (r"(\d{1,2})(?:st|nd|rd|th)?[\s\-\./]+([a-z]{3,})[\s\-\./,]+(\d{2,4})", "d_mon_y"),
        
        # Matches: "Jan 15 2000", "February 2nd, 1998"
        (r"([a-z]{3,})\s+(\d{1,2})(?:st|nd|rd|th)?,?[\s\-\./,]+(\d{2,4})", "mon_d_y"),

        # 2. Numeric formats with separators (-, /, space, .)
        # CAUTION: 01-02-2000 is ambiguous. We prioritize DMY (Indian standard) over MDY.
        
        # YYYY-MM-DD (ISO format, very safe)
        (r"(\d{4})[\-\./\s]+(\d{1,2})[\-\./\s]+(\d{1,2})", "ymd"),

        # DD-MM-YYYY or DD/MM/YYYY
        (r"(\d{1,2})[\-\./\s]+(\d{1,2})[\-\./\s]+(\d{4})", "dmy"),

        # DD-MM-YY (Two digit year)
        (r"(\d{1,2})[\-\./\s]+(\d{1,2})[\-\./\s]+(\d{2})", "dmy"),
        
        # Special case mentioned in prompt: "ddmm-yy" (e.g., 2509-01)
        (r"(\d{2})(\d{2})\-(\d{2,4})", "dmy") 
    ]

    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            
            try:
                day, month, year = 0, 0, 0
                
                if fmt == "d_mon_y":
                    day = int(groups[0])
                    month = get_month_number(groups[1])
                    year = int(groups[2])
                
                elif fmt == "mon_d_y":
                    month = get_month_number(groups[0])
                    day = int(groups[1])
                    year = int(groups[2])
                
                elif fmt == "ymd":
                    year = int(groups[0])
                    month = int(groups[1])
                    day = int(groups[2])
                
                elif fmt == "dmy":
                    day = int(groups[0])
                    month = int(groups[1])
                    year = int(groups[2])

                # Validate month range for numeric parse
                if month is None or not (1 <= month <= 12):
                    continue # Try next pattern if month is invalid
                
                year = normalize_year(year)
                
                return day, month, year

            except ValueError:
                continue # If conversion fails, try next pattern

    return None

# ------------------------------------------------------------------------------
# Function: calculate_age
# ------------------------------------------------------------------------------
def calculate_age(day, month, year):
    today = date.today()
    age = today.year - year
    # Subtract 1 if birthday hasn't happened yet this year
    if (today.month, today.day) < (month, day):
        age -= 1
    return age

# ------------------------------------------------------------------------------
# Function: extract_surname
# Description: Extracts the last word of a string as the surname.
# ------------------------------------------------------------------------------
def extract_surname(full_name):
    # Regex to find the last word in the string
    # \b indicates word boundary, \w+ is the word
    clean_name = full_name.strip()
    match = re.search(r"(\w+)$", clean_name)
    if match:
        return match.group(1)
    return None

# ------------------------------------------------------------------------------
# Function: detect_mood
# Description: Identifies mood using regex keywords, handling typos.
# ------------------------------------------------------------------------------
def detect_mood(text):
    text = text.lower()
    
    # Dictionary of regex patterns for moods.
    # We use flexible patterns to catch typos (e.g., 'hap.*y' matches 'happy', 'hapy')
    mood_patterns = {
        "anger": r"ang.*r|mad|fur.*i|annoy|frust.*|irrit.*",
        "disgust": r"disgust|gross|yuck|nasty|awful",
        "fear": r"fear|scar.*d|afraid|terr.*|worr.*",
        "happiness": r"hap.*y|joy|glad|excit.*|good|great|amaz.*",
        "sadness": r"sad|depres.*|cry.*|lonel.*|sorrow|unhap.*|low",
        "surprise": r"wow|omg|shock.*|surpris.*|amaz.*"
    }

    for mood, pattern in mood_patterns.items():
        if re.search(pattern, text):
            return mood
    
    return "unknown"

# ------------------------------------------------------------------------------
# Function: main
# ------------------------------------------------------------------------------
def main():
    print("\n" + "="*40)
    print("      REGGY++ : The Regex Chatbot      ")
    print("="*40)

    # --- Step 1: Name Detection ---
    name_input = input("Reggy: Hello! What is your full name?\nYou:   ")
    surname = extract_surname(name_input)
    
    if surname:
        print(f"Reggy: Nice to meet you, Mr./Ms. {surname}!")
    else:
        print("Reggy: Nice to meet you!")

    # --- Step 2: Date Parsing & Age Calculation ---
    print("\nReggy: When were you born? (You can type it any way you like!)")
    dob_input = input("You:   ")
    
    dob = extract_dob_auto(dob_input)
    
    if dob:
        d, m, y = dob
        age = calculate_age(d, m, y)
        print(f"Reggy: I understood that date: {d:02d}-{m:02d}-{y}")
        print(f"Reggy: That makes you {age} years old.")
    else:
        print("Reggy: I'm sorry, I couldn't figure out that date format.")

    # --- Step 3: Mood Detection ---
    print("\nReggy: How are you feeling right now?")
    mood_input = input("You:   ")
    
    mood = detect_mood(mood_input)
    
    responses = {
        "anger": "Reggy: Take a deep breath. It's okay to be upset sometimes.",
        "disgust": "Reggy: Eww! That sounds unpleasant.",
        "fear": "Reggy: You are safe here. Don't worry.",
        "happiness": "Reggy: That's wonderful! Your happiness is contagious!",
        "sadness": "Reggy: I'm sorry to hear that. Sending virtual hugs. <3",
        "surprise": "Reggy: Whoa! I wasn't expecting that either!",
        "unknown": "Reggy: I see. Thanks for sharing that with me."
    }
    
    print(responses[mood])
    
    print("\n" + "="*40)
    print("Reggy: Goodbye!")

if __name__ == "__main__":
    main()