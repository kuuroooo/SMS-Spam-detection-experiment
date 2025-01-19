import random
import pandas as pd
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re
import nltk
import idna
from textattack.augmentation import WordNetAugmenter

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)


df = pd.read_csv('datasets/super_sms_dataset.csv', encoding='ISO-8859-1')

stop_words = set(stopwords.words("english"))

# Perturbation Methods

def apply_spacing(text):
    words = text.split()
    perturbed = []
    for word in words:
        if random.random() > 0.6:  # 40% chance to apply spacing
            if len(word) > 2:
                spaced_word = " ".join(list(word))  # Split all characters
                perturbed.append(spaced_word)
            else:
                perturbed.append(word)
        else:
            perturbed.append(word)
    # Randomly add spaces at the beginning or end of the sentence
    if random.random() > 0.5:
        perturbed = [""] + perturbed
    if random.random() > 0.5:
        perturbed += [""]
    return " ".join(perturbed)

SPAM_KEYWORDS = {
    "win": ["achieve", "earn", "secure", "gain", "attain", "obtain", "collect"],
    "free": ["complimentary", "gratis", "no cost", "without charge", "on the house", "zero cost"],
    "money": ["cash", "funds", "capital", "currency", "dollars", "income", "profits"],
    "urgent": ["important", "critical", "vital", "essential", "imperative", "pressing"],
    "click": ["tap", "press", "hit", "select", "choose", "touch"],
    "offer": ["proposal", "deal", "discount", "promotion", "sale", "bargain"],
    "prize": ["award", "reward", "gift", "trophy", "jackpot", "benefit"],
    "gift": ["present", "reward", "souvenir", "offering", "bonus", "handout"],
    "win": ["achieve", "secure", "earn", "claim", "grab", "snatch"],
    "urgent": ["important", "critical", "vital", "pressing", "imperative", "crucial"],
    "deal": ["offer", "bargain", "discount", "arrangement", "promotion", "transaction"],
    "buy": ["purchase", "acquire", "order", "procure", "invest in", "obtain"],
    "discount": ["offer", "rebate", "markdown", "sale", "promotion", "deal"],
    "credit": ["loan", "funds", "money", "capital", "balance", "currency"],
    "claim": ["retrieve", "obtain", "grab", "collect", "earn", "secure"],
    "account": ["profile", "login", "membership", "username", "credentials", "user"],
    "lottery": ["jackpot", "sweepstakes", "prize draw", "raffle", "contest", "game"],
    "verify": ["check", "confirm", "validate", "authenticate", "prove", "ensure"],
    "update": ["upgrade", "refresh", "renew", "revise", "improve", "enhance"],
    "password": ["passcode", "key", "security code", "login credentials", "PIN", "access code"],
    "subscribe": ["sign up", "enroll", "register", "join", "opt in", "follow"],
    "unlimited": ["endless", "boundless", "infinite", "unrestricted", "limitless", "all you can"],
    "save": ["reduce", "cut back", "economize", "conserve", "spare", "set aside"],
    "win": ["claim", "achieve", "secure", "earn", "grab", "collect"],
    "cash": ["money", "funds", "currency", "capital", "notes", "profit"],
    "urgent": ["important", "vital", "imperative", "critical", "pressing", "crucial"],
    "free": ["complimentary", "gratis", "zero cost", "on the house", "no charge"],
    "limited": ["exclusive", "restricted", "rare", "short-term", "finite", "special"],
    "important": ["urgent", "critical", "essential", "vital", "noteworthy", "pressing"],
    "reward": ["prize", "gift", "benefit", "bonus", "perquisite", "return"],
    "jackpot": ["lottery", "prize", "windfall", "reward", "grand prize", "pot"],
    "alert": ["notification", "warning", "notice", "reminder", "heads-up", "signal"],
    "new": ["latest", "updated", "fresh", "recent", "modern", "current"],
    "limited-time": ["short-term", "exclusive", "temporary", "restricted", "special offer", "flash deal"],
    "congratulations": ["kudos", "felicitations", "well done", "applause", "celebration", "cheers"],
    "winner": ["victor", "champion", "prize holder", "awardee", "top scorer", "titleholder"],
    "now": ["immediately", "instantly", "right away", "at this moment", "straightaway", "this instant"],
    "today": ["now", "this day", "as of now", "current day", "present day", "on this date"],
    "improve": ["boost", "enhance", "increase", "upgrade", "advance", "elevate"],
    "secure": ["protect", "lock", "safeguard", "ensure", "fortify", "shield"],
    "guarantee": ["promise", "assurance", "commitment", "pledge", "warranty", "certify"],
    "convenient": ["easy", "simple", "handy", "accessible", "user-friendly", "effortless"],
    "special": ["unique", "exclusive", "particular", "rare", "one-of-a-kind", "exceptional"]
}

def synonym_replacement(text):
    words = text.split()
    perturbed = []
    for word in words:
    
        if word.lower() in SPAM_KEYWORDS and random.random() < 0.9:
            replacement = random.choice(SPAM_KEYWORDS[word.lower()])
      
            if random.random() < 0.5:  
                replacement += " that you can't miss" if random.random() < 0.5 else " for a limited time only"
            
            perturbed.append(replacement)
        else:
            perturbed.append(word)
    
    # Add additional obfuscation by manipulating spacing and adding irrelevant words
    obfuscated_text = []
    for word in perturbed:
        # Randomly split a word into two parts with a space
        if len(word) > 5 and random.random() > 0.7:
            insert_pos = random.randint(1, len(word) - 1)
            word = word[:insert_pos] + " " + word[insert_pos:]
        
        # Insert distractor words 
        if random.random() > 0.8:
            distractors = ["note:", "special", "important", "read this", "alert!"]
            obfuscated_text.append(random.choice(distractors))
        
        obfuscated_text.append(word)
    
    return " ".join(obfuscated_text)

def apply_magic_word(text):
    distractors = [
    "Hello, how are you?", 
    "Family update:", 
    "Let's discuss this later.", 
    "Meeting scheduled for tomorrow", 
    "Hope you're doing well!", 
    "Call me when you're free.", 
    "Looking forward to catching up.", 
    "Have you seen the latest news?", 
    "Important reminder:", 
    "Don't forget about our appointment.", 
    "Good morning, how's your day going?", 
    "Please let me know if you need anything.", 
    "Talk to you soon!", 
    "Happy birthday!", 
    "Let’s reschedule the meeting.", 
    "Can we talk later today?", 
    "Thanks for your help!", 
    "Did you get my last message?", 
    "Please confirm your availability.", 
    "Meeting postponed to next week.", 
    "Can you send me the documents?", 
    "Just checking in.", 
    "Let's grab lunch this week.", 
    "Let me know your thoughts.", 
    "I’ll call you back later.", 
    "It was great catching up with you!", 
    "Hope you had a good weekend.", 
    "Don’t forget to reply to this email.", 
    "Can you follow up on this?", 
    "Let me know if this works for you.", 
    "Can we finalize the details?", 
    "I’ll share the updates soon.", 
    "Thanks for letting me know!", 
    "Let’s discuss this in our next call.", 
    "When can we meet?", 
    "Don’t worry, take your time.", 
    "Please forward this to the team.", 
    "Hope everything is going well!", 
    "Let’s plan the trip details.", 
    "Looking forward to your response.", 
    "Please confirm receipt of this email.", 
    "Do you have time for a quick chat?", 
    "Let’s connect soon.", 
    "Can we move the deadline?", 
    "Please send me your feedback.", 
    "Have a great day!", 
    "What time works best for you?", 
    "I’ll send you the details shortly.", 
    "When are you available?", 
    "Thanks for your quick response!", 
    "Apologies for the delay in responding.", 
    "Please update me on this.", 
    "Let’s touch base tomorrow.", 
    "Looking forward to our meeting.", 
    "Can we reschedule our call?", 
    "Don’t forget to check this out.", 
    "Thanks for the update!", 
    "Hope to hear from you soon.", 
    "I’ll get back to you shortly.", 
    "Have a safe trip!", 
    "Wishing you a speedy recovery.", 
    "Let me know if you need help.", 
    "Thank you for your patience.", 
    "Please send me the final version.", 
    "When’s the best time to call you?", 
    "Just a quick reminder.", 
    "Thank you for following up.", 
    "Please confirm the changes.", 
    "I’ll keep you posted on this.", 
    "Hope you’re having a productive day!", 
    "Let me know when you’re free.", 
    "Please share your thoughts on this.", 
    "Can you check on this for me?", 
    "Let’s coordinate on this.", 
    "Looking forward to hearing from you!", 
    "Good afternoon, how can I help?", 
    "Please prioritize this request.", 
    "Do you have any updates for me?", 
    "Let’s plan for next week.", 
    "Thank you for letting me know!", 
    "Hope you’re enjoying your day.", 
    "Can we discuss this tomorrow?", 
    "Don’t forget to update me on this.", 
    "Please let me know if you have questions."
]
    selected_phrase = random.choice(distractors)
    words = text.split()
    insert_pos = random.randint(0, len(words))  
    words.insert(insert_pos, selected_phrase)
    if random.random() > 0.5:
        words.insert(0, "Hi,")
    return " ".join(words)

def apply_typos(text):
    typo_map = {"o": "0", "e": "3", "a": "@", "s": "$"}
    perturbed = []
        
    for char in text:
        if random.random() > 0.7:  
            if random.random() > 0.5:
                # Replace with a typo
                perturbed.append(typo_map.get(char.lower(), char))
            else:
                # Insert a random character
                perturbed.append(char + random.choice("xyz"))
        else:
            perturbed.append(char)
    
    # Random deletions
    perturbed = [char for char in perturbed if random.random() > 0.1]
    
    return "".join(perturbed)

def perturb_text(text, method):
    if method == "spacing":
        return apply_spacing(text)
    elif method == "synonym":
        return synonym_replacement(text)
    elif method == "typos":
        return apply_typos(text)
    elif method == "magic_word":
        return apply_magic_word(text)
    return text

# Apply Perturbation to Spam Messages
def augment_data(df, method):
    augmented_rows = []
    for i, row in df.iterrows():
        if row["Labels"] == 1:  # Spam message
            perturbed_message = perturb_text(row["SMSes"], method)
            augmented_rows.append({"SMSes": perturbed_message, "Labels": row["Labels"]})
        else:
            augmented_rows.append({"SMSes": row["SMSes"], "Labels": row["Labels"]})
    return pd.DataFrame(augmented_rows)

# Generate Augmented Datasets for Each Perturbation Method
methods = ["spacing", "synonym", "typos", "magic_word"]

for method in methods:
    augmented_df = augment_data(df, method)
    augmented_df.to_csv(f"datasets/{method}_dataset.csv", index=False)

# Generate Mixed Attack Dataset
def augment_data_mixed(df, methods):
    augmented_rows = []
    for i, row in df.iterrows():
        if row["Labels"] == 1:  # Spam message
            method = random.choice(methods)
            perturbed_message = perturb_text(row["SMSes"], method)
            augmented_rows.append({"SMSes": perturbed_message, "Labels": row["Labels"]})
        else:
            augmented_rows.append({"SMSes": row["SMSes"], "Labels": row["Labels"]})
        # Debug: Print progress
        if i % 100 == 0:
            print(f"Processed {i} rows for mixed methods")
    return pd.DataFrame(augmented_rows)

mixed_augmented_df = augment_data_mixed(df, methods)
mixed_augmented_df.to_csv("datasets/mixed_dataset.csv", index=False)

print("Improved augmented datasets have been saved successfully!")