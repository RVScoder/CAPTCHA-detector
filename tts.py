import pyttsx3

def speak_captcha(prediction):
    
    engine = pyttsx3.init()

    speech = []

    for char in prediction:
        if char.isupper():
            speech.append(f"capital {char}")
        elif char.islower():
            speech.append(f"small {char}")
        elif char.isdigit():
            speech.append({
                "0": "zero",
                "1": "one",
                "2": "two",
                "3": "three",
                "4": "four",
                "5": "five",
                "6": "six",
                "7": "seven",
                "8": "eight",
                "9": "nine"
            }[char])
        else:
            speech.append(char)

    engine.say(", ".join(speech))
    engine.runAndWait()