import time
now = time.ctime()

def chatbot(user_input):
    user_input = user_input.lower()

    if "hi" in user_input or "hello" in user_input:
        return "Hi there! I'm Feroz, your chatbot assistant."
    elif "what is your name" in user_input:
        return "My name is Feroz, your friendly chatbot assistant."
    elif "where are you from" in user_input:
        return "I'm from the digital world, always ready to chat!"
    elif "how are you" in user_input:
        return "I'm doing great! Thanks for asking."
    elif "do you have any hobbies" in user_input or "interests" in user_input:
        return "I'm always busy helping users, so my hobby is chatting with people like you!"
    elif "what did you eat today" in user_input or "what do you like to eat" in user_input:
        return "I don't eat, but I can help you find delicious recipes and food-related information."
    elif "favorite color" in user_input:
        return "I'm a chatbot, so I don't have personal preferences for colors."
    elif "do you enjoy listening to music" in user_input:
        return "I can't listen to music, but I'm here to chat about it!"
    elif "tell me a joke" in user_input or "another joke" in user_input:
        return "Why did the scarecrow win an award? Because he was outstanding in his field!"
    elif "tell me an interesting fact" in user_input:
        return "I don't have an interesting fact right now, but I'll try to come up with one soon!"
    elif "weather in" in user_input:
        return "I'm sorry, I don't have real-time weather data."
    elif "latest news" in user_input:
        return "I'm sorry, I don't have real-time news updates."
    elif "translate" in user_input:
        return "I don't support real-time language translation."
    elif "what is the time now" in user_input:
        return now
    elif "bye" in user_input:
        return "Goodbye! Take care and have a great day!"
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase your sentence?"

print("Feroz: Hi! I'm Feroz, your friendly chatbot assistant!")

while True:
    user_input = input("User: ")
    if user_input.lower() == 'bye':
        print("Feroz: Goodbye! Have a great day!")
        break

    response = chatbot(user_input)
    print("Feroz:", response)

