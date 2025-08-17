

# Load document
f = open("/content/data.txt", 'r', errors='ignore')
raw_doc = f.read()

# Preprocessing
raw_doc = raw_doc.lower()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting
greet_inputs = ("hello", "hi", "whassup", "how are you")
greet_responses = ("hi", "hey", "nods", "hi there")

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

# Response generator
def response(user_response):
    robo1_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens + [user_response])  # include query
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]

    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        robo1_response = "I am sorry! I don’t understand you."
        return robo1_response
    else:
        robo1_response = sentence_tokens[idx]
        return robo1_response

# Chat loop
flag = True
print("robo: My name is Robo. I will answer your queries about chatbots. If you want to exit, type 'bye'.")

while flag:
    user_response = input().lower()

    if user_response != 'bye':
        if user_response in ["thanks", "thank you"]:
            flag = False
            print("robo: You are welcome.")
        else:
            if greet(user_response) is not None:
                print("robo: " + greet(user_response))
            else:
                print("robo:", response(user_response))
    else:
        flag = False
        print("robo: Goodbye!")
