from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

ps = PorterStemmer()

stop_words = set(stopwords.words('english'))

important_words = {
    'not',
    'no',
    'never',
    "don't",
    "didn't",
    "won't",
    "can't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "without"
}

for word in important_words:
    stop_words.discard(word)

negative_cues = {
    "bad", "worse", "worst", "awful", "terrible", "boring", "hate",
    "waste", "poor", "disappoint", "disappointed", "disappointing",
    "pathetic", "horrible", "trash", "ugly"
}

positive_cues = {
    "good", "great", "amazing", "awesome", "excellent", "love", "best",
    "nice", "fantastic", "wonderful", "brilliant"
}

def preprocess(text):

    text = re.sub(r'<.*?>', '', text)

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [ps.stem(word) for word in words]

    return " ".join(words)

def is_obvious_negative(raw_text):
    text = raw_text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    if not tokens:
        return False

    has_negative = any(token in negative_cues for token in tokens)
    has_positive = any(token in positive_cues for token in tokens)
    return has_negative and not has_positive

@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = ""
    user_input = ""
    prediction_input = ""

    if request.method == 'POST':

        user_input = request.form['feedback']

        processed_text = preprocess(user_input)
        prediction_input = processed_text

        vector_input = cv.transform([processed_text])

        result = model.predict(vector_input)[0]

        if is_obvious_negative(user_input):
            prediction = "Negative Sentiment"
        elif result == 1:
            prediction = "Positive Sentiment"

        else:
            prediction = "Negative Sentiment"

    return render_template(
        'index.html',
        prediction=prediction,
        user_input=user_input,
        prediction_input=prediction_input
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
