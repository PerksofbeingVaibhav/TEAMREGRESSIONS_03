from flask import Flask,render_template,request
import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

chats = []

repo = []



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")


def tokk(sentence):

    data = "I can't Understand!"

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                data = random.choice(intent['responses'])
                print(data)

    print(data)            #repo.append(f"{data}")
    return data



app = Flask(__name__)

@app.route("/")
def beginning():
    chats.clear()

    repo.clear()

    return render_template('begin.html')


@app.route("/index.html", methods=["GET","POST"])
def hello ():
    return render_template('/index.html')


@app.route("/get")
#function for the bot response
#def get_bot_response():
#    userText = request.args.get('msg')
#    return str(englishBot.get_response(userText))
def hello_world(name = None):

    sentences = None

    #while True:
        # sentence = "do you use credit cards
        #if request.method == 'POST':
    sentences = request.args.get("msg")
    print(sentences)
        #chats.append( f"{sentences}" )

        #else:
        #    sentences = "Hi!!"
        #    chats.append( f"{sentences}" )


    loop = tokk(sentences)

    return str(loop)
        #return render_template('/index.html',value = sentences, response = loop)

if __name__ == "__main__":
    app.run(debug=True)
