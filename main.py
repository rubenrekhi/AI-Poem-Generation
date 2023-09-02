import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

## HELPER FUNCTION DECLERATIONS ----------------------
# sample function from the official keras tutorial
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

def generator(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1) # get a start index in the text and "steal" the first 40 characters
    # the function will now aim to create text using the predictions for the next characters using the first 40 characters of shakespeares actual work
    creation = ""
    innit_text = text[start_index: start_index + SEQ_LENGTH]
    creation += innit_text
    for i in range (length):
        x = np.zeros((1, SEQ_LENGTH, len(char_occ)), dtype=bool)
        for k, char in enumerate(innit_text):
            x[0, k, char_index[char]] = 1
            
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature) 
        next_char = index_char[next_index]
        creation += next_char
        innit_text = innit_text[1:] + next_char
        
    return creation
## --------------------------------------------------

## main program loop
while True:

    TOGGLE = int(input("Welcome to the Shakespeare poem generator. Select an option:\n\n1.  Generate a new model\n2.  Use the existing model\n\nEnter only the number. Note that option 1 may take some time.\n"))

    # get the data to train the model (compilation of shakespeare poems)
    filepath = 'shakespeare.txt'

    text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
    # using only lowercase letters since only having the option of lowercase letters and not uppercase reduces the possible choices and increases accuracy

    # set that contains all the characters that have appeared in the text
    char_occ = sorted(set(text))

    # create a dictionary with a numeric value for each character that shows up in the data
    char_index = {}
    for index, character in enumerate(char_occ):
        char_index[character] = index
    
    # create the opposite dictionary as well (index -> char)
    index_char = {}
    for index, character in enumerate(char_occ):
        index_char[index] = character

    # Set up the training data


    if TOGGLE == 1:
        SEQ_LENGTH = int(input("Input the sequence length that you would like to use to build the model: "))
        STEP_SIZE = int(input("Input the step size that you would like to use to build the model: "))
        file = open('memory.txt', 'wt')
        file.write("{}\n{}".format(SEQ_LENGTH, STEP_SIZE))

    else:
        try:
            file = open('memory.txt', 'rt')
        except:
            print("The model has not been defined. Please try again.")
            continue
        SEQ_LENGTH = int(file.readline())
        STEP_SIZE = int(file.readline())

    sentences = [] # feature
    next_char = [] # target

    for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
        sentences.append(text[i:i+SEQ_LENGTH]) # make a list of sentences with 100 characters, 5 characters apart
        next_char.append(text[i+SEQ_LENGTH]) # list of the characters that are after each sentence (the model is supposed to predic these)
        
    # convert the training data into numerical format using numpy arrays

    # create a 3 dim array with one dimension for the number of sentences, one dimention for all the positions in these sentences, 
    # one dimention for all the possible characters that can be in that sentence[position] 
    # if a character appears at sentence x, position y, and it is character z, then that position will be true and every other character
    # for that sentence and position will be false
    x = np.zeros((len(sentences), SEQ_LENGTH, len(char_occ)), dtype = bool)
    for sentence_pos, sentence in enumerate(sentences):
        for char_pos, character in enumerate(sentence):
            x[sentence_pos,char_pos, char_index[character]] = True


    # create a 2 dim array for the target data that has the length (amount) of sentences, and the possible characters, and the true 
    # character will be the one that comes next after that particular sentence, and the rest will be false
    y = np.zeros((len(sentences), len(char_occ)), dtype=bool)
    for sentence_pos, char in enumerate(next_char):
        y[sentence_pos, char_index[char]] = True
        
    # Now the training data has been fully set up, and I will create the neural network (if the user chooses to make a new model)
    if TOGGLE == 1:
        model=Sequential()
        model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(char_occ))))
        model.add(Dense(len(char_occ)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
        model.fit(x, y, batch_size=256, epochs=4)
        model.save('writepoems.model')    

    else:
        # load the model that has been created
        # do not want to create a model every time the code is run, so I instead saved it
        try:
            model = tf.keras.models.load_model('writepoems.model')
        except:
            print("The model has not been defined. Please try again.")
            continue
        
    len_req = int(input("Input the number of characters you want your text to be. The first {} characters will be original text from Shakespeare.: ".format(SEQ_LENGTH)))
    temp_req = float(input("Enter the temperature you would like to use: "))
    print("----------{}----------".format(temp_req))
    print(generator(len_req, temp_req))
    redo = input("Would you like to generate a new poem? Enter Y/N: ")
    if redo=="Y":
        continue
    else:
        break
