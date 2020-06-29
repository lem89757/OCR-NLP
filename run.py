# Ranjit Kathiriya (R00183586)

import random
import sys
import numpy as np
import cv2
import editdistance
import tensorflow as tf

# regular expression libreary
import re
# Alternative of Decontraction libreary(Faced Java issue)
import contractions
from textsearch import TextSearch

import numpy as np
# for calculate weighted levenshtein
from weighted_levenshtein import lev, osa, dam_lev
# nltk libreary for biagrams
from nltk import word_tokenize, ngrams



ExpDir = '/Users/ranjitsmac/Documents/Ms. Study/4. Natural Language Processing/Code/Assignment 1/Project1NLP_Model/'
#print ('Experiment Dir is:' + ExpDir)
modelDir = '/Users/ranjitsmac/Documents/Ms. Study/4. Natural Language Processing/Code/Assignment 1/Project1NLP_Model/model/'
#print ('Model is in:' + modelDir)
#You need to change this path with your local path to the saved model I provides with the code

class FilePaths:
    "filenames and paths to data"
    fnCharList = modelDir+'charList.txt'
    fnAccuracy = modelDir+'accuracy.txt'
    fnInfer = ExpDir+'unitek12.png'



class Batch:
    "batch containing images and ground truth texts"
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2

def preprocess(img, imgSize, dataAugmentation=False):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    # increase dataset size by applying random stretches to the images
    if dataAugmentation:
        stretch = (random.random() - 0.5) # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1) # random width, but at least 1
        img = cv2.resize(img, (wStretched, img.shape[0])) # stretch horizontally by factor 0.5 .. 1.5

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # transpose for TF
    img = cv2.transpose(target)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img

def train(model, loader):
    "train NN"
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best valdiation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = 5 # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        recognized = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    return charErrorRate

modelDir = '/Users/ranjitsmac/Documents/Ms. Study/4. Natural Language Processing/Code/Assignment 1/Project1NLP_Model/model/'

class Model:
    "minimalistic TF model for HTR"

    # model constants
    batchSize = 50
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
        "init model: add CNN, RNN and CTC and initialize TF"
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        # CNN
        self.inputImgs = tf.placeholder(tf.float32, shape=(Model.batchSize, Model.imgSize[0], Model.imgSize[1]))
        cnnOut4d = self.setupCNN(self.inputImgs)

        # RNN
        rnnOut3d = self.setupRNN(cnnOut4d)

        # CTC
        (self.loss, self.decoder) = self.setupCTC(rnnOut3d)

        # optimizer for NN parameters
        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        # initialize TF
        (self.sess, self.saver) = self.setupTF()


    def setupCNN(self, cnnIn3d):
        "create CNN layers and return output of these layers"
        cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)

        # list of parameters for the layers
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
        numLayers = len(strideVals)

        # create layers
        pool = cnnIn4d # input to first CNN layer
        for i in range(numLayers):
            kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

        return pool


    def setupRNN(self, rnnIn4d):
        "create RNN layers and return output of these layers"
        rnnIn3d = tf.squeeze(rnnIn4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = 256
        cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        return tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])


    def setupCTC(self, ctcIn3d):
        "create CTC loss and decoder and return them"
        # BxTxC -> TxBxC
        ctcIn3dTBC = tf.transpose(ctcIn3d, [1, 0, 2])
        # ground truth text as sparse tensor
        self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
        # calc loss for batch
        self.seqLen = tf.placeholder(tf.int32, [None])
        loss = tf.nn.ctc_loss(labels=self.gtTexts, inputs=ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True)
        # decoder: either best path decoding or beam search decoding
        if self.decoderType == DecoderType.BestPath:
            decoder = tf.nn.ctc_greedy_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.BeamSearch:
            decoder = tf.nn.ctc_beam_search_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
        elif self.decoderType == DecoderType.WordBeamSearch:
            # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
            word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

            # prepare information about language (dictionary, characters in dataset, characters forming words)
            chars = str().join(self.charList)
            wordChars = open(modelDir+'wordCharList.txt').read().splitlines()[0]
            corpus = open(modelDir+'corpus.txt').read()

            # decode using the "Words" mode of word beam search
            decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

        # return a CTC operation to compute the loss and a CTC operation to decode the RNN output
        return (tf.reduce_mean(loss), decoder)


    def setupTF(self):
        "initialize TF"
        print('Python: '+sys.version)
        print('Tensorflow: '+tf.__version__)

        sess=tf.Session() # TF session

        saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
        modelDir = '/Users/ranjitsmac/Documents/Ms. Study/4. Natural Language Processing/Code/Assignment 1/Project1NLP_Model/model/'
        latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)

        # load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())

        return (sess,saver)


    def toSparse(self, texts):
        "put ground truth texts into sparse tensor for ctc_loss"
        indices = []
        values = []
        shape = [len(texts), 0] # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)


    def decoderOutputToText(self, ctcOutput):
        "extract texts from output of CTC decoder"

        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(Model.batchSize)]

        # word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank=len(self.charList)
            for b in range(Model.batchSize):
                for label in ctcOutput[b]:
                    if label==blank:
                        break
                    encodedLabelStrs[b].append(label)

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded=ctcOutput[0][0]

            # go over all indices and save mapping: batch -> values
            idxDict = { b : [] for b in range(Model.batchSize) }
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0] # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


    def trainBatch(self, batch):
        "feed a batch into the NN to train it"
        sparse = self.toSparse(batch.gtTexts)
        rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
        (_, lossVal) = self.sess.run([self.optimizer, self.loss], { self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * Model.batchSize, self.learningRate : rate} )
        self.batchesTrained += 1
        return lossVal


    def inferBatch(self, batch):
        "feed a batch into the NN to recngnize the texts"
        decoded = self.sess.run(self.decoder, { self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * Model.batchSize } )
        return self.decoderOutputToText(decoded)


    def save(self):
        "save model to file"
        self.snapID += 1
        self.saver.save(self.sess, 'model/snapshot', global_step=self.snapID)


# this function sepreates sentences with .,!,?,\n regular expression.
def listSplit(name):
    file = open(name, "r")
    doclist = [ line.lower() for line in file ]
    docstr = ''. join(doclist)
    sentences = re.split(r'[.!?\n]', docstr)
    sentences = [x for x in sentences if x != '']
    return sentences

# Add contractions in sentances if necessary . Eg: I'd -> I would
def expand_contractions(sens):
    deContraction = []
    for i,j in enumerate(sens):
        deContraction.append(contractions.fix(j))
    return deContraction

# Seperate tokens from each sentences.
def token_sept(sens):
    tokens= []
    for i,j in enumerate(sens):
        tokens.append(j.split(' '))
    return tokens
# pick all words from corpus whose size is (len(recognised_word) -1) or len(recognised_word) +1.
# For example my rec_word is : hello len is 5. Then all letters from corpus of size 4 to 6 will take for dictonary error correction.
def picked_words(expand_contractions,lenWord):
    counts_word = []
    for d in expand_contractions:
        for j in d:
            if len(j) >= lenWord-1 and len(j) <= lenWord+1:
                counts_word.append(j)
    return counts_word

# Typographic errors detection refered "https://pdfs.semanticscholar.org/c64f/1bd3a1bd7f7fe4cadc469b4b94c45ad12b5d.pdf" Research paper
def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    split = []
    deletes = []
    transposes = []
    replaces = []
    inserts = []

    for i in range(len(word)+1):
        split.append([word[:i], word[i:]])

    for i,j in split:
        if j:
            deletes.append(i + j[1:])

    for i,j in split:
        if len(j)>1:
            transposes.append(i + j[1] + j[0] + j[2:])

    for i,j in split:
        if j:
            for c in letters:
                replaces.append(i + c + j[1:])


    for i,j in split:
        for c in letters:
            inserts.append(i + c + j)


    return list(deletes+inserts+transposes+replaces)

# check all possiblities of Typographic errors in to the word picked.
def result(final_select,picked_words):
    data = []
    for i in final_select:
        if i in picked_words:
            data.append(i)
    return data

# Answer with weighted levenshtein with characted differences.
def data_print(answer,recWord):
    substitute_costs = np.ones((128, 128), dtype=np.float64)
    for i in answer:
         print("The word may be : {} the change in char is of  : {} digits".format(i,int(dam_lev(recWord, i, substitute_costs=substitute_costs))))

# biagram function. Eg. 'Hello' -> ['he','el','ll','lo']
def ngram(s1):
    return list(ngrams(s1, 2))

# remove doubles into the biagram for calcuation purpus. refered in the same research paper : A. N-gram Analysis [3] Section
def remove_double(s1):
    count = 0
    for (filename,filepath) in s1:
        count = count + 1
        for (filename1,filepath1) in s1[count:]:
            if filename == filename1:
                s1.remove((filename1,filepath1))
    return s1

# Accuracy count for 2 biagram.
def accuracy_count(s1,s2):
    data = []
    for i,j in s1:
        for k,l in s2:
            if i==k and j==l:
                data.append((i,j))
    return (2*len(data)) / (len(s1)+len(s2))


def infer(model, fnImg):
    "recognize text in image provided by file path"
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img] * Model.batchSize) # fill all batch elements with same input image
    recognized = model.inferBatch(batch) # recognize text
    print('Recognized:', '"' + recognized[0] + '"') # all batch elements hold same result
    return recognized[0]

def main():
    "main function"


    decoderType = DecoderType.BestPath

    print(open(FilePaths.fnAccuracy).read())
    model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
    recWord = infer(model, FilePaths.fnInfer)

    # data fetch
    sentences = listSplit('europarl-v7.de-en.en')

    # sentences creations.
    sentences = sentences[0:len(sentences)-1]

    # identifie contractions.
    expand_contrac = expand_contractions(sentences)

    # convert sentennces to tokens.
    tokens = token_sept(expand_contrac)


    # pick all words from corpus according sto size.
    picked_word = picked_words(tokens,len(recWord))

    # checking all possiblities of Typographic errors
    final_selection=edits1(recWord)

    # results with uniquness between recognised word and corpus word
    answer = result(final_selection,picked_word)

    # print the result with Weighted Levenshtein
    data_print(answer,recWord)
    # if the multiple answer is found then gives the output of all multiple word occurance with accuracy of biagram.
    for i in answer:
        s1_N = ngram(i)
        s2_N = ngram(recWord)

        s1_rm = remove_double(s1_N)
        s2_rm = remove_double(s2_N)

        print('Biagram --> word from corpus = {} and recognises = {} and accuracy is = {}'.format(recWord,i,accuracy_count(s1_rm,s2_rm)))



if __name__ == '__main__':
    main()

# It will take minimum 10 min to run. Typographic errors + N-gram analysis = More accurate results.



## Test case 1:
    #  Biagram --> word from corpus = unetek and recognises = unitek and accuracy is = 0.4444444444444444

# Test Case 2 :
    # Biagram --> word from corpus = little and recognises = little and accuracy is = 1.0
    # Biagram --> word from corpus = little and recognises = tittle and accuracy is = 0.3333333333333333
