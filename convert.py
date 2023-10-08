import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Mute TF messages

import pyautogui
import numpy as np
import PIL
import cv2

import argparse

from detect import locate

def load_weights(weightsPath):
    #Load Tensorflow weights
    print("Loading weights...")
    with tf.io.gfile.GFile(weightsPath, "rb") as f:
        weight_def = tf.compat.v1.GraphDef()
        weight_def.ParseFromString(f.read())
    with tf.Graph().as_default() as weights:
        tf.import_graph_def(weight_def, name="tcb")
    return weights

def get_prediction(tiles, sess, probabilities, prediction, x, keep_prob):

    #Reshape tileset into format used by NN
    validation_set = np.swapaxes(np.reshape(tiles, [32*32, 64]),0,1)

    #Run NN
    guess_prob, guessed = sess.run(
        [probabilities, prediction], 
        feed_dict={x: validation_set, keep_prob: 1.0})
    a = np.array(list(map(lambda x: x[0][x[1]], zip(guess_prob, guessed))))
    # tile_probablities = a.reshape([8,8])[::-1,:]

    #Convert Guesses into FEN
    labelIndex2Name = lambda label_index: ' KQRBNPkqrbnp'[label_index]
    pieceNames = list(map(lambda k: '1' if k == 0 else labelIndex2Name(k), guessed)) #Convert " " to "1" (format used in FEN)
    fen = '/'.join([''.join(pieceNames[i*8:(i+1)*8]) for i in reversed(range(8))])

    return fen

def get_fen(args):
    #Load Model
    model_path = "./weights/model.pb"
    if args.weights:
        model_path = args.weights
    weights = load_weights(model_path)
    sess, x_weights, keep_prob, prediction, probabilities = tf.compat.v1.Session(graph=weights), weights.get_tensor_by_name('tcb/Input:0'), \
        weights.get_tensor_by_name('tcb/KeepProb:0'), weights.get_tensor_by_name('tcb/prediction:0'), weights.get_tensor_by_name('tcb/probabilities:0')

    #Args
    if args.filename:
        img = cv2.imread(args.filename)
    else:
        img = np.asarray(pyautogui.screenshot())
        args.locate = True
    if args.locate:
        x, x2, y, y2 = locate(img)
        img = img[y:y2, x:x2]

    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_gray_img = np.asarray(PIL.Image.fromarray(grayscale_img).resize([256,256], PIL.Image.BILINEAR), dtype=np.uint8) / 255.0

    #By resizing to 257 instead of 256 - it gives for a little more room for the edge pieces
    #Generate tiles for NN
    tiles = np.zeros([32,32,64], dtype=np.float32)
    for rank in range(8): # rows (numbers)
        for file in range(8): # columns (letters)
            tiles[:,:,(rank*8+file)] = processed_gray_img[(7-rank)*32:((7-rank)+1)*32,file*32:(file+1)*32]
    fen = get_prediction(tiles, sess, probabilities, prediction, x_weights, keep_prob)

    #make compressed fen (/11111111/ --> /8/)
    if args.compress:
        for length in reversed(range(2, 9)):
            fen = fen.replace(length * "1", str(length))

    #Reverese FEN if black
    if args.reverse:
        return "/".join(fen.split("/")[::-1])
    return fen

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Get FEN from image. Use -f to input filename of image (default will take a screenshot). Use -w to input filepath of model. Use -l if the board needs to be located in image. Use -r,-c as FEN modifiers.")
    parser.add_argument("-f", "--filename", help="Input image filename, otherwise screenshot will be taken")
    parser.add_argument("-w", "--weights", help="Input the filename for the weights. (default is './weights/model.pb')")
    parser.add_argument("-l", "--locate", help="Locate chessboard in image", action="store_true")
    parser.add_argument("-c", "--compress", help="Compress FEN", action="store_true")
    parser.add_argument("-r", "--reverse", help="Reverse FEN", action="store_true")

    args = parser.parse_args()

    FEN = get_fen(args)
    print(f"FEN: {FEN}")