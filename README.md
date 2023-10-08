# TF Chess Piece Recognition
Extract FEN from image of chessboard using Tensorflow AI Recognition model, implemented with ```tf.Graph()```.

## Usage

Install requirements
```
pip install -r requirements.txt
```
Pass the following arguments to convert.py:
```
usage: convert.py [-h] [-f FILENAME] [-w WEIGHTS] [-l] [-c] [-r]

Get FEN from image. Use -f to input filename of image (default will take a screenshot). Use -w to input filepath of model. Use -l if the board needs to be located in image. Use -r,-c as FEN modifiers.

options:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Input image filename, otherwise screenshot will be taken
  -w WEIGHTS, --weights WEIGHTS
                        Input the filename for the weights. (default is './weights/model.pb')
  -l, --locate          Locate chessboard in image
  -c, --compress        Compress FEN
  -r, --reverse         Reverse FEN
```
Input filename or a screenshot will be taken. Use "-l" or "--locate" if the chess board is not cropped within the image. Output should be something like this:
```
python convert.py -f ./README_Imgs/chessboard1.jpg -l

Loading weights...
FEN: rnbqkbnr/pppppppp/11111111/11111111/11111111/11111111/PPPPPPPP/RNBQKBNR
```
### FEN Commands:
"-r" or "--reverse" reverses the FEN
```
FEN: rnbqkbnr/pppppppp/11111111/11111111/11111111/11111111/PPPPPPPP/RNBQKBNR
Reversed FEN: RNBQKBNR/PPPPPPPP/11111111/11111111/11111111/11111111/pppppppp/rnbqkbnr
```
"-c" or "--compress" compresses the FEN (takes a string of ones and replaces them with its length)
```
FEN: rnbqkbnr/pppppppp/11111111/11111111/11111111/11111111/PPPPPPPP/RNBQKBNR
Compressed FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
```

## Breakdown
