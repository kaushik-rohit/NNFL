1) The model is saved under ./save directory.

2) CNN expects a image with dimesions 64*64.

3) Reshaping, preprocessing and Resizing has been taken care of
   in the predict function.
   
4) For evaluation, the code needs to modified under if __name__.py
   section and load images and call iclass.predict on images and check
   against output.
   
5) The output is one-hot encoded (8 bits), and a expected output format is
   np array [0, 1, 0, 0, 0, 0, 0, 0] for "chair" prediction.
