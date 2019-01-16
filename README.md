# Spec_ResNet
    Spectrogram is selected as preprocessing feature of audio samples and a feature representation 
    method based on deep residual network (Spec-ResNet) is implemented here to detect audio steganography
    based on AAC(Advanced Audio Coding).
    
    First of all, we analysed existing audio steganography based on AAC. Although the embedding domain
    is different, the final impact of existing schemes are the change of the MDCT coefficients of the encoded
    audio signals in different frequency range. Spectrogram, a basic visual representation of the spectrum of
    frequencies of audio signals which can show the energy amplitude information of different frequency bands
    changing over time is chosen as analysis object of ResNet to get 40-dimension classification featureto 
    decide whether or not a piece of audio contains secret message.
    
    Still, this is a work on progress, we will update it lately.

# Requirements
    Specifically, this project is based on Tensorflow 1.0.0 + CUDA 8.0.61 + CuDNN 5.1.10 + Python 2.7.15.
    The code has been tested on Ubuntu 16.04.
     
# Data set
    The data set is too large to be uploaded, but the structure of the data directory is as follows: 
        cover/
            *.txt
        lsbee_0.1/
            *.txt
        lsbee_0.3/
            *.txt
        lsbee_0.5/
            *.txt
        lsbee_0.8/
            *.txt
        lsbee_1.0/
            
