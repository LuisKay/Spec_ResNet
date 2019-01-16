


># Spec_ResNet  
>**Spectrogram** is selected as preprocessing feature of audio samples and a feature representation method based on deep residual network (Spec-ResNet) is implemented here to detect audio steganography based on AAC(Advanced Audio Coding).
    
    First of all, we analysed existing audio steganography based on AAC. Although the embedding domain is different, the final impact of existing schemes are the change of the MDCT coefficients of the encoded audio signals in different frequency range. Spectrogram, a basic visual representation of the spectrum of frequencies of audio signals which can show the energy amplitude information of different frequency bands changing over time is chosen as analysis object of ResNet to get 40-dimension classification feature to decide whether or not a piece of audio contains secret message.
    
    Still, this is a work on progress, we will update it lately.

># Requirements
    Specifically, this project is based on Tensorflow 1.0.0 + CUDA 8.0.61 + CuDNN 5.1.10 + Python 2.7.15. The code has been tested on Ubuntu 16.04.
     
># Data set
>The data set is too large to be uploaded, there is 15 data directory `algo_ebr` besides `cover`: `lsbee_0.1`, `lsbee_0.3`, `lsbee_0.5`，`lsbee_0.8`，`lsbee_1.0`，`min_0.1`, `min_o.3`, `min_0.5`, `min_0.8`, `min_1.0`,`sign_0.1`, `sign_0.3`, `sign_0.5`, `sign_0.8`, `sign_1.0`. In `algo_ebr`, `algo`: [LSB_EE](https://www.computer.org/csdl/proceedings/mines/2010/4258/00/4258a841-abs.html), [MIN](http://en.cnki.com.cn/Article_en/CJFDTOTAL-XXWX201107046.htm), [SIGN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5629745) represents the steganographic algorithm from which this sample is generated and `ebr` is the relative embedding rate: 0.1, 0.3, 0.5, 0.8, 1.0.

