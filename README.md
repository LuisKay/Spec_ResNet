


# Spec_ResNet  
**Spectrogram** is selected as preprocessing feature of audio samples and a feature representation method based on deep residual network (**Spec-ResNet**) is implemented here to detect audio steganography based on AAC(Advanced Audio Coding) and MP3(MPEG-1 Audio Layer III).

First of all, we analysed existing audio steganography based on AAC and MP3. Although the embedding domain is different, the final impact of existing schemes are the subtle modifications of MDCT coefficients of the encoded audio signals in different frequency range. Spectrogram, a basic visual representation of the spectrum of frequencies of audio signals which can show energy amplitude information of different frequency bands over time and contains abundant time-frequency information about the audio signal.

Still, this is a work on progress, we will update it lately. You will find responding paper "Spec-ResNet: A General Audio Steganalysis scheme based on Deep Residual Network of Spectrogram" by Yanzhen Ren, Dengkai Liu, Qiaochu Xiong, Jianming Fu, Lina Wang in [https://arxiv.org/abs/1901.06838](https://arxiv.org/abs/1901.06838).

# Requirements
* Tensorflow 1.0.0
* CUDA 8.0.61
* CuDNN 5.1.10
* Python 2.7.15. 
     
# Data set
* [In-house build dataset](https://pan.baidu.com/s/1a6oT_iQcZLXB9CXs6Fg_rA#list/path=%2F)
  * stego with 5 relative embedding rate for AAC is provided. (2s, stereo, 1024Kbps, decoded WAV)
    * [lsb_ee](https://www.computer.org/csdl/proceedings/mines/2010/4258/00/4258a841-abs.html) (0.1, 0.2, 0.3, 0.5, 1.0) 
    * [min](http://en.cnki.com.cn/Article_en/CJFDTOTAL-XXWX201107046.htm) (0.1, 0.2, 0.3, 0.5, 1.0)
    * [sign](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5629745) (0.1, 0.2, 0.3, 0.5, 1.0)
  * stego with approximately maxium embedding capcity for Mp3Stego is provided.(5s, mono, 705Kbps, decoded wav)
    * [Mp3Stego](https://www.petitcolas.net/steganography/mp3stego/)
  
    spectrogram feature extraction script is in **AAC (2s, stereo, 1024Kbps, decoded WAV).rar** together with a small sample subset, whose spectrogram matrix with window size `N=512` in **Spectrogram.rar**.

* [ASDIIE](https://pan.baidu.com/s/1rYCzJRksHkgbOOYI9MqQjA#list/path=%2F)(***extract code: z28d***)
  * [EECS](https://link.springer.com/chapter/10.1007/978-3-319-64185-0_16)
  * [AHCM](https://ieeexplore.ieee.org/abstract/document/8626153/)
  * [Mp3UnderCover](https://sourceforge.net/projects/ump3c/)

# Experiments
For AAC based steganographic algorithm, we choosed above three as objects. Considering the capicity of MP3stego, we choosed several emerging MP3 adaptive steganography([AHCM](https://ieeexplore.ieee.org/abstract/document/8626153), [EECS](https://link.springer.com/chapter/10.1007/978-3-319-64185-0_16)) with distortion minimization and UnderMp3Cover to test the detection ability. The [dataset](https://github.com/Charleswyt/tf_audio_steganalysis) we choose was also changed. It wil be coming soon...
