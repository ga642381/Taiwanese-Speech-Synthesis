# taiwanese-speech-synthesis

* This is a Taiwanese Speech Synthesis System based on Tacotron2 and WaveRNN
* The input is in [臺羅拼音](https://zh.wikipedia.org/wiki/%E8%87%BA%E7%81%A3%E9%96%A9%E5%8D%97%E8%AA%9E%E7%BE%85%E9%A6%AC%E5%AD%97%E6%8B%BC%E9%9F%B3%E6%96%B9%E6%A1%88)，and the output is Taiwanese Speech
* The model that translate from Chinese(中文，華語) to 臺羅拼音 will be released soon
* The trained TTS model can be combined with [TaiwaneseTTS](https://github.com/ga642381/TaiwaneseTTS), so that we will have a awesome GUI interface. I intend to merge these two repositories into one in the future

## Qucik Start

1. Download pretrained model
2. Specify the inputs in "sentences.txt"
3. generate with the command
```
python gen_tts.py --tts_weights ./pretrained_models/tacotron2.pyt --voc_weights ./pretrained_models/wavernn.pyt --save_dir ./audio_samples
```

