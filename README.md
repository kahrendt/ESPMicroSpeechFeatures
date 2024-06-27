# ESPMicroSpeechFeatures

A fork of the [TensorFlow Lite Micro's Microfrontend](https://github.com/tensorflow/tflite-micro) for [microWakeWord](https://github.com/kahrendt/microWakeWord/). It is used to generate spectrogram features from audio samples. Using this library avoids having to use a TFLM interpeter for generating features which increases performance on Espressif chips.