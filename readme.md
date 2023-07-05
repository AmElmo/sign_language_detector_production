This repo contains the production setup for this project: https://github.com/AmElmo/sign_language_detector

It is a simple app that:
- displays a live video stream with holistic keypoints detected in real-time (using the Mediapipe library in JavaScript)
- load an LSTM model previously trained using Tensorflow JS
- automatically detects sequences of sign languages
- translates the sequences with the relevant symptom


For more details, see [project repository](https://github.com/AmElmo/sign_language_detector).


TODOs:
- ~~fix the innerHTML so it displays a different message when there is no prediction~~
- ~~retrain model to have better predictions (currently completely off)~~
- Change the model in production
