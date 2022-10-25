# ASL-to-Text-Translation-as-an-Assistive-Technology

## Motivation
There is always a need for improved assistive technology to help better the lives of the speech impaired in society. There is certainly a lot of prior work which has been done in the field but still several limitations which prevent the widespread adoption and use of such technology. Current day research aims to build more energy efficient systems to achieve these objectives. For this project we aim to build an on-device ASL recognition system which can translate ASL gestures into english text. 

## Task Setup
The primary task is to recognize ASL gestures from live video footage captured using a Picam  and convert it into text which is to be displayed to a user.

The secondary stage of the project will involve converting the translated text to speech which can then also be heard by a user thus  further helping to bridge the gap between both the aurally and visually impaired. 

The sign language recognition model will be trained and developed using the multi-modal ASL dataset ‘How2Sign’. It will be trained, evaluated and tested on subsets of data from the same distribution i.e. the train, validation and test sets. The model will be further evaluated on sample test footage captured using the Picam which will be manually annotated and labeled. Evaluation will be performed on the basis of both performance and efficiency. Performance metrics such as accuracy, recall, precision and f1-score shall be used while for efficiency we shall use metrics such as the number of parameters, FLOPS, inference time, etc. in order to perform benchmarking and a comparative study between our models and prior work. 
