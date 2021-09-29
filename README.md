# DTW-distance-in-kNN-classifier
**Consonant classification with Dynamic Time Warped (DTW) distance metric in kNN classifier**


**Problem Statement:**

Carry out consonant classification and find out the role of different features and distance measures for the classification task. Six different consonants will be used for this purpose, namely, /p/, /t/, /k/, /b/, /d/, and /g/. There are four different classification tasks (CTs) as follows:
● Classification Task 1 (CT1): Two-class classification with class 1 (C1) comprising /p/, /t/, /k/ and class 2 (C2) comprising /b/, /d/, /g/.
● Classification Task 2 (CT2): Three-class classification where C1: /p/, C2: /t/, and C3: /k/
● Classification Task 3 (CT3): Three-class classification where C1: /b/, C2: /d/, and C3: /g/
● Classification Task 4a (CT4a): Direct six-class classification where C1: /p/, C2: /t/, C3: /k/, C4: /b/, C5: /d/, and C6: /g/
● Classification Task 4b (CT4b): Hierarchical six-class classification where C1: /p/, C2: /t/, C3: /k/, C4: /b/, C5: /d/, and C6: /g/. 

Here, first run CT1. Depending on the predicted class from CT1, run either CT2 or CT3.
For this purpose, record (at 16kHz sampling rate) vowel-consonant-vowel (VCV) where C is one of the six consonants (/p/, /t/, /k/, /b/, /d/, and /g/) and V is one of the following six vowels: V1 (as in Hid), V2 (as in Head), V3 (as in Had), V4 (as in Hudd), V5 (as in Hod), and V6 (as in Hood).
For a chosen C, record the following six VCVs, each repeated FIVE times: V1CV1, V2CV2, V3CV3, V4CV4, V5CV5, and V6CV6
Hence, for a chosen C record and prepare the following 30 wav files:
V1CV1_1.wav, V1CV1_2.wav, V1CV1_3.wav, V1CV1_4.wav, V1CV1_5.wav, V2CV2_1.wav, V2CV2_2.wav, ……., V6CV6_5.wav
Repeat this for all six choices of C, i.e., /p/, /t/, /k/, /b/, /d/, and /g/. Thus, a folder containing 30x6=180 recorded wavfiles is needed for the experiments. The naming convention should be following the list given above, i.e., V1pV1_1.wav, V1pV1_2.wav etc.
For each of the 180 wavfiles, mark the following three timestamps:
1. Begin (B) of pre-consonant vowel
2. Middle (M) of the consonant
3. End (E) of post-consonant vowel
Mark these timestamps using Audacity software by marking a segment from B to M (VC) and marking another segment from M to E (CV). For every wavfile, prepare a text file (e.g., V1pV1_1.txt, V1pV1_2.txt etc.) containing the timestamps. 
For all the classification tasks, first THREE of the five repetitions should be used in training and remaining TWO for every vowel and consonant combination should be used for testing.

Experiment with various segment lengths from each VCV recording centered around the middle of the consonant. 

A segment (as shown by the red colored part in the figure below) starts from M-p(M-B) and ends at M+p(E-M). Thus, the segment length is parameterized by p (0<p<1). When p=1, the entire VCV from B to E is used.
![image](https://user-images.githubusercontent.com/79351706/135352729-f158856a-e564-4d5d-ab7f-6c65818fb1e8.png)


Experiment with four values of p, namely, 0.25, 0.50, 0.75 and 1.0.

For the classification task, consider a short-time feature (D-dimensional) every 20msec with a 10msec shift. Thus, each VCV recording will be represented as a sequence of D-dim features, i.e., a feature matrix, the length of which varies from one recording to another and also depending on the choice of p. Use the K Nearest Neighborhood (KNN) classifier for all the classification tasks. The distance metric to be used in the KNN classifier is dynamic time warped (DTW) distance between the test feature matrix and a training feature matrix. Choose K=10 in the KNN classifier.

Use the short-time spectrum computed using FFT order N as the feature vector. Vary N=128, 256, 1024. Experiment separately with the Euclidean and the Itakura Saito distance measure in computing DTW distance. For each of these two distance measures, report the classification accuracies for all combinations of p and N. Do this separately for CT1, CT2, CT3, CT4a, CT4b.

**Results:**

![image](https://user-images.githubusercontent.com/79351706/135353333-3a8d7a4c-66a2-4726-9dd7-2488cb6b7d05.png)
![image](https://user-images.githubusercontent.com/79351706/135353373-6f8021d8-0a17-4583-8df9-3459ad588e7a.png)
![image](https://user-images.githubusercontent.com/79351706/135353410-4f2e4e2c-24fa-4185-9a16-a3a9070d49e3.png)
