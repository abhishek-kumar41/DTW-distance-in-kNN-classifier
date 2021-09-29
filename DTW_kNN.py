import numpy as np
import librosa
from scipy.fft import fft


def classification_task_1():

    N_val = [128, 256, 1024]
    for i in range(1, 5):
        p = i*0.25
        for j in range(len(N_val)):
            N = N_val[j]
            print(f"CT1: p={p}, N= {N}")
            CT1(p, N)
            print(f"CT2: p={p}, N= {N}")
            CT2(p, N)
            print(f"CT3: p={p}, N= {N}")
            CT3(p, N)
            print(f"CT4a: p={p}, N= {N}")
            CT4a(p, N)
            print(f"CT4b: p={p}, N= {N}")
            CT4b(p, N)

    # CT1(p, N)
    # CT2(p, N)
    # CT3(p, N)
    # CT4a(p, N)
    #CT4b(p, N)
    return


def CT1(p, N):

    consonants1 = ["p", "t", "k"]
    consonants2 = ["b", "d", "g"]
    # count = 0
    # p = 1
    # N = 128
    train_feature = []
    train_class = []
    test_feature = []
    test_class = []

    for i in range(6):
        for k in range(3):
            for j in range(len(consonants1)):
                y, fs = librosa.load("V" + str(i+1) + consonants1[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)
                # print("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav")
                time_stamp_list = open("V" + str(i+1) + consonants1[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])
                # print(time_stamp)
                # print((float(time_stamp_list[0][1])))
                # count = count + 1
                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)
                # print(start_index)
                # print(end_index)
                # print(len(y))
                y = y[start_index:end_index]
                # print(len(y))
                hop_len = int(0.01*fs)
                win_len = int(0.02*fs)
                #print(win_len)
                #y_stft1 = librosa.stft(y=y, n_fft=N, hop_length=hop_len, win_length=win_len)
                #f, t, y_stft = scipy.signal.stft(x=y, fs=fs, nperseg=win_len, noverlap=hop_len, nfft=N)
                #print(y_stft)
                y_stft = n_fft(y, N)
                train_feature.append(y_stft)
                train_class.append(1)

            for j in range(len(consonants2)):
                y, fs = librosa.load("V" + str(i+1) + consonants2[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)
                # print("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav")
                time_stamp_list = open("V" + str(i+1) + consonants2[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])

                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)

                y = y[start_index:end_index]

                y_stft = n_fft(y, N)
                train_feature.append(y_stft)
                train_class.append(2)


    # print(len(train_class))
    # print(len(train_feature))

    # train_feature = np.array(train_feature)


    # knn = KNeighborsClassifier(n_neighbors=10, metric=DTW)
    # knn.fit(train_feature, train_class)

        for k in range(3, 5):
            for j in range(len(consonants1)):
                y, fs = librosa.load("V" + str(i+1) + consonants1[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)
                # print("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav")
                time_stamp_list = open("V" + str(i+1) + consonants1[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])
                # print(time_stamp)
                # print((float(time_stamp_list[0][1])))
                # count = count + 1
                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)
                # print(start_index)
                # print(end_index)
                # print(len(y))
                y = y[start_index:end_index]
                # print(len(y))
                hop_len = int(0.01*fs)
                win_len = int(0.02*fs)
                #print(win_len)
                #y_stft1 = librosa.stft(y=y, n_fft=N, hop_length=hop_len, win_length=win_len)
                #f, t, y_stft = scipy.signal.stft(x=y, fs=fs, nperseg=win_len, noverlap=hop_len, nfft=N)
                #print(y_stft)
                y_stft = n_fft(y, N)
                test_feature.append(y_stft)
                test_class.append(1)

            for j in range(len(consonants2)):
                y, fs = librosa.load("V" + str(i+1) + consonants2[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)
                # print("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav")
                time_stamp_list = open("V" + str(i+1) + consonants2[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])

                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)

                y = y[start_index:end_index]

                y_stft = n_fft(y, N)
                test_feature.append(y_stft)
                test_class.append(2)

    knn_function_ct1(train_feature, test_feature, train_class, test_class)

    return


def CT2(p, N):

    consonants = ["p", "t", "k"]

    # count = 0
    # p = 1
    # N = 128
    train_feature = []
    train_class = []
    test_feature = []
    test_class = []

    for i in range(6):
        for k in range(3):
            for j in range(len(consonants)):
                y, fs = librosa.load("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)

                time_stamp_list = open("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])

                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)

                y = y[start_index:end_index]

                y_stft = n_fft(y, N)
                train_feature.append(y_stft)
                if consonants[j] == "p":
                    train_class.append(1)
                if consonants[j] == "t":
                    train_class.append(2)
                if consonants[j] == "k":
                    train_class.append(3)

        for k in range(3, 5):
            for j in range(len(consonants)):
                y, fs = librosa.load("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)
                # print("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav")
                time_stamp_list = open("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])

                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)

                y = y[start_index:end_index]

                y_stft = n_fft(y, N)
                test_feature.append(y_stft)
                if consonants[j] == "p":
                    test_class.append(1)
                if consonants[j] == "t":
                    test_class.append(2)
                if consonants[j] == "k":
                    test_class.append(3)



    knn_function_ct2(train_feature, test_feature, train_class, test_class)

    return


def CT3(p, N):

    consonants = ["b", "d", "g"]

    # count = 0
    # p = 1
    # N = 128
    train_feature = []
    train_class = []
    test_feature = []
    test_class = []

    for i in range(6):
        for k in range(3):
            for j in range(len(consonants)):
                y, fs = librosa.load("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)

                time_stamp_list = open("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])

                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)

                y = y[start_index:end_index]

                y_stft = n_fft(y, N)
                train_feature.append(y_stft)
                if consonants[j] == "b":
                    train_class.append(1)
                if consonants[j] == "d":
                    train_class.append(2)
                if consonants[j] == "g":
                    train_class.append(3)

        for k in range(3, 5):
            for j in range(len(consonants)):
                y, fs = librosa.load("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)
                # print("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav")
                time_stamp_list = open("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])

                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)

                y = y[start_index:end_index]

                y_stft = n_fft(y, N)
                test_feature.append(y_stft)
                if consonants[j] == "b":
                    test_class.append(1)
                if consonants[j] == "d":
                    test_class.append(2)
                if consonants[j] == "g":
                    test_class.append(3)

    # print(len(test_feature))
    # print(len(test_class))
    # print(test_class)

    knn_function_ct3(train_feature, test_feature, train_class, test_class)

    return


def CT4a(p, N):

    consonants = ["p", "t", "k", "b", "d", "g"]

    # count = 0
    # p = 1
    # N = 128
    train_feature = []
    train_class = []
    test_feature = []
    test_class = []

    for i in range(6):
        for k in range(3):
            for j in range(len(consonants)):
                y, fs = librosa.load("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)

                time_stamp_list = open("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])

                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)

                y = y[start_index:end_index]

                y_stft = n_fft(y, N)
                train_feature.append(y_stft)
                if consonants[j] == "p":
                    train_class.append(1)
                if consonants[j] == "t":
                    train_class.append(2)
                if consonants[j] == "k":
                    train_class.append(3)
                if consonants[j] == "b":
                    train_class.append(4)
                if consonants[j] == "d":
                    train_class.append(5)
                if consonants[j] == "g":
                    train_class.append(6)

        for k in range(3, 5):
            for j in range(len(consonants)):
                y, fs = librosa.load("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)
                # print("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav")
                time_stamp_list = open("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])

                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)

                y = y[start_index:end_index]

                y_stft = n_fft(y, N)
                test_feature.append(y_stft)
                if consonants[j] == "p":
                    test_class.append(1)
                if consonants[j] == "t":
                    test_class.append(2)
                if consonants[j] == "k":
                    test_class.append(3)
                if consonants[j] == "b":
                    test_class.append(4)
                if consonants[j] == "d":
                    test_class.append(5)
                if consonants[j] == "g":
                    test_class.append(6)

    # print(len(test_feature))
    # print(len(test_class))
    # print(test_class)

    knn_function_ct4a(train_feature, test_feature, train_class, test_class)

    return


def CT4b(p, N):

    consonants = ["p", "t", "k", "b", "d", "g"]

    # count = 0
    # p = 1
    # N = 128
    train_feature = []
    train_feature_ct2 = []
    train_feature_ct3 = []
    train_class = []
    train_class_ct2 = []
    train_class_ct3 = []
    test_feature = []
    test_class = []

    for i in range(6):
        for k in range(3):
            for j in range(len(consonants)):
                y, fs = librosa.load("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)

                time_stamp_list = open("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])

                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)

                y = y[start_index:end_index]

                y_stft = n_fft(y, N)
                train_feature.append(y_stft)
                if consonants[j] == "p":
                    train_class.append(1)
                    train_feature_ct2.append(y_stft)
                    train_class_ct2.append(1)
                if consonants[j] == "t":
                    train_class.append(1)
                    train_feature_ct2.append(y_stft)
                    train_class_ct2.append(2)
                if consonants[j] == "k":
                    train_class.append(1)
                    train_feature_ct2.append(y_stft)
                    train_class_ct2.append(3)
                if consonants[j] == "b":
                    train_class.append(2)
                    train_feature_ct3.append(y_stft)
                    train_class_ct3.append(4)
                if consonants[j] == "d":
                    train_class.append(2)
                    train_feature_ct3.append(y_stft)
                    train_class_ct3.append(5)
                if consonants[j] == "g":
                    train_class.append(2)
                    train_feature_ct3.append(y_stft)
                    train_class_ct3.append(6)

        for k in range(3, 5):
            for j in range(len(consonants)):
                y, fs = librosa.load("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav", sr=None)
                # print("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".wav")
                time_stamp_list = open("V" + str(i+1) + consonants[j] + "V" + str(i+1) + "_" + str(k+1) + ".txt").read()
                time_stamp_list = string_to_matrix(time_stamp_list)

                time_stamp = np.zeros(shape=(2, 2), dtype=float)
                time_stamp[0, 0] = float(time_stamp_list[0][0])
                time_stamp[0, 1] = float(time_stamp_list[0][1])
                time_stamp[1, 0] = float(time_stamp_list[1][0])
                time_stamp[1, 1] = float(time_stamp_list[1][1])

                start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
                end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
                start_index = int(start_time*fs)
                end_index = int(end_time*fs)

                y = y[start_index:end_index]

                y_stft = n_fft(y, N)
                test_feature.append(y_stft)
                if consonants[j] == "p":
                    test_class.append(1)
                if consonants[j] == "t":
                    test_class.append(2)
                if consonants[j] == "k":
                    test_class.append(3)
                if consonants[j] == "b":
                    test_class.append(4)
                if consonants[j] == "d":
                    test_class.append(5)
                if consonants[j] == "g":
                    test_class.append(6)

    # print(len(test_feature))
    # print(len(test_class))
    # print(test_class)
    test_class_ct1 = knn_function_ct4b_ct1(train_feature, test_feature, train_class)
    # print(len(test_class_ct1))
    # print(len(test_class))
    # print(len(train_class))
    # print(len(train_class_ct2))
    # print(test_class_ct1)
    final_test_class = []
    for i in range(len(test_class_ct1)):
        if test_class_ct1[i] == 1:
            ct2_class = knn_function_ct4b_ct2(train_feature_ct2, test_feature[i], train_class_ct2)
            final_test_class.append(ct2_class)
        elif test_class_ct1[i] == 2:
            ct3_class = knn_function_ct4b_ct3(train_feature_ct3, test_feature[i], train_class_ct3)
            final_test_class.append(ct3_class)
    # print(final_test_class)
    # print(test_class)

    accuracy_count = 0
    for i in range(len(test_class)):
        if test_class[i] == final_test_class[i]:
            accuracy_count = accuracy_count + 1
    accuracy = accuracy_count/len(test_class)
    print(f"Accuracy is {accuracy}")

    return



def n_fft(y, N):
    length = len(y)
    hop_len = 160
    win_len = 320
    n_loop = int((length - hop_len)/(win_len - hop_len))
    nfft = np.zeros(shape=(int(N/2), n_loop))
    for i in range(n_loop):
        start_index = i*hop_len
        end_index = i*hop_len + win_len
        y_segment = y[start_index:end_index]
        ft = abs(fft(x=y_segment, n=N))
        nfft[:, i] = ft[:int(N/2)]


    return nfft


def knn_function_ct1(train_feature, test_feature, train_class, test_class):
    test_class_knn = []

    for i in range(len(test_feature)):
        dict = {}
        for j in range(len(train_feature)):
            distance = DTW(np.array(test_feature[i]), np.array(train_feature[j]))

            dict[distance] = train_class[j]

        count = 0
        num_class1 = 0
        num_class2 = 0
        for k in sorted(dict):
            if dict[k] == 1:
                num_class1 = num_class1 + 1
            if dict[k] == 2:
                num_class2 = num_class2 + 1
            count = count+1
            if count >= 10:
                break
        if num_class1>=num_class2:
            test_class_knn.append(1)
        else:
            test_class_knn.append(2)
    # print(test_class_knn)
    # print(test_class)

    accuracy_count = 0
    for i in range(len(test_class)):
        if test_class[i] == test_class_knn[i]:
            accuracy_count = accuracy_count + 1
    accuracy = accuracy_count/len(test_class)
    print(f"Accuracy is {accuracy}")

    return


def knn_function_ct2(train_feature, test_feature, train_class, test_class):
    test_class_knn = []

    for i in range(len(test_feature)):
        dict = {}
        for j in range(len(train_feature)):
            distance = DTW(np.array(test_feature[i]), np.array(train_feature[j]))

            dict[distance] = train_class[j]

        count = 0
        num_class1 = 0
        num_class2 = 0
        num_class3 = 0
        for k in sorted(dict):
            if dict[k] == 1:
                num_class1 = num_class1 + 1
            if dict[k] == 2:
                num_class2 = num_class2 + 1
            if dict[k] == 3:
                num_class3 = num_class3 + 1
            count = count + 1
            if count >= 10:
                break

        max_class = max(num_class1, num_class2, num_class3)
        if num_class1 == max_class:
            test_class_knn.append(1)
        elif num_class2 == max_class:
            test_class_knn.append(2)
        else:
            test_class_knn.append(3)

    accuracy_count = 0
    for i in range(len(test_class)):
        if test_class[i] == test_class_knn[i]:
            accuracy_count = accuracy_count + 1
    accuracy = accuracy_count / len(test_class)
    # print(test_class_knn)
    # print(test_class)
    print(f"Accuracy is {accuracy}")

    return


def knn_function_ct3(train_feature, test_feature, train_class, test_class):
    test_class_knn = []

    for i in range(len(test_feature)):
        dict = {}
        for j in range(len(train_feature)):
            distance = DTW(np.array(test_feature[i]), np.array(train_feature[j]))

            dict[distance] = train_class[j]

        count = 0
        num_class1 = 0
        num_class2 = 0
        num_class3 = 0
        for k in sorted(dict):
            if dict[k] == 1:
                num_class1 = num_class1 + 1
            if dict[k] == 2:
                num_class2 = num_class2 + 1
            if dict[k] == 3:
                num_class3 = num_class3 + 1
            count = count + 1
            if count >= 10:
                break

        max_class = max(num_class1, num_class2, num_class3)
        if num_class1 == max_class:
            test_class_knn.append(1)
        elif num_class2 == max_class:
            test_class_knn.append(2)
        else:
            test_class_knn.append(3)

    accuracy_count = 0
    for i in range(len(test_class)):
        if test_class[i] == test_class_knn[i]:
            accuracy_count = accuracy_count + 1
    accuracy = accuracy_count / len(test_class)
    print(f"Accuracy is {accuracy}")

    return


def knn_function_ct4a(train_feature, test_feature, train_class, test_class):
    test_class_knn = []

    for i in range(len(test_feature)):
        dict = {}
        for j in range(len(train_feature)):
            distance = DTW(np.array(test_feature[i]), np.array(train_feature[j]))

            dict[distance] = train_class[j]

        count = 0
        num_class1 = 0
        num_class2 = 0
        num_class3 = 0
        num_class4 = 0
        num_class5 = 0
        num_class6 = 0
        for k in sorted(dict):
            if dict[k] == 1:
                num_class1 = num_class1 + 1
            if dict[k] == 2:
                num_class2 = num_class2 + 1
            if dict[k] == 3:
                num_class3 = num_class3 + 1
            if dict[k] == 4:
                num_class4 = num_class4 + 1
            if dict[k] == 5:
                num_class5 = num_class5 + 1
            if dict[k] == 6:
                num_class6 = num_class6 + 1
            count = count + 1
            if count >= 10:
                break
        max_num_class = max(num_class1, num_class2, num_class3, num_class4, num_class5, num_class6)
        if num_class1 == max_num_class:
            test_class_knn.append(1)
        elif num_class2 == max_num_class:
            test_class_knn.append(2)
        elif num_class3 == max_num_class:
            test_class_knn.append(3)
        elif num_class4 == max_num_class:
            test_class_knn.append(4)
        elif num_class5 == max_num_class:
            test_class_knn.append(5)
        else:
            test_class_knn.append(6)

    accuracy_count = 0
    for i in range(len(test_class)):
        if test_class[i] == test_class_knn[i]:
            accuracy_count = accuracy_count + 1
    accuracy = accuracy_count / len(test_class)
    # print(test_class_knn)
    # print(test_class)
    print(f"Accuracy is {accuracy}")

    return


def knn_function_ct4b_ct1(train_feature, test_feature, train_class):
    test_class_knn = []

    for i in range(len(test_feature)):
        dict = {}
        for j in range(len(train_feature)):
            distance = DTW(np.array(test_feature[i]), np.array(train_feature[j]))

            dict[distance] = train_class[j]

        count = 0
        num_class1 = 0
        num_class2 = 0
        for k in sorted(dict):
            if dict[k] == 1:
                num_class1 = num_class1 + 1
            if dict[k] == 2:
                num_class2 = num_class2 + 1
            count = count+1
            if count >= 10:
                break
        if num_class1 >= num_class2:
            test_class_knn.append(1)
        else:
            test_class_knn.append(2)

    return test_class_knn

def knn_function_ct4b_ct2(train_feature_ct2, test_feature, train_class_ct2):

    dict = {}
    for j in range(len(train_feature_ct2)):
        distance = DTW(np.array(test_feature), np.array(train_feature_ct2[j]))
        dict[distance] = train_class_ct2[j]

    count = 0
    num_class1 = 0
    num_class2 = 0
    num_class3 = 0
    for k in sorted(dict):
        if dict[k] == 1:
            num_class1 = num_class1 + 1
        if dict[k] == 2:
            num_class2 = num_class2 + 1
        else:
            num_class3 = num_class3 + 1
        count = count+1
        if count >= 10:
            break
    max_class = max(num_class1, num_class2, num_class3)
    if num_class1 == max_class:
        test_class_knn = 1
    elif num_class2 == max_class:
        test_class_knn = 2
    else:
        test_class_knn = 3
    return test_class_knn


def knn_function_ct4b_ct3(train_feature_ct2, test_feature, train_class_ct2):

    dict = {}
    for j in range(len(train_feature_ct2)):
        distance = DTW(np.array(test_feature), np.array(train_feature_ct2[j]))
        dict[distance] = train_class_ct2[j]

    count = 0
    num_class1 = 0
    num_class2 = 0
    num_class3 = 0
    for k in sorted(dict):
        if dict[k] == 1:
            num_class1 = num_class1 + 1
        if dict[k] == 2:
            num_class2 = num_class2 + 1
        else:
            num_class3 = num_class3 + 1
        count = count+1
        if count >= 10:
            break
    max_class = max(num_class1, num_class2, num_class3)
    if num_class1 == max_class:
        test_class_knn = 4
    elif num_class2 == max_class:
        test_class_knn = 5
    else:
        test_class_knn = 6
    return test_class_knn


def DTW(X, Y):
    n = X.shape[1]
    m = Y.shape[1]
    min_val = min(n, m)
    max_val = max(n, m)
    dtw_matrix, w = librosa.sequence.dtw(X=X, Y=Y, C=None, subseq=True, metric='euclidean')
    #dtw_matrix, w = librosa.sequence.dtw(X=X, Y=Y, C=None, subseq=True, metric=itakura_saito)
    return dtw_matrix[min_val-1, max_val-1]


def itakura_saito(s1, s2):
    return np.sum(s1/s2 - np.log(s1/s2) - 1)


def string_to_matrix(string):
    line_split = list(string.split("\n"))
    matrix = []

    for item in line_split:
        line = []
        for data in item.split("\t"):
            line.append(data)
        matrix.append(line)

    return matrix


def main():

    classification_task_1()

    return


if __name__ == '__main__':
    main()
