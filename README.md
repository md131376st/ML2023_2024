# ML2023_2024
The project task consists of a binary classification problem. The goal is to perform fingerprint spoofing
detection, i.e. to identify genuine vs counterfeit fingerprint images. The dataset consists of labeled
samples corresponding to the genuine (True, label 1) class and the fake (False, label 0) class. The
samples are computed by a feature extractor that summarizes high-level characteristics of a fingerprint
image. The data is 6-dimensional.
The training files for the project are stored in file Data/Train.txt. The format of the file is
the same as for the Iris dataset, i.e. a csv file where each row represents a sample. The first 6 values of
each row are the features, whereas the last value of each row represents the class (1 or 0). The samples
are not ordered.