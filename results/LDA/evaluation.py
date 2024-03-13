import pandas as pd
from sklearn import metrics

data = pd.read_csv('udn-news-topics-lda.csv')
labels_true = data['label'].tolist()[149:]
labels_pred = data['topic'].tolist()[149:]

print(metrics.homogeneity_score(labels_true, labels_pred))  # 0.2638309111292339
print(metrics.completeness_score(labels_true, labels_pred))  # 0.24769998257728482
print(metrics.v_measure_score(labels_true, labels_pred))  # 0.2555111055620984
print(metrics.adjusted_rand_score(labels_true, labels_pred))  # 0.05189511898035007
print(metrics.normalized_mutual_info_score(labels_true, labels_pred))  # 0.2555111055620984

