from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

mini_df2 = mini_df.copy()
for col in mini_df.columns:

    mini_df2 = fp.encode_diagnoses(mini_df2, col, col)

x = mini_df2.drop(['DIAGNOSIS_SHORT_1', 'AGE', 'INJURY', 'DIAGNOSIS_LONG_1', 'DIAGNOSIS_LONG_2', 'DIAGNOSIS_LONG_3'
			,'DIAGNOSIS_SHORT_2', 'DIAGNOSIS_SHORT_3'], 1)

y = mini_df2['DIAGNOSIS_SHORT_1']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
pca = PCA()
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_