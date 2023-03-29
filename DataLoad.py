import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class BirdsData:
    def __init__(self, project_folder):
        self.project_folder = project_folder


    def get_sound_label(self, species, sample):
        folder_path = f'{self.project_folder}/{species}'
        file_list = os.listdir(f'{self.project_folder}/{species}')
        for file in file_list:
            if file.endswith('.npy'):
                file_path = os.path.join(folder_path, sample)
                data = np.load(f'{file_path}.npy')
                label = np.load(f'{file_path}.labels.npy')
                return np.array(data), np.array(label)

    def compute_agreement(self, species):
        folder_path = f'{self.project_folder}/{species}'
        file_list = os.listdir(f'{self.project_folder}/{species}')
        labels = [f for f in file_list if 'labels' in f]
        labels_paths = [os.path.join(folder_path, i) for i in labels]
        labels_ar_list = []
        for i in range(len(labels_paths)):
            labels_ar_list.append(np.load(labels_paths[i]))
        general_agreements = 0
        for i in range(len(labels_ar_list)):
            for j in range(0, 100):
                                                              #index of label which occurs the most
                general_agreements += np.mean(labels_ar_list[i][j] == np.argmax(np.bincount(labels_ar_list[i][j][1:])))

        positive_agreements = 0
        count_p = 0
        negative_agreements = 0
        count_n = 0
        for i in range(len(labels_ar_list)):
            for j in range(0, 100):
                if labels_ar_list[i][j][0] == 0:
                    negative_agreements += np.mean(labels_ar_list[i][j] == np.argmax(np.bincount(labels_ar_list[i][j][1:])))
                    count_n += 1
                else:
                    positive_agreements += np.mean(labels_ar_list[i][j] == np.argmax(np.bincount(labels_ar_list[i][j][1:])))
                    count_p += 1

        general_agreements = general_agreements/(len(labels) * 100)
        positive_agreements = positive_agreements/count_p
        negative_agreements = negative_agreements/count_n
        return np.array([general_agreements, positive_agreements, negative_agreements])


    def compute_feature_correlations_for_species(self, species):
        folder_path = f'{self.project_folder}/{species}'
        file_list = os.listdir(f'{self.project_folder}/{species}')
        features = [f for f in file_list if not 'labels' in f]
        cor_mat_list = []
        file_paths = [os.path.join(folder_path, i) for i in features]
        for i in range(len(file_paths)):
            cor_mat_list.append(np.corrcoef(np.load(file_paths[i]).T))
        return np.mean(cor_mat_list, axis=0)





# print(BirdsData('ptichki').get_sound_label('comcuc', '25627'))
# print(BirdsData('ptichki').compute_agreement('comcuc'))
# print(BirdsData('ptichki').compute_agreement('cowpig1'))
# print(BirdsData('ptichki').compute_agreement('eucdov'))
# print(BirdsData('ptichki').compute_agreement('eueowl1'))
# print(BirdsData('ptichki').compute_agreement('grswoo'))
# print(BirdsData('ptichki').compute_agreement('tawowl1'))
print(((BirdsData('ptichki').compute_feature_correlations_for_species('comcuc'))).shape)