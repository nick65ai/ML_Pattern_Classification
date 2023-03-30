
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re

class BirdsData:
    def __init__(self, project_folder):
        self.project_folder = project_folder

    # simply retrieve the features and corresponding label for given sample of specified feature
    def get_sound_label(self, species, sample):
        folder_path = os.path.join(self.project_folder, species)
        file_list = os.listdir(folder_path)
        for file in file_list:
            if file.endswith('.npy'):
                file_path = os.path.join(folder_path, sample)
                data = np.load(f'{file_path}.npy')
                label = np.load(f'{file_path}.labels.npy')
                return np.concatenate((data, label[:, 0].reshape(label.shape[0], 1)), axis=1)


    #method to create one huge dataset
    def united_dataset(self):
        birds = [os.path.join(self.project_folder, b) for b in os.listdir(self.project_folder)]
        all_files = [os.path.join(bird, f) for bird in birds for f in os.listdir(bird)]
        all_files = sorted([f for f in all_files if not 'labels' in f])


        comcuc = [f for f in all_files if 'comcuc' in f]
        comcuc_features = [str(re.search(r"\d+(?=.npy)", s).group(0)) for s in comcuc]
        comcuc_ars = []
        for f in comcuc_features:
            comcuc_ars.append(BirdsData(self.project_folder).get_sound_label('comcuc', f))
        comcuc_ars = np.concatenate(comcuc_ars)

        cowpig1 = [f for f in all_files if 'cowpig1' in f]
        cowpig1_features = [str(re.search(r"\d+(?=.npy)", s).group(0)) for s in cowpig1]
        cowpig1_ars = []
        for f in cowpig1_features:
            cowpig1_ars.append(BirdsData(self.project_folder).get_sound_label('cowpig1', f))
        cowpig1_ars = np.concatenate(cowpig1_ars)

        eucdov = [f for f in all_files if 'eucdov' in f]
        eucdov_features = [str(re.search(r"\d+(?=.npy)", s).group(0)) for s in eucdov]
        eucdov_ars = []
        for f in eucdov_features:
            eucdov_ars.append(BirdsData(self.project_folder).get_sound_label('eucdov', f))
        eucdov_ars = np.concatenate(eucdov_ars)

        eueowl1 = [f for f in all_files if 'eueowl1' in f]
        eueowl1_features = [str(re.search(r"\d+(?=.npy)", s).group(0)) for s in eueowl1]
        eueowl1_ars = []
        for f in eueowl1_features:
            eueowl1_ars.append(BirdsData(self.project_folder).get_sound_label('eueowl1', f))
        eueowl1_ars = np.concatenate(eueowl1_ars)

        grswoo = [f for f in all_files if 'grswoo' in f]
        grswoo_features = [str(re.search(r"\d+(?=.npy)", s).group(0)) for s in grswoo]
        grswoo_ars = []
        for f in grswoo_features:
            grswoo_ars.append(BirdsData(self.project_folder).get_sound_label('grswoo', f))
        grswoo_ars = np.concatenate(grswoo_ars)

        tawowl1 = [f for f in all_files if 'tawowl1' in f]
        tawowl1_features = [str(re.search(r"\d+(?=.npy)", s).group(0)) for s in tawowl1]
        tawowl1_ars = []
        for f in tawowl1_features:
            tawowl1_ars.append(BirdsData(self.project_folder).get_sound_label('tawowl1', f))
        tawowl1_ars = np.concatenate(tawowl1_ars)


        all_birdiiies = np.concatenate((comcuc_ars, cowpig1_ars, eucdov_ars, eueowl1_ars, grswoo_ars, tawowl1_ars))

        return all_birdiiies


    def pca(self):
        scaler = StandardScaler()
        dataset = BirdsData(self.project_folder).united_dataset()
        X = dataset[:, 0:-1]
        y = dataset[:, -1]
        scaler.fit(X)
        X_scaled = scaler.transform(X)

    def labels_distribution(self):
        dataset = BirdsData(self.project_folder).united_dataset()
        others_count = len(dataset[dataset[:, -1] == 0])
        class_counts = []
        for i in range(0, 7):
            class_counts.append(len(dataset[dataset[:, -1] == i]))
        x = ['other', 'comcuc', 'cowpig1', 'eucdov', 'eueowl1', 'grswoo', 'tawowl1']
        y = class_counts
        plt.figure(figsize=(10, 7))
        plt.plot(x, y)
        plt.title('Plot of labels distribution')
        plt.xlabel('Species (labels)')
        plt.ylabel('Samples')
        plt.show()

        return class_counts

    # potential method to complete "annotators agreement" part. Returns general agreement, positive agreement (how
    # many annotators think positive class to be positive, negative agreement (how many annotators think "others" part
    # to be "others"

    def compute_agreement(self, species):
        folder_path = os.path.join(self.project_folder, species)
        file_list = os.listdir(folder_path)
        labels = [f for f in file_list if 'labels' in f]
        labels_paths = [os.path.join(folder_path, i) for i in labels]
        labels_ar_list = []
        for i in range(len(labels_paths)):
            labels_ar_list.append(np.load(labels_paths[i]))
        general_agreements = 0
        for i in range(len(labels_ar_list)):
            for j in range(0, 100):
                # index of label which occurs the most
                general_agreements += np.mean(labels_ar_list[i][j] == np.argmax(np.bincount(labels_ar_list[i][j][1:])))

        positive_agreements = 0
        count_p = 0
        negative_agreements = 0
        count_n = 0
        for i in range(len(labels_ar_list)):
            for j in range(0, 100):
                if labels_ar_list[i][j][0] == 0:
                    negative_agreements += np.mean(
                        labels_ar_list[i][j] == np.argmax(np.bincount(labels_ar_list[i][j][1:])))
                    count_n += 1
                else:
                    positive_agreements += np.mean(
                        labels_ar_list[i][j] == np.argmax(np.bincount(labels_ar_list[i][j][1:])))
                    count_p += 1

        general_agreements = general_agreements / (len(labels) * 100)
        positive_agreements = positive_agreements / count_p
        negative_agreements = negative_agreements / count_n
        return np.array([general_agreements, positive_agreements, negative_agreements])

    # Method below might be of use to further calculate which features are important and which are not.


    def compute_feature_correlations_for_species(self, species):
        #folder_path = f'{self.project_folder}/{species}'
        #file_list = os.listdir(f'{self.project_folder}/{species}')
        folder_path = os.path.join(self.project_folder, species)
        file_list = os.listdir(folder_path)
        features = [f for f in file_list if not 'labels' in f]
        cor_mat_list = []
        file_paths = [os.path.join(folder_path, i) for i in features]
        for i in range(len(file_paths)):
            cor_mat_list.append(np.corrcoef(np.load(file_paths[i]).T))
        return np.mean(cor_mat_list, axis=0)





    def plot_cor_distribution(self, species, save = False):
        plt.figure(figsize=(14, 10))
        y = np.unique(BirdsData(self.project_folder).compute_feature_correlations_for_species(species))
        plt.plot(y)
        plt.grid()
        plt.show()
        if save:
            plt.savefig(self.project_folder)



#print(BirdsData('ptichki').labels_distribution())
