import pandas as pd

class DataImporter:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        data = pd.read_csv(f'{self.filepath}/train/X_train_update.csv')
        data['description'] = data['designation'] + str(data['description'])
        data = data.drop(['Unnamed: 0', 'designation'], axis=1)

        target = pd.read_csv(f'{self.filepath}/train/Y_train_CVw08PX.csv')
        target = target.drop(['Unnamed: 0'], axis=1)
        modalite_mapping = {modalite: i for i, modalite in enumerate(target['prdtypecode'].unique())}
        target['prdtypecode'] = target['prdtypecode'].replace(modalite_mapping)

        df = pd.concat([data, target], axis=1)

        return df

    def split_train_test(self, df, samples_per_class = 600):

        grouped_data = df.groupby('prdtypecode')

        X_train_samples = []
        X_test_samples = []

        for _, group in grouped_data:
            samples = group.sample(n=samples_per_class, random_state=42)
            X_train_samples.append(samples)

            remaining_samples = group.drop(samples.index)
            X_test_samples.append(remaining_samples)

        X_train = pd.concat(X_train_samples)
        X_test = pd.concat(X_test_samples)

        X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        X_test = X_test.sample(frac=1, random_state=42).reset_index(drop=True)

        y_train = X_train['prdtypecode']
        X_train = X_train.drop(['prdtypecode'], axis=1)

        y_test = X_test['prdtypecode']
        X_test = X_test.drop(['prdtypecode'], axis=1)

        val_samples_per_class = 50

        grouped_data_test = pd.concat([X_test, y_test], axis=1).groupby('prdtypecode')

        X_val_samples = []
        y_val_samples = []

        for _, group in grouped_data_test:
            samples = group.sample(n=val_samples_per_class, random_state=42)
            X_val_samples.append(samples[['description', 'productid', 'imageid']])
            y_val_samples.append(samples['prdtypecode'])

        X_val = pd.concat(X_val_samples)
        y_val = pd.concat(y_val_samples)

        X_val = X_val.sample(frac=1, random_state=42).reset_index(drop=True)
        y_val = y_val.sample(frac=1, random_state=42).reset_index(drop=True)

        return X_train, X_val, X_test, y_train, y_val, y_test