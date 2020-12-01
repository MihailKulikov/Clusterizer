import pandas as pd

import CNN
from LoadDataset import get_dataset

all_data = get_dataset('./Orbits')
patient_features = CNN.get_patients_features(CNN.get_model('efficientnet-b0'), all_data)
patient_features = pd.DataFrame(patient_features)
patient_features.columns = ['f'+str(x) for x in range(1, patient_features.shape[1] + 1)]

