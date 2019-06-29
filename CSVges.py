from scipy.misc import imread
import numpy as np
import pandas as pd
import os
for (directory, subd, files) in os.walk('./ges', topdown=True):
    print('dir:',directory)
    print('sub-d:',subd)
    print(len(files))
    if len(files)>1:
        for file in files:
            im = imread(os.path.join(directory,file))
            value = im.flatten()
            value = np.hstack((directory[1:],value))
            print('values:',value)
            df = pd.DataFrame(value).T
            df = df.sample(frac=1)
            with open('train_ges.csv', 'a') as dataset:
                df.to_csv(dataset, header=False, index=False)