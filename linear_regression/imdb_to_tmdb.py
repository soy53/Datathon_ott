import random
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

random.seed(42)

def train(data_path):
    
    df = pd.read_csv(data_path)
    print(df.shape)
    print(df.columns)
    
    imdb_score = df['imdb_score']
    tmdb_score = df['tmdb_score']
            
    trainval = []
    test_dict = dict()
    
    # imdb_score와 tmdb_score 모두 있는 경우 trainval에 추가
    # imdb_score는 있으나 tmdb_score 없는 경우 test에 추가 -> 결측치 예측 대상
    for idx, _ in enumerate(imdb_score):
        
        x = imdb_score[idx]
        y = tmdb_score[idx]
        if np.isnan(x):
            continue
        
        if np.isnan(y):
            # test.append((idx, x, y))  # idx: row id
            test_dict[idx] = x
        else:            
            trainval.append((x, y))
                        
    print(f'both imdb and tmdb score exist: {len(trainval)}')            
    print(f'only imdb score exists: {len(test_dict)}')
    
    # train / validation split
    random.shuffle(trainval)
    split_ratio = 0.2
    split_index = int(split_ratio * len(trainval))
    train = trainval[split_index:]
    val = trainval[:split_index]

    print(f'number of train data: {len(train)}')
    print(f'number of validation data: {len(val)}')
    
    train_X = np.array([item[0] for item in train]).reshape(-1, 1)
    train_Y = np.array([item[1] for item in train]).reshape(-1, 1)
    val_X = np.array([item[0] for item in val]).reshape(-1, 1)
    val_Y = np.array([item[1] for item in val]).reshape(-1, 1)
    test_X = np.array([test_dict[key] for key in test_dict]).reshape(-1, 1)
    
    # 학습
    model = LinearRegression()
    model.fit(train_X, train_Y)
    
    # validation 데이터에 대하여 성능 (MSE) 측정
    val_predicted = model.predict(val_X)
    mse = mean_squared_error(val_Y, val_predicted)
    print(f'validation MSE: {mse}')
    
    # test 데이터에 대하여 예측 (결측치 예측)
    test_predicted = model.predict(test_X)
    # print(test_predicted)

    cnt = 0
    for idx, _ in enumerate(imdb_score):
        if idx in test_dict:
            df.loc[idx, 'prediction'] = round(test_predicted[cnt][0], 5)
            cnt += 1
        else:
            df.loc[idx, 'prediction'] = None
    
    df.to_csv('amazon_imdb_to_tmdb_prediction.csv')
                

if __name__ == '__main__':
    
    data_path = r'C:\Users\Jennie\Desktop\aiffel\Datathon\linear_regression\data\amazon.csv'
    train(data_path)