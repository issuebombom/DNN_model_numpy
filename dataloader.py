import numpy as np

# DataLoader
def shuffle_data(X, Y):
    """Data와 Label을 concat 후 shuffle -> Data, Label 분리
    """
    
    concat = np.concatenate((X, Y), axis=1)
    np.random.shuffle(concat)

    X = concat[:, :-Y.shape[1]]
    Y = concat[:, -Y.shape[1]:]

    return X, Y


def train_validation_split(X, Y, val_size, seed=None, shuffle=False):
    """train_validation_split
    
    batch단위 split에서 데이터 셔플링을 진행하므로
    train_val_split 단계에서는 suffle=False를 Default로 설정
    """
    
    np.random.seed(seed)
    
    if shuffle:
        X, Y = shuffle_data(X, Y)
    
    boundary = int(X.shape[0] * (1-val_size))

    X_train = X[:boundary]
    Y_train = Y[:boundary]
    X_val = X[boundary:]
    Y_val = Y[boundary:]
    
    return X_train, Y_train, X_val, Y_val


def split_into_batches(X, Y, batch_size, drop=True, seed=None, shuffle=True):
    """batch size 단위로 데이터를 reshape합니다.
    """
    
    np.random.seed(seed)
    
    if shuffle:
        X, Y = shuffle_data(X, Y)
    
    # batch_size가 딱 맞아 떨어지지 않을 경우
    if X.shape[0] % batch_size != 0:
        if drop: # 나머지 뒷단의 데이터를 버린다.
            num_to_select = X.shape[0] // batch_size * batch_size # 맞아떨어지는 개수
            X, Y = X[:num_to_select], Y[:num_to_select]
            
        else: # 데이터를 추가한다.(랜덤 추출해서)
            num_to_fill = (X.shape[0] // batch_size + 1) * batch_size - X.shape[0] # 추가로 필요한 개수
            indices_to_add = np.random.choice(range(0, X.shape[0]+1), num_to_fill, replace=False) # 추가할 인덱스 랜덤 선정
            X_to_add, Y_to_add = X[indices_to_add], Y[indices_to_add] # 추가할 데이터셋
            X, Y = np.concatenate((X, X_to_add)), np.concatenate((Y, Y_to_add)) # 기존 데이터에 추가
    
    X_batch_datasets = X.reshape(-1, batch_size, X.shape[-1]) # reshape (iter, bs, data)
    Y_batch_datasets = Y.reshape(-1, batch_size, Y.shape[-1]) # reshape (iter, bs, labels)
    
    return X_batch_datasets, Y_batch_datasets