class Model():
    # constructor
    def __init__(self):
      pass

    def splitData(self,X,Y,rate):
        X_train = X[int(X.shape[0]*rate):]
        Y_train = Y[int(Y.shape[0]*rate):]
        X_val = X[:int(X.shape[0]*rate)]
        Y_val = Y[:int(Y.shape[0]*rate)]
        return X_train, Y_train, X_val, Y_val

    def moving_avg(self,data):
        day = 16
        avg = []
        for i in range(day,data.shape[0]-day):
          avg.append(data.iloc[i-day:i,:].mean())
        return avg

    # train
    def train(self, data):
        epoch = 100
        self.n_timestamp = 16 # 每次訓練的資料數目
        self.n_predict = self.n_timestamp # 每次預測的資料數目
        self.label = pd.DataFrame(data["備轉容量(萬瓩)"]) # 把 label (備轉容量) 獨立出來
        label = self.label
        data = data.drop(columns=["日期","備轉容量(萬瓩)"]) # 除了 label 及日期以外的 feature
        
        # -------------- data preprocessing --------------
        
        # moving average
        label = self.moving_avg(self.label) # return type list
        data = self.moving_avg(data) # return type list
        
        # normalization
        self.labelsc = MinMaxScaler(feature_range=(0,1))
        label = self.labelsc.fit_transform(label) # return type numpy array
        sc = MinMaxScaler(feature_range=(0,1))
        data = sc.fit_transform(data)
        
        # concate
        new_data = np.concatenate((label, data), axis=1)
        print("new_data shape:", new_data.shape)
        data = new_data
        
        self.feature_num = data.shape[1]
    
        # split train and test
        train_data = data[:-self.n_timestamp,:]
        self.test_data = data[len(data)-self.n_timestamp:,:]
        row = len(train_data)
        
        # -------------- create training data --------------
        X_train = []
        y_train = []
        
        for i in range(self.n_timestamp,row-self.n_timestamp):
            X_train.append(train_data[i-self.n_timestamp:i,:]) # 取預測點前n_timestamp天資料
            y_train.append(train_data[i:i+self.n_predict,:]) # 預測點
        
        # 轉成 numpy array
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train, y_train, X_val, y_val = self.splitData(X_train,y_train,0.2) # split data into train and val

        # 轉成 LSTM 的三維輸入形式
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], self.feature_num))
        
        # -------------- bulid model (LSTM) --------------
        LR = 0.0001
        # model
        self.model = Sequential()
        self.model.add(LSTM(128, activation = "relu", return_sequences = True, input_shape = (X_train.shape[1], self.feature_num)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(128, activation = "relu", return_sequences = True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(128, activation = "relu", return_sequences = True))
        self.model.add(TimeDistributed(Dense(1)))
        # optimizer
        opt = keras.optimizers.Adam(learning_rate=LR)
        self.model.compile(optimizer = opt, loss = 'mse')
        self.model.summary()

        callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="auto")
        history = self.model.fit(X_train, y_train, epochs = epoch, batch_size = 1, 
                                 validation_data=(X_val, y_val), callbacks=[callback])

    # test
    def predict(self):

      X_test = self.test_data
      y_test = self.test_data[0] # 備轉容量

      # 轉成 numpy array
      X_test = np.array([X_test])

      # 轉成 LSTM 的三維輸入形式
      X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], self.feature_num))

      predict_y = self.model.predict(X_test) # predict
      
      # descaled
      predict_descale = self.labelsc.inverse_transform(predict_y[0])
      real = self.label.iloc[len(self.label)-self.n_timestamp:,0].values

      # write the result into csv file
      output = pd.DataFrame()
      time_range = pd.date_range('20220330',periods=self.n_timestamp-1,freq='D')
      output['date'] = time_range
      # 僅保留 3/30 以後的預測結果
      output['operating_reserve(MW)'] = predict_descale[1:]
      return output



    # test
    def predict(self):
      
      X_test = self.test_data
      y_test = self.test_data[0]
      
      '''
      for i in range(self.n_timestamp,len(self.data)-self.n_predict):
        X_test.append(self.data[i-self.n_timestamp:i,0]) # 取預測點前n_timestamp天資料
        y_test.append(self.data[i:i+self.n_predict,0]) # 預測點
      '''

      # 轉成 numpy array
      X_test = np.array([X_test])

      X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], self.feature_num))
      predict_y = self.model.predict(X_test)
      
      # descaled
      predict_descale = self.labelsc.inverse_transform(predict_y[0])
      real = self.label.iloc[len(self.label)-self.n_timestamp:,0].values
      
      # Visualising the results
      plt.plot(real, label = 'Real')
      plt.plot(predict_descale, label = 'Predicted')
      plt.xlabel('Time')
      plt.ylabel('Operating reserve')
      plt.legend()
      plt.show()

      # write the result into csv file
      output = pd.DataFrame()
      #time_range = pd.data_range('20220329',periods=16,freq='D')
      time_range = pd.date_range('20220313',periods=16,freq='D')
      output['date'] = time_range
      output['operating_reserve(MW)'] = predict_descale
      # 僅保留 3/30 以後的預測結果
      # -------- todo -------------

      # output submission.csv
      output.to_csv('submission.csv', index=False)      
      return output
    

# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    # keras libraries
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, TimeDistributed
    from keras.layers import LSTM, Bidirectional
    from keras.callbacks import EarlyStopping
    from tensorflow import keras
    # make result reprodicible
    from numpy.random import seed
    seed(1)
    import tensorflow as tf
    tf.random.set_seed(1)
    # other
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    #from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import scale, MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    df_training = pd.read_csv(args.training)
    model = Model()
    model.train(df_training)
    y, df_result = model.predict()
    df_result.to_csv(args.output, index=0)