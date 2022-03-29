# electric_forecasting
Electric Forecasting (DSAI HW1)
## Data
Features:
1. Operating reserve
2. Percent operating reserve
3. Highest temperature of Taipei
4. Hghest temperature of Kaohsiung 
## Data Preprocessing
1. Normalization
Use sklearn.preprocessing.MinMaxScaler to normalize the data. The MinMaxScaler fit the data into range [1,0].
```python
sc = MinMaxScaler(feature_range=(0,1))
data = sc.fit_transform(data)
```
Since the output also in range [0,1], we inverse the scale of output to fit the original data.
```python
data = sc.inverse_transform(data)
```
2. Data smoothing
There are many peak value in the original data, so we apply moving average technique to get better prediction.
```python
def moving_avg(self,data):
 day = 7
 avg = []
 for i in range(day,data.shape[0]-day):
  avg.append(data.iloc[i-day:i,:].mean())
 return avg
```
3. Split training and validation set
## Model
Use LSTM as training model. Below is the architecture of LSTM:
```python
self.model = Sequential()
self.model.add(LSTM(128, activation = "relu", return_sequences = True, input_shape = (X_train.shape[1], self.feature_num)))
self.model.add(LSTM(128, activation = "relu", return_sequences = True))
self.model.add(Dropout(0.2))
self.model.add(LSTM(128, activation = "relu"))
self.model.add(Dropout(0.2))
self.model.add((Dense(1)))
opt = keras.optimizers.Adam(learning_rate=LR)
self.model.compile(optimizer = opt, loss = 'mse')
self.model.summary()
callback = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="auto")
history = self.model.fit(X_train, y_train, epochs = epoch, batch_size = 1, validation_data=(X_val, y_val), callbacks=[callback])
```
1. Add dropout layers to avoid overfitting
2. Use Adam optimizer 
3. Apply early stopping by monitor validation loss to avoid over training
4. Store the trainging history to trace training result
