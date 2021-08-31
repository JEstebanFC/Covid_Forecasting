- Prediction:
    1. Linear Regression
    2. Polynomial regression with lasso
    3. Simple and Exponential Smoothing 
    4. Holt's Winter Model
    5. Dickey-Fuller test
    6. AR, MA and ARIMA Models
    7. Auto AR, MA, ARIMA and SARIMA Models
    8. AR, MA, ARIMA, SARIMA  using VARMAX Modeling
    9. Facebook's Prophet Model

- DeepLearning:
    1. LSTM
    2. CNN
    3. LSTM + CNN
    4. CONV2LSTM

###### Installation ######
## Jupyter ##
pip install jupyterlab
pip install notebook
pip install voila
pip install ipython

pip install --upgrade pip
pip install --upgrade setuptools

pip install datetime
pip install numpy
pip install pandas
pip install matplotlib
pip install -U kaleido
pip install -U scikit-learn
pip install tslearn
pip install pmdarima
pip install tensorflow

## FBProphet ##
python -m pip install pystan==2.17.1.0
python -m pip install fbprophet==0.6   
python -m pip install --upgrade fbprophet
pip install --upgrade plotly


Confirm:
    pip show tensorflow
    pip list | grep tensorflow
    pip list | findstr tensorflow