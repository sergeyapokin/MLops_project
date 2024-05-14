**LSTM for Time Series Prediction**

Long Short-Term Memory (LSTM) is a structure that can be used in neural network. It is a type of recurrent neural network (RNN) that expects the input in the form of a sequence of features. It is useful for data such as time series or string of text. We will use LTSM to forecast sales of certain products based on historical data.


**Waiting for input**

Since the LSTM cell expects the input  in the form of multiple time steps, each input sample should be a 2D tensors: One dimension for time and another dimension for features. The power of an LSTM cell depends on the size of the hidden state or cell memory, which usually has a larger dimension than the number of features in the input.

**[Dataset](https://www.kaggle.com/datasets/apoorvaappz/global-super-store-dataset?select=Global_Superstore2.csvhttps%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fapoorvaappz%2Fglobal-super-store-dataset%3Fselect%3DGlobal_Superstore2.csv)**

Shopping online is currently the need of the hour. Because of this COVID, it's not easy to walk in a store randomly and buy anything you want. Once you download the file the rows you see are the details of the order done online by people across the globe in the time frame 1-jan-2011 to 31-dec-2014. There are no missing values in the majority of columns except postal code, you can drop it if not required.

**Pipeline**

<img width="714" alt="Снимок экрана 2024-05-14 в 15 31 08" src="https://github.com/sergeyapokin/MLops_project/assets/153228526/774512fa-5b96-4626-a5fa-e27de4aff7c8">

The project consists of an infer.py and a train.py. The main launch of the project comes from the infer.py, which implements loading the dataset and dividing it into train and test. To run it, you need to create an env using requirements or build a DockerFile and using the created environment you can run our project.