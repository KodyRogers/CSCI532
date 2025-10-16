This project provides a pipeline for downloading, cleaning, and partitioning the PathMNIST dataset
into client-specific datasets for experiments such as federated learning or distributed training.

To Run this Code

First install the dependencies using 
pip install -r requirements.txt

Second Run
clean_data.py
This will take a while as it needs to download the data if it isn't provided
This will also normalize the data and clean it by removing duplicates and blanks

Third run
convert_clients.py
This will put the data into the client_data folder which will partition it and label the data