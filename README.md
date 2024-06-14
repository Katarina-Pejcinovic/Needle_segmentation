
# Needle Segmentation Project Documentation

This documentation provides instructions on how to train and make predictions with the code available at: [https://github.com/Katarina-Pejcinovic/Needle_segmentation](https://github.com/Katarina-Pejcinovic/Needle_segmentation)

****This code and environment will ONLY run on an Apple machine with Apple silicon chip****

**Project Setup**



1. Clone the repository

   ```
    git clone https://github.com/Katarina-Pejcinovic/Needle_segmentation.git
   ```


2. The data and model weights are available on Google Drive upon request. 

    The folder contains: 

* A `data.zip` file with data (already preprocessed and additional augmented data as well as the test set) 
* A `final_model.h5` file with the model weights
3. Unzip the data folder. Add the `data` folder and the `final_model.h5` file to the cloned `needle_segmentation` github folder. The final paths should be `needle_segmentation/data` and `needle_segmentation/final_model.h5 `respectively. 
4. An `environment.yml` file is in the GitHub project. It contains a conda environment with all the necessary packages to run the code. 
* To activate the environment:
    * Install either anaconda or miniconda 
    * Navigate to the project directory and run:

            
            conda env create -f environment.yml
            


    * Then run: 

            
            conda activate needle_env

            


**Training the Model**

_Cross-validation_

To train and perform cross-validation on the training data, run the `U_Net.py` script. Make sure you are running from the project directory and the conda environment is active. `U_Net.py` will output a `dataframe_figure.png` and `epoch_loss.png`. 

_Training on Full Dataset_

To train the model on the full dataset, run `U_Net_final.py`. The script will save the final weights in a file called `final_model.h5` that can be used in prediction. 

You can expect training to take at least 1-2 hours. If you do not wish to run the whole training script, you may use the already generated weights from Google Drive and follow the instructions below. 

**Predicting with Model**

Once you have created the weights or you are using the weights already provided:



1. In terminal, run:

    ```
    python U_Net_predictions.py
    ```



    This will output a directory called `output_images_test` with the predicted masks for the test set in the project folder. 

2. Next, run: 

    ```
    python postprocessing.py
    ```



    This will output a directory called `output_images_postprocessing`

3. Finally, run:

    ```
    python process_images.py
    ```



    This will create a `submission.csv` file that can be uploaded to Kaggle. 


    That’s it!
