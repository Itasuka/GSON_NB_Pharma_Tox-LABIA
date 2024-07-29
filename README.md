# 2 months GSON stage with NB Pharma Tox

In this stage I discovered and experienced an IA tools formation. Indeed, in this stage I learned the basis of deep learning models mechanisms with online lectures.


## Use DeepLabCut software to create datas

### An acceleration / decceleration model
In order to get more data such as the acceleration and decceleration of a mouse from open field videos, I used a open source software named DeepLabCut (https://deeplabcut.github.io/DeepLabCut/README.html). This software is used to create deep learning models that take a video in parameter and create a .csv with the position of any part of the animal the model is trained to recognize (we can chose which part of the animal we want to recognize).

To train and use the deep learning model more fluently I installed CUDA and pytorch in order to use my GPU instead of my CPU to significantly decrease my training and evaluating time.

The file **model.ipynb** contains all the code that is needed to create and train the model with DeepLabCut. To create this code, I followed the DeepLabCut software official guide.

I created a model that recognize the head, the body and the tail of the mouse in order to make some acceleration calculation. To train this model, I used open field videos from which DeepLabCut extract each frame to give them one by one to the model which predict the position of the head, body and tail.

This model worked mostly well and may only need more training to be more consistent (the model may be wrong on some frames due to a small training dataset). The model can be found in the **open_field_model** folder.

I used this model to calculate the acceleration, the decceleration, the speed and the distance crossed of the mouse in the video. All the code of these calculation can be found in the **acceleration_calculation.ipynb** file. I also used the openCV librairy in the **mouse_size.py** script in order to create a correspondance between the length in pixels on the images and the real centimeters length.

I saved the created datas in .csv files in the **result_acceleration** folder.

### An area specific acceleration / decceleration model
I tried to do a more complex model with DeepLabCut that can recognize the open field border and center but the lack of visual indication of their position in the video make this model impossible to create with DeepLabCut. This model would have been useful to calculate the acceleration and decceleration of the mouse in specific area of the open field.

## Fine tuning a ViT model

### An rearing / grooming classification fine tuned ViT model
In order to create more useful information from thoses videos, I fine_tuned a Vision Transformer (ViT) model. This model is very powerful to classify some images in category. I used it to classify the images of mouse based on the action the mouse does. The actions classified are the rearing, the grooming and normal (the mouse does nothing in particular). The code used to fine tuned the ViT model can be found in the **fine_tuning_ViT/fine_tuning_ViT.ipynb** file.

Therefore I needed to have a dataset of those actions. I extracted the frames of the videos I had with the DeepLabCut software and sort them in the 3 categories. Thus I created a 718 images dataset with 551 normal images, 110 rearing images and 57 grooming images.

Then, I fine tuned the model to recognize the actions and I trained it on my custom dataset.

The model doesn't have great result on the different images and that was predictible because the size of the dataset which is very small so the model can't train sufficiently on the differents categories.

In order to make this model usable (with a lower loss) it is mandatory to enlarge the dataset.

This model as been adapted with some openCV code in order to analyse an entire videos and get the time of rearing and grooming in the videos.


# Organization of the files

- fine_tuning_ViT : the ViT fine tuned model used to mesure the time of rearing and grooming
    * dataset : contains the datasets for rearing/grooming detection 
    * model : contains the model
    * training : contains the training infos
    * videos : the videos used for analysis
    * **fine_tuning_ViT.ipynb** : the script for the ViT fine tuned model
- images_for_openCV : the images used as references for the mouse_size.py script
- open_field_model : the DeepLabCut generated folder containing all the infos about the dlc model created in the model.ipynb file
- result_acceleration : contains all the output .csv of the acceleration_calculation.ipynb file
- videos : contains all the videos used for the tests
- **acceleration_calculation_ipynb** : the script that contains the calculation of the acceleration using the dlc model from the model.ipynb file
- **model.ipynb** : the file containing all the code to create and train the dlc model in the open_field_model folder
- **mouse_size.py** : the openCV script that calculate the ratio between image pixels and real centimeters (based on a mouse length)

