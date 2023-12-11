# Estonian Fingerspelling Recognition Model and Dataset
This project was created for the "Introduction to Data Science" course at University of Tartu and aims to train a model to recognize Estonian fingerspelling signs. 
<figure>    
    <img src="./Estonian_Fingerspelling-poster-final-1.png"
        alt="Poster"
        width="69%">
    <figcaption>Poster created for the presentation.</figcaption>
</figure>


## Dataset
The `data` folder contains a dataset of Estonian sign language fingerspelling signs for the 32 letters in the Estonian alphabet.
Each label is accompanied by over 200 images from 8 different individuals (4 men and 4 women in an age range of 18-21). 

## Scripts
The `scripts` folder contains Python scripts that we used for renaming the dataset, cropping the images and performing cursori PCA on the dataset.

## Training the model
A recognition model can be trained by running the Jupyter notebook in Colab or locally. When running locally, there may be some problems with Mediapipe when not using Linux.
Running the notebook takes about 30 minutes.
The Colab notebook is [here](https://colab.research.google.com/drive/1RgLQycIGeySCx58SFGnGl-DNT7RI3tMR?usp=sharing).

## References
The model training code is largely based on the [Hand Recognition Customization Guide](https://developers.google.com/mediapipe/solutions/customization/gesture_recognizer) by Mediapipe.
