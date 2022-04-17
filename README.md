# Lyric-Generation-Model
Final project CSC413: Neural networks and Deep Learning

## Introduction
The network we designed writes song lyrics. In essence, this means the
model performs text generation. From a given seed text (the first line of lyrics for
example) the model predicts the next character based on the conditional probabilities of
all characters in its learned vocabulary.
So if we feed it “Everybody put your hands u”, we hope it will return “p” so that we get
“Everybody put your hands up”.
This process is applied repeatedly, where we concatenate the seed and predicted
character serving as the new seed text for the next inference. This will generate more and
more characters sequentially until we have created a wholly new string. We can continue this
process arbitrarily and stop after a desired length is reached.

Basically, our target is to generate lyrics which require a relatively long input sequence in the character level. To prevent the gradient vanishing in simple RNN, we decided to use unidirectional LSTM to selectively choose the information we would like to keep in the long short-term memory. Lyrics generation is a subset of text generation. In the data preprocessing stage, we have assigned each unique character in the dataset a unique integer. The input of our model is a matrix, which length is the batch size and height is the number of input characters. Our model begins with a stacked LSTM. In the training phase, we use teaching force as well in order to speed up the convergence. The output of previous LSTM is fed into the next LSTM layer. After the stacked LSTM layer, we flatten out the output and then use a few fully connected layers to compress the output size. After the fully connected layers, we use softmax as our activation function for the output, then use cross entropy loss function with adam optimizer to calculate the loss.


## Model

## Data

&nbsp;&nbsp;&nbsp;&nbsp;The data for this model is acquired from the following Kaggle [link](https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres?select=lyrics-data.csv). This dataset has an [Open Database License](https://opendatacommons.org/licenses/dbcl/1-0/), which means we are free to use its content for educational purposes. To view a complete break-down of the dataset, including data acquisition, visualization, and modification, please see the notebook in this repository. 

&nbsp;&nbsp;&nbsp;&nbsp;To summarize, when we extract only English songs from the dataset, we get a total of 191,814 songs, with a mean word count of approximately 250, and a standard deviation of 159. If we exclude outliers in the dataset, namely songs with extremely low word counts (like less than 100) and songs with extremely high word counts (like over 400), we don’t actually lose that many songs, with a new total of 148,828 songs, but a lower standard deviation of 75, and now the distribution looks more normal. We also decided to simply exclude songs that had symbols or other languages as a part of the vocabulary. To do this, we had a list of acceptable characters, and then constrained our dataset to have only songs that adhere to our character list. Therefore, our final set of data was 117,708 songs, with a mean of 215 and standard deviation of 74, which is still enough data points for our model. After all these constraints on the dataset, we get an average character length of roughly 1079, and a standard deviation of 371. This means that the model should have enough variance in the data to train from, so that it hopefully generalizes well, and it will train from songs with word counts that you would typically expect.

&nbsp;&nbsp;&nbsp;&nbsp;We did not apply any data augmentation techniques, simply because we already had enough data as is, however we did have to use torchdata to load our datasets into useable training, validation, and testing sets. This means that our data was split into training, validation, and testing sets with 58,855, 39,237, and 19,619 data points respectively (split 50%, 30%, 20%). The process of splitting the dataset was not straightforward, as we ended up needing to create separate csv files for each set, and then load them into our notebook from each file. 


## Training

![image](https://user-images.githubusercontent.com/68927580/163728451-40702e80-8ecb-4c42-8809-e3a9ae7c7acf.png)

We tested a range of learning rates and different values for the embedding size and model size. We opted to stick with these parameters as the results we deemed were of sufficient quality.

## Results
The final loss is 1.284720 with training and validation accuracies 0.584776 and 0.584184 respectively.

Here's a quote of generated lyrics:

map reactions, i don't warning stand
every child and rain
they say i can get down the dark
don't tell me come in time you

In this example lyrics we can identify a few notable features:
- Real words with few spelling mistakes
- Somewhat coherent grammar
- Line breaks that make sense
- Does not often make actual sense

For our final model we will often see generated text that exhibit these features. These features are not present in an untrained model (that generates random strings). We can reasonably assume our model learned a large number of words and has some intuition of how to structure some lyrics. But it is unable to create lyrics that have substantial meaning.
We think that using a bigger model with more training time would allow us to generate lyrics that make more sense as opposed to just grammatically correct strings of words. We're confident our dataset was large enough, so we think the main bottleneck was the size of our model and the computing resources at our disposal.

## Ethical Considerations

Our lyric generation model has several use cases. For starters, it could be used for
inspiration for up and coming song-writers. It could also be used for people that love music
and are looking for a fun and entertaining way to generate songs. However, our model has
some limitations and could be misused. For instance, people could use it to generate offensive
(slurs, racist, discriminatory comments) lyrics if they were to manipulate the training data. If
someone were to license songs written with the help of our model they could face plagiarism
and copyrights issues if the lyrics were to be very similar to already existing songs. Finally,
since the model might not be 100% perfect, thus, our model might spit out ungrammatical or
meaningless lyrics occasionally.

## Authors
