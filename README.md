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

![LSTM-Model](https://user-images.githubusercontent.com/55116264/163865319-d3eedbc3-bfa8-4bd5-bf8c-edc991a30557.jpeg)



Our model consists of embedding layers, LSTM layer and a fully connected layer. The embedding layer featurize a vector representation of each character given the number of vocabulary in the dataset. The featurized vectore is fed into the LSTM layer as input. Our model only uses one LSTM layer and we think one layer is sufficient for our lyrics generator. We have also used bi-directional and stacked LSTM. Both of these model overfits the data easily. So we choosed the single LSTM model for better generalization. The fully connected layer converts LSTM output (hidden state at each time stamp) to desired shape (number of vocabulary).

#### Number of parameters

Let V be the number of vocabularies in the dataset, E be the number of embedding units ,H be the number of hidden units in LSTM, T be the number of Time stamps and B be the batch size.

In the embedding layer, we need to train the weights in the dense layer. The input shape of the dense layer is (B, S, 1, V), one hot encoded character representation. The output shape of this layer is (B, S, 1, E). Shape of the trainable weight is (B, S, V, E). 

![LSTM-Function](https://user-images.githubusercontent.com/55116264/163867543-032cebc3-77f4-4e24-b46e-189e302d604f.jpg)

The forward pass for LSTM is shown above. We can see that for each of the gate and memory, there are two weights and two biases that our model need to train. The first pair of weight and bias respect to the current input sequence. Therefore, the shape of weight is (B, H, E), given the input shape is (B, 1, E) for each time stamp, and the shape of bias is (B, 1, H). The second pair of weight and bias respect to previous hidden state, so the shape of weight is (B, H, H) and shape of bias is (B, 1, H).

At each time stamp, our model needs to train the above paramters for each gate and memory cell (4 times).

At last, we feed the output of each time stamp to the dense layer to produce output with shape (B, S, 1, V). So that the shape of weight in this layer is (B, S, H, V).

#### Examples

The lyrics generator takes in a trained model and a pair of char-to-int and int-to-char mapping, maximum length of character generated and temperature (degree of divergence).

This function call the forward method of the model at most maxmum length times or the output for forward call is <EOS> (end of string) and return a predicted string.

#### Successful example

![Success](https://user-images.githubusercontent.com/55116264/163878525-ecf6b6c8-eab3-49d7-a324-f1f80f51b0fe.jpg)

#### Failure example 

![Fail](https://user-images.githubusercontent.com/55116264/163879318-414709b3-6eed-41f8-9b90-62ab16d2081a.jpg)


## Data

&nbsp;&nbsp;&nbsp;&nbsp;The data for this model is acquired from the following Kaggle [link](https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres?select=lyrics-data.csv). This dataset has an [Open Database License](https://opendatacommons.org/licenses/dbcl/1-0/), which means we are free to use its content for educational purposes. To view a complete break-down of the dataset, including data acquisition, visualization, and modification, please see the notebook in this repository. 

&nbsp;&nbsp;&nbsp;&nbsp;To summarize, when we extract only English songs from the dataset, we get a total of 191,814 songs, with a mean word count of approximately 250, and a standard deviation of 159. If we exclude outliers in the dataset, namely songs with extremely low word counts (like less than 100) and songs with extremely high word counts (like over 400), we don’t actually lose that many songs, with a new total of 148,828 songs, but a lower standard deviation of 75, and now the distribution looks more normal. We also decided to simply exclude songs that had symbols or other languages as a part of the vocabulary. To do this, we had a list of acceptable characters, and then constrained our dataset to have only songs that adhere to our character list. Therefore, our final set of data was 117,708 songs, with a mean of 215 and standard deviation of 74, which is still enough data points for our model. After all these constraints on the dataset, we get an average character length of roughly 1079, and a standard deviation of 371. This means that the model should have enough variance in the data to train from, so that it hopefully generalizes well, and it will train from songs with word counts that you would typically expect.

&nbsp;&nbsp;&nbsp;&nbsp;We did not apply any data augmentation techniques, simply because we already had enough data as is, however we did have to use torchdata to load our datasets into useable training, validation, and testing sets. This means that our data was split into training, validation, and testing sets with 58,855, 39,237, and 19,619 data points respectively (split 50%, 30%, 20%). The process of splitting the dataset was not straightforward, as we ended up needing to create separate csv files for each set, and then load them into our notebook from each file. 


## Training

![image](https://user-images.githubusercontent.com/68927580/163728451-40702e80-8ecb-4c42-8809-e3a9ae7c7acf.png)

We tried a couple of different embedding size and hidden size before choosing 128 and 256 respectively.  Moreover, we tried a variety of different learning rates and adjusted whenever we deemed necessary (i.e. if the model was converging slowly we would increase the learning rate and if we were getting too much instability we would decrease it). We ended up by settling on a learning rate of 0.003 for the main model. We also kept a log of hyperparameters tested so that we don’t repeat ourselves and waste time since the models took a  while to train.

## Results
The final loss is 1.284720 with training and validation accuracies 0.584776 and 0.584184 respectively.

Here's a quote of generated lyrics:

map reactions, i don't warning stand\
every child and rain\
they say i can get down the dark\
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

We each contributed to this final project, but please see the breakdown of tasks below for more information:

Vishay: Worked on data acquisition/visualization and summary statistics, model creation, overfitting models, training batch models/checkpointing, learning curve generation, picking the best model, song generation, and general notebook organization/writing. As for the README, I worked on Data, and Authors.

Justin: Worked on model architecture and coding. Worked on training and overfitting small dataset. Iterated on different model variations to create a higher performing model. Also, bucketing the data, and tuning parameters. For the README, I worked on Results.

Thomas: Worked on data formatting/acquisition, training and choosing models, tuning hyperparameters. For the README, I was responsible for the Introduction, Training, and Ethical considerations sections.  

Litao: Worked on cleaning dataset, splitting dataset into train, test and validation, overfitting small data, batch model initialization, training model and helper functions for calculating accuracies and plotting graphs. For the README, I am responsible for the model section.

