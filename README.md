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


## Training


## Results



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
