# Char-Level-Sequence-Generator-Keras-
In this project a Keras model is used for generate sequences using a char-level approach. The code uses Tensorflow 1.13 as backed and Keras 2.2.4, because all previous public repositories I found where deprecated.

The model was trained and tested first with a dinosaur name dataset. This was a DeepLearning.ai specialization project I took, but in that moment I did it from scratch using numpy. So, I decided to develop a general model with Keras which allows to generated sequences char-by-char using one-hot encoding representation of chars, in this case dinosaurs names.

## Dependencies
* [numpy](https://pypi.python.org/pypi/numpy)
* [tensorflow](https://tensorflow.org)
* [keras](https://keras.io/)

## Training on Dino Names Dataset
The script dino_names_train.py allows to train the model on the dino dataset. The dataset wasn't uploaded to let the repo clean, in case you needed I will send you. To run the training type into the command line:
```bash
python dino_names_train.py dataset_path output_json_configuration_path output_weights_h5file_path \
Tx n_a epochs

```
The parameter Tx will be removed soon, by now it defines the max lenght of the sequences and all names are padded to size Tx. For more information about the command line arguments, type:
```bash
python dino_names_train.py -h

```

## Generating Names Sequences
For sampling new sequences, run the script generator.py as follow:
```bash
python names_generator.py json_configuration_file_path weights_h5file_path number_of_samples

```
Note that the json and h5 files requeried are the same we stored in the training step. For more information about the arguments run the -h command as mentioned above in the training section.

## Some examples of generated names
* injiangovenator
* unowasia
* roaesaurus
* nuanoddes
* aganocacer
* ndosaurus
* anatasaura
* ianghangosaurus
* aacorenator
* aoeyia
* oteuarlausaurn
* ingxiusaurus
* uoasaurus
* olindapteryx
* unchisaurus
* acraasaurus
* iniaosaurus


