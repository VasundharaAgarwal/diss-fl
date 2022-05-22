from cgi import test
from pathlib import Path

import tensorflow as tf
from flwr.common.typing import  Weights
import tensorflow_federated as tff

def get_model():
    data_format = 'channels_last'
    initializer = tf.keras.initializers.GlorotNormal()
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1),
          kernel_initializer=initializer),
      tf.keras.layers.Conv2D(
          64,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          kernel_initializer=initializer),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          128, activation='relu', kernel_initializer=initializer),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          62),
  ])
    # model = tf.keras.models.Sequential([
    #   tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
    #   tf.keras.layers.Dense(200, activation=tf.nn.relu),
    #   tf.keras.layers.Dense(200, activation=tf.nn.relu),
    #   tf.keras.layers.Dense(10)])
    model.compile(optimizer = tf.keras.optimizers.SGD(0.032), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def element_fn(element):
    return (tf.expand_dims(element['pixels'], -1), element['label'])

def get_emnist_dataset():
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=False)


  def preprocess_train_dataset(dataset):
    # Use buffer_size same as the maximum client dataset size,
    # 418 for Federated EMNIST
    return (dataset.map(element_fn)
                   .shuffle(buffer_size=418)
                   .repeat(1)
                   .batch(20, drop_remainder=False))

  def preprocess_test_dataset(dataset):
    return dataset.map(element_fn).batch(128, drop_remainder=False)

  emnist_train = emnist_train.preprocess(preprocess_train_dataset)
  emnist_test = preprocess_test_dataset(
      emnist_test.create_tf_dataset_from_all_clients())
  return emnist_train, emnist_test



def get_eval_fn(
) :
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: Weights) :
        """Use the entire test set for evaluation."""

        model = get_model()
        model.set_weights(weights)
        loss, accuracy = model.evaluate(x=test_data)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
def get_num_total_clients():
    return len(train_data.client_ids)

def load_testset():
    return test_data

def load_trainset(cid):
    return train_data.create_tf_dataset_for_client(train_data.client_ids[int(cid)])
    
train_data, test_data = get_emnist_dataset()


  
