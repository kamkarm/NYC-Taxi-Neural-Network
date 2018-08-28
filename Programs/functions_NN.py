import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.data import Dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pd.options.mode.chained_assignment = None



def process_df(df):
  """Prepares dataframe from NYC Taxi data set

  Args:
  NYC taxi dataframe: A Pandas df expected to contain data from the cleaned NYC taxi data set.

      Returns:
     A DataFrame that contains the features to be used for the model, including synthetic features.
  """

  #Add synthetic features

  #Create new synthetic features from pickup datetime
  pickup_datetime  = pd.to_datetime(df["tpep_pickup_datetime"])
  day_of_week =  pd.DatetimeIndex(pickup_datetime).weekday      
  day_of_month =  pd.DatetimeIndex(pickup_datetime).day         
  df["pickup_time"]  = pickup_datetime.dt.hour + pickup_datetime.dt.minute/60
  df['day_of_week'] = day_of_week   
  df['weekday'] = df['day_of_week'].apply(lambda x:(1 if x < 5 else 0))
  df['day_of_month'] = day_of_month   
  df['day_of_month'] = df['day_of_month'].apply(lambda x:(x-1)) 

  #Drop datetime feature after processing
  df = df.drop('tpep_pickup_datetime',1)

  #Create new feature counting amount of rides for a given time and day as a measure of traffic
  ride_counts = df.groupby(['day_of_week','pickup_time']).size()
  ride_counts = pd.DataFrame(ride_counts).reset_index()
  ride_counts['ride_counts'] = ride_counts[0]
  ride_counts = ride_counts.drop(0,1)
  df =  df.merge(ride_counts, on=['day_of_week',
                          'pickup_time'], how='left')

  #Process features for algorithm

  #Convert PULocationID and DOLocationID for bucketizing
  df['PULocationID'] = df['PULocationID'].apply(lambda x:(x-1)) 
  df['DOLocationID'] = df['DOLocationID'].apply(lambda x:(x-1)) 
  df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map(lambda x: 1 if x == 'Y' else 0)
  


  return df


def construct_feature_columns():
  """Construct the TensorFlow Feature Columns.

  
  Returns:
    A set of feature columns
  """ 
  
  #Set these features as continuous values
  pickup_time = tf.feature_column.numeric_column("pickup_time")
  trip_distance = tf.feature_column.numeric_column("trip_distance")
  ride_counts = tf.feature_column.numeric_column("ride_counts")


 #Categorizes features for each unique value in each feature, then converts them to one-hot encoding (necessary for NNs)
  weekday = tf.feature_column.categorical_column_with_identity(
    key='weekday',
    num_buckets=2)
  weekday = tf.feature_column.indicator_column(weekday)

  store_and_fwd_flag = tf.feature_column.categorical_column_with_identity(
    key='store_and_fwd_flag',
    num_buckets=2)
  store_and_fwd_flag = tf.feature_column.indicator_column(store_and_fwd_flag)

  day_of_week = tf.feature_column.categorical_column_with_identity(
    key='day_of_week',
    num_buckets=7)
  day_of_week = tf.feature_column.indicator_column(day_of_week)

  day_of_month = tf.feature_column.categorical_column_with_identity(
    key='day_of_month',
    num_buckets=28)
  day_of_month = tf.feature_column.indicator_column(day_of_month)

  PULocationID = tf.feature_column.categorical_column_with_identity(
    key='PULocationID',
    num_buckets=263)
  PULocationID =  tf.feature_column.indicator_column(PULocationID)

  DOLocationID = tf.feature_column.categorical_column_with_identity(
    key='DOLocationID',
    num_buckets=263)
  DOLocationID = tf.feature_column.indicator_column(DOLocationID)
 

  feature_columns = [
        pickup_time,
        day_of_week,
        day_of_month,
        weekday,
        trip_distance,
        PULocationID,
        DOLocationID,
        ride_counts,
        store_and_fwd_flag,
        ]
  
  return feature_columns


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a regression model.
  
    Args:
      features: pandas df of features
      targets: pandas df of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                            
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels



def train_nn_regression_model(
    my_optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    ):
  """Trains a neural network regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
    training_examples: A `df` containing one or more columns from
      to use as input features for training.
    training_targets: A `df` containing exactly one column from
       to use as target for training.
    validation_examples: A `df` containing one or more columns from
      to use as input features for validation.
    validation_targets: A `df` containing exactly one column from
       to use as target for validation.
    
      
  Returns:
      estimator: the trained `DNNRegressor` object.
      
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a DNNRegressor object.
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets, 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets, 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets, 
                                                    num_epochs=1, 
                                                    shuffle=False)
  

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))

    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)

  print("Model training finished.")
  
  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()
  plt.show()

  
  
  print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)
 

  return dnn_regressor

