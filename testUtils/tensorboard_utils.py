from datetime import datetime
import io
import itertools
from packaging import version

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import seaborn as sns
import einops

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
#   image = tf.expand_dims(image, 0)
#   image = tf.expand_dims(image, 0)
  image = einops.rearrange(image,'h w c-> 1 h w c' )  
  return image

def plot_heatmap_to_image(arr):
    fig = sns.heatmap(arr).get_figure()
    return plot_to_image(fig)



# plot_heatmap_to_image(np.random.random((20,20)))
# def image_grid():
#   """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
#   # Create a figure to contain the plot.
#   figure = plt.figure(figsize=(10,10))
#   for i in range(25):
#     # Start next subplot.
#     plt.subplot(5, 5, i + 1, title=class_names[train_labels[i]])
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)

#   return figure

# # Prepare the plot
# figure = image_grid()
# # Convert to image and log
# with file_writer.as_default():
#   tf.summary.image("Training data", plot_to_image(figure), step=0)