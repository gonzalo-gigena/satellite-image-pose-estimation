import numpy as np
import tensorflow as tf

class SequenceDataGenerator(tf.keras.utils.Sequence):
  def __init__(self,images, numerical, targets, sequence_length=10, shuffle=False, 
                batch_size=32, augment=False):
    self.images = tf.convert_to_tensor(images, dtype=tf.float32)
    self.numerical = tf.convert_to_tensor(numerical, dtype=tf.float32)
    self.targets = tf.convert_to_tensor(targets, dtype=tf.float32)
    self.sequence_length = sequence_length
    self.batch_size = batch_size
    self.shuffle = shuffle
    
    # Ensure we only use indices where we have enough previous frames
    self.valid_indices = np.arange(sequence_length, len(self.targets))
    if self.shuffle:
      np.random.shuffle(self.valid_indices)
        
    self.augment = augment
    self.augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomBrightness(factor=0.2),
      tf.keras.layers.RandomContrast(factor=0.2),
      tf.keras.layers.GaussianNoise(0.1)
    ])

  def on_epoch_end(self):
    if self.shuffle:
        np.random.shuffle(self.valid_indices)

  def __len__(self):
    return int(np.ceil(len(self.valid_indices) / self.batch_size))

  def __getitem__(self, idx):
    # Get indices for this batch
    batch_indices = self.valid_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
    
    # Initialize batch arrays
    batch_images = np.zeros((len(batch_indices), self.sequence_length, 
                            self.images.shape[1], self.images.shape[2], 1))
    batch_numerical = np.zeros((len(batch_indices), self.sequence_length, 
                              self.numerical.shape[1]))
    batch_targets = np.zeros((len(batch_indices), 4))  # 4 for quaternion
    
    for i, idx in enumerate(batch_indices):
      # Get sequence indices
      sequence_indices = np.arange(idx - self.sequence_length, idx)
      
      # Convert tensors to numpy arrays for indexing
      images_np = self.images.numpy()
      numerical_np = self.numerical.numpy()
      targets_np = self.targets.numpy()
      
      # Fill in sequences
      batch_images[i] = images_np[sequence_indices]
      batch_numerical[i] = numerical_np[sequence_indices]
      batch_targets[i] = targets_np[idx]
        
    # Convert back to tensors
    batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
    batch_numerical = tf.convert_to_tensor(batch_numerical, dtype=tf.float32)
    batch_targets = tf.convert_to_tensor(batch_targets, dtype=tf.float32)

    if self.augment:
      batch_images = self.augmentation(batch_images, training=True)
        
    return {
        'image_sequence': batch_images,
        'numerical_sequence': batch_numerical
    }, batch_targets


class SequenceModel(tf.keras.Model):
  def __init__(self, image_height=102, image_width=102, sequence_length=10, numerical_features=4):
    super(SequenceModel, self).__init__()
    
    # Save parameters
    self.sequence_length = sequence_length
    self.image_height = image_height
    self.image_width = image_width
    self.numerical_features = numerical_features
    
    # CNN for processing individual frames
    self.image_encoder = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                            input_shape=(image_height, image_width, 1)),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.GlobalAveragePooling2D()
    ])
    
    # Process numerical features
    self.numerical_encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(32, activation='relu', 
                          input_shape=(numerical_features,)),
      tf.keras.layers.Dense(16, activation='relu')
    ])
    
    # LSTM layers for sequence processing
    self.sequence_processor = tf.keras.Sequential([
      tf.keras.layers.LSTM(256, return_sequences=True, 
                          input_shape=(sequence_length, 144)),  # 128 (CNN) + 16 (numerical)
      tf.keras.layers.LSTM(128)
    ])
    
    # Final prediction layers
    self.quaternion_predictor = tf.keras.Sequential([
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(4)  # 4 quaternion components
    ])
  
  """
  def build(self, input_shape):
      # Create dummy inputs to build the model
      dummy_images = tf.zeros((1, self.sequence_length, self.image_height, self.image_width, 1))
      dummy_numerical = tf.zeros((1, self.sequence_length, self.numerical_features))
      
      self.call({
          'image_sequence': dummy_images,
          'numerical_sequence': dummy_numerical
      })
      
      super().build(input_shape)
  """
  def call(self, inputs):
    # Unpack inputs and verify shapes
    image_sequence = inputs['image_sequence']  # Shape: (batch_size, seq_len, height, width, 1)
    numerical_sequence = inputs['numerical_sequence']  # Shape: (batch_size, seq_len, numerical_features)
    
    # Verify sequence length
    tf.debugging.assert_equal(tf.shape(image_sequence)[1], self.sequence_length,
                            message="Image sequence length doesn't match model's sequence_length")
    tf.debugging.assert_equal(tf.shape(numerical_sequence)[1], self.sequence_length,
                            message="Numerical sequence length doesn't match model's sequence_length")
    
    batch_size = tf.shape(image_sequence)[0]
    
    # Process each image in the sequence through CNN
    image_sequence_reshaped = tf.reshape(image_sequence, 
                                        [-1, self.image_height, self.image_width, 1])
    image_features = self.image_encoder(image_sequence_reshaped)
    image_features = tf.reshape(image_features, 
                              [batch_size, self.sequence_length, -1])
    
    # Process numerical features
    numerical_features = tf.reshape(numerical_sequence, 
                                  [-1, self.numerical_features])
    numerical_features = self.numerical_encoder(numerical_features)
    numerical_features = tf.reshape(numerical_features, 
                                  [batch_size, self.sequence_length, -1])
    
    # Combine features
    combined_features = tf.concat([image_features, numerical_features], axis=-1)
    
    # Process sequence
    sequence_features = self.sequence_processor(combined_features)
    
    # Predict quaternion
    quaternions = self.quaternion_predictor(sequence_features)
    
    # Normalize quaternion
    quaternions_normalized = tf.math.l2_normalize(quaternions, axis=-1)
    
    return quaternions_normalized