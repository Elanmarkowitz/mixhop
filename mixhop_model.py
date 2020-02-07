import copy
import json

import tensorflow as tf

def sparse_dropout(x, drop_prob, num_entries, is_training):
  """Dropout for sparse tensors."""
  keep_prob = 1.0 - drop_prob
  is_test_float = 1.0 - tf.cast(is_training, tf.float32)
  random_tensor = is_test_float + keep_prob
  random_tensor += tf.random_uniform([num_entries])
  dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
  pre_out = tf.sparse_retain(x, dropout_mask)
  return pre_out * (1./tf.maximum(is_test_float, keep_prob))


def psum_output_layer(x, num_classes):
  import pdb; pdb.set_trace()
  num_segments = int(x.shape[1]) // num_classes
  if int(x.shape[1]) % num_classes != 0:
    print('Wasted psum capacity: %i out of %i' % (
        int(x.shape[1]) % num_classes, int(x.shape[1])))
  sum_q_weights = tf.get_variable(
      'psum_q', shape=[num_segments], initializer=tf.zeros_initializer, dtype=tf.float32, trainable=True)
  tf.losses.add_loss(tf.reduce_mean((sum_q_weights ** 2)) * 1e-3 )
  softmax_q = tf.nn.softmax(sum_q_weights)  # softmax
  psum = 0
  for i in range(int(num_segments)):
    segment = x[:, i*num_classes : (i+1)*num_classes]
    psum = segment * softmax_q[i] + psum
  return psum


def random_walk_layer(B, adj_list, degree_matrix, num_steps, num_sims):
  """Computes omega using random walks
  
  Omega has shape (batch size, k, num sims)
  """
  P = tf.tile(B, (num_sims,))
  Ps = [P]
  for i in range(num_steps):
    x = tf.random.uniform((num_sims,))
    x = tf.repeat(x, B.shape[0])
    degrees = tf.gather(degree_matrix, P)
    neighbor_indices = tf.cast(tf.floor(x*tf.cast(degrees, 'float32')), 'int32')
    adj_indices = tf.transpose(tf.stack([P, neighbor_indices]))
    P_ = tf.gather_nd(adj_list, adj_indices)
    Ps.append(P_)
    P = P_

  return tf.reshape(tf.transpose(tf.stack(Ps)), (B.shape[0], num_steps + 1, num_sims))


def random_walk_adj_times_x(x, omega, adj, degree_matrix, adj_pow=1):
  """Computes (adj^adj_pow)*x using random walk matrix, omega"""
  omega_k = omega[:, adj_pow, :]
  omega_start = omega[:, 0, :]
  x_omega = tf.gather(x, omega_k)
  deg_v = tf.gather(degree_matrix, omega_k)
  deg_u = tf.gather(degree_matrix, omega_start)
  d = tf.cast((deg_u/deg_v)**0.5, 'float32')
  return tf.reduce_mean(tf.expand_dims(d, -1) * x_omega, 1)


def adj_times_x(adj, x, adj_pow=1):
  """Multiplies (adj^adj_pow)*x."""
  for i in range(adj_pow):
    x = tf.sparse_tensor_dense_matmul(adj, x)
  return x

def mixhop_layer(x, sparse_adjacency, adjacency_powers, dim_per_power,
                 kernel_regularizer=None, layer_id=None, replica=None,
                 random_walk_omega=None, degree_matrix=None):
  """Constructs MixHop layer.

  Args:
    sparse_adjacency: Sparse tensor containing square and normalized adjacency
      matrix.
    adjacency_powers: list of integers containing powers of adjacency matrix.
    dim_per_power: List same size as `adjacency_powers`. Each power will emit
      the corresponding dimensions.
    layer_id: If given, will be used to name the layer
  """
  #
  replica = replica or 0
  layer_id = layer_id or 0
  segments = []

  for p, dim in zip(adjacency_powers, dim_per_power):
    if random_walk_omega is None:
      net_p = adj_times_x(sparse_adjacency, x, p)
    else: 
      net_p = random_walk_adj_times_x(x, random_walk_omega, sparse_adjacency, degree_matrix, p)

    with tf.variable_scope('r%i_l%i_p%s' % (replica, layer_id, str(p))):
      layer = tf.layers.Dense(
          dim,
          kernel_regularizer=kernel_regularizer,
          activation=None, use_bias=False)
      net_p = layer.apply(net_p)

    segments.append(net_p)
  return tf.concat(segments, axis=1)


MODULE_REFS = {
    'tf': tf,
    'tf.layers': tf.layers,
    'tf.nn': tf.nn,
    'tf.sparse': tf.sparse,
    'tf.contrib.layers': tf.contrib.layers
}

class MixHopModel(object):
  """Builds MixHop architectures. Used as architectures can be learned.
  
  Use like:
    model = MixHopModel(sparse_adj, x, is_training, kernel_regularizer)
    ...
    model.add_layer('<module_name>', '<fn_name>', args_to_fn)
    model.add_layer( ... )
    ...

  Where <module_name> must be a string defined in MODULE_REFS, and <fn_name>
  must be a function living inside module indicated by <module_name>, finally,
  args_to_fn are passed as-is to the function (with name <fn_name>), with the
  exception of arguments:
    pass_kernel_regularizer: if argument is present, then we pass
      kernel_regularizer argument with value given to the constructor.
    pass_is_training: if argument is present, then we pass is_training argument
      with value given to the constructor.
    pass_training: if argument is present, then we pass training argument with
      value of is_training given to the constructor.
  
  In addition <module_name> can be:
    'self': invokes functions in this class.
    'mixhop_model': invokes functions in this file.

  See example_pubmed_model() for reference.
  """
  
  def __init__(self, sparse_adj, sparse_input, is_training, kernel_regularizer,
               adj_list=None, degrees=None):
    self.is_training = is_training
    self.kernel_regularizer = kernel_regularizer 
    self.sparse_adj = sparse_adj
    self.adj_list = adj_list
    self.degrees = degrees
    self.sparse_input = sparse_input
    self.layer_defs = []
    self.activations = [sparse_input]
    self.omega = None

  def save_architecture_to_file(self, filename):
    with open(filename, 'w') as fout:
      fout.write(json.dumps(self.layer_defs, indent=2))

  def load_architecture_from_file(self, filename):
    if self.layer_defs:
      raise ValueError('Model is (partially) initialized. Cannot load.')
    layer_defs = json.loads(open(filename).read())
    for layer_def in layer_defs:
      self.add_layer(layer_def['module'], layer_def['fn'], *layer_def['args'],
                     **layer_def['kwargs'])

  def add_layer(self, module_name, layer_fn_name, *args, **kwargs):
    #
    self.layer_defs.append({
        'module': module_name,
        'fn': layer_fn_name,
        'args': args,
        'kwargs': copy.deepcopy(kwargs),
    })
    #
    if 'pass_training' in kwargs:
      kwargs.pop('pass_training')
      kwargs['training'] = self.is_training
    if 'pass_is_training' in kwargs:
      kwargs.pop('pass_is_training')
      kwargs['is_training'] = self.is_training
    if 'pass_kernel_regularizer' in kwargs:
      kwargs.pop('pass_kernel_regularizer')
      kwargs['kernel_regularizer'] = self.kernel_regularizer
    #
    fn = None
    if module_name == 'mixhop_model':
      fn = globals()[layer_fn_name]
    elif module_name == 'self':
      fn = getattr(self, layer_fn_name)
    elif module_name in MODULE_REFS:
      fn = getattr(MODULE_REFS[module_name], layer_fn_name)
    else:
      raise ValueError(
          'Module name %s not registered in MODULE_REFS' % module_name)
    self.activations.append(
        fn(self.activations[-1], *args, **kwargs))

  def mixhop_layer(self, x, adjacency_powers, dim_per_power,
                   kernel_regularizer=None, layer_id=None, replica=None):
    return mixhop_layer(x, self.sparse_adj, adjacency_powers, dim_per_power,
                        kernel_regularizer, layer_id, replica, 
                        random_walk_omega=self.omega, degree_matrix=self.degrees)

  def use_random_walk_approx(self, B, num_steps=5, num_sims=300):
    if self.adj_list is None or self.degrees is None:
      raise Exception('Cannot do random walk approximation without adj_list and degrees.')
    self.omega = random_walk_layer(B, self.adj_list, self.degrees, num_steps, num_sims)

def example_pubmed_model(
    sparse_adj, x, num_x_entries, is_training, kernel_regularizer, input_dropout,
    layer_dropout, num_classes=3, random_walk_approx=False):
  """Returns PubMed model with test performance ~>80.4%.
  
  Args:
    sparse_adj: Sparse tensor of normalized adjacency matrix.
    x: Sparse tensor of feature matrix.
    num_x_entries: number of non-zero entries of x. Used for sparse dropout.
    is_training: boolean scalar Tensor.
    kernel_regularizer: Keras regularizer object.
    input_dropout: Float in range [0, 1.0). How much to drop out from input.
    layer_dropout: Dropout value for dense layers.
  """ 
  model = MixHopModel(sparse_adj, x, is_training, kernel_regularizer)

  model.add_layer('mixhop_model', 'sparse_dropout', input_dropout,
                  num_x_entries, pass_is_training=True)
  model.add_layer('tf', 'sparse_tensor_to_dense')
  model.add_layer('tf.nn', 'l2_normalize', axis=1)

  # MixHop Conv layer
  model.add_layer('self', 'mixhop_layer', [0, 1, 2], [17, 22, 21], layer_id=0,
                  pass_kernel_regularizer=True)

  model.add_layer('tf.contrib.layers', 'batch_norm')
  model.add_layer('tf.nn', 'tanh')

  model.add_layer('tf.layers', 'dropout', layer_dropout, pass_training=True)
  # MixHop Conv layer
  model.add_layer('self', 'mixhop_layer', [0, 1, 2], [3, 1, 6], layer_id=1,
                  pass_kernel_regularizer=True)
  model.add_layer('tf.layers', 'dropout', layer_dropout, pass_training=True)
  # MixHop Conv layer
  model.add_layer('self', 'mixhop_layer', [0, 1, 2], [2, 4, 4], layer_id=2,
                  pass_kernel_regularizer=True)
  model.add_layer('tf.contrib.layers', 'batch_norm')
  model.add_layer('tf.nn', 'tanh')
  model.add_layer('tf.layers', 'dropout', layer_dropout, pass_training=True)
  
  # Classification Layer
  model.add_layer('tf.layers', 'dense', num_classes, use_bias=False,
                  activation=None, pass_kernel_regularizer=True)
  return model
