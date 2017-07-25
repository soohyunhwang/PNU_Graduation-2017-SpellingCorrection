# 절대 임포트 설정
from __future__ import absolute_import
from __future__ import print_function

# 필요한 라이브러리들을 임포트
import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Word Embedding을 수행할 Training Set을 로드한다.
filename = 'train_data/morp_train_set.zip'

# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()    # zip 파일 내부의 data 파일은 UTF-8로 인코딩되어 있어야 함
  return data

words = read_data(filename)
print('Data size', len(words))

# Step 2: 로드한 데이터로 사전을 구성하고, 희소한 단어들은 UNK로 저장한다.
# UNK는 unknown의 약자. 출현빈도 낮은 단어를 의미한다.
vocabulary_size = 50000    # 사전 크기는 데이터에 맞춰서 조정해야 함. UNK token의 Count가 너무 크지 않도록 적절하게 맞추는 것이 좋음.

def build_dataset(words):
  count = [['UNK', -1]] #-1: 초기화
  # collections.Counter(words) -> words내에서 중복값 제거
  # .most_common(vocabulary_size - 1) -> 단어에 대한 출현 빈도를 vocabulary_size -1 개 단어에 대해 계산.
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  # 즉, 중복을 제거한 후 words 내에서 출현 빈도가 높은 단어를 vocabulary_size -1 개 추출.

  dictionary = dict() # dictionary 객체 생성

  # count 는 [char, int] 쌍을 원소로 가지는 list형태 이므로
  # 이 쌍에서 char타입을 word라는 변수로 읽고, int는 _로 읽는데 아마도 그냥 읽기위한 임의형태의 변수(?)인듯하다.
  for word, _ in count:
    dictionary[word] = len(dictionary)
  # 이 for문이 다 돌고나면 dictionary는 (UNK, 0) (word1, 1) (word2, 2) (word3, 3) (word4, 4) ...이런 형태를 갖는다.
  # int 값으로는 len(dictionary)가 들어가는데 이는 dictionary에 word가 들어갈 때마다 1씩 증가한다.
  # 어떤 training data가 들어가느냐에 따라서 (word1, 1) (word2, 2) (word3, 3) (word4, 4) ...는 달라지지만
  #  (UNK, 0)가 dictionary의 첫번째 원소임은 변하지 않는다!!

  data = list() # list 객체 생성
  unk_count = 0 # 아마도 출현빈도가 낮은 단어의 갯수를 세는 변수... (?)

  # words 안에 있는 값을 하나씩 뽑아내서 이 값이 dictionary에 존재하면, index값을 얻는다.
  # 이 값이 존재하지  않으면, index = 0 이 되어 UNK단어로 분류한다. dictionary에 없으면 출현빈도가 낮은것으로 보기 때문에! 그리고, UNK단어의 갯수 unk_count를 증가시킨다.
  # 그리고 쓰인 index는 data라는 list에 추가한다.

  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count #처음엔 ['UNK',-1]로 설정된 값에서  -1(int)값을 하나 증가시킨 unk_count로 갱신한다.

  # 만들어진 dictionary의 값들을 zip 해서 다시 dictionary를 만든다는건데 zip이란게... http://www.dreamy.pe.kr/zbxe/CodeClip/165019 여기 보면될듯함..
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary #data:단어에 대한 index list, count:단어에 대한 빈도수 dictionary:빈도순대로 정렬된 단어들

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Skip Gram Model을 위한 Batch를 생성하기 위한 함수.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window #2보다 작으면 좌우로 1개씩 단어들을 모두 다 뽑지는 않는다는 의미.(랜덤으로 일부만비교)
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ] -> 입력들어가는 배치의 형태
  buffer = collections.deque(maxlen=span) # 사이즈 안변함
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)

  for i in range(batch_size // num_skips):#몇개의 target을 할 것인가
    target = skip_window  # target label at the center of the buffer: 학습하고자하는 target의 좌우와 비교 , target은 무조건 중간
    #skip_window가 2라면 양쪽에 2개씩 있어야하니까 target이 되는 배열의 인덱스가 2가 되어야 함
    targets_to_avoid = [skip_window]#나왔던 단어가 다시 나오면 안되니까 계속 추가해서 피하려고
    for j in range(num_skips):#numskip이 1이면 target과 비교해야할 페어들 중 1개만 뽑게된다는것 무조건다 뽑을수도 있고 좌우 문맥중 하나만 뽑아서 비교할 수도 있다는 소리
      #다 비교하지않아고 랜덤으로 일부만 써서 속도를 올릴 수도 있음. 이게 더 효율이 좋을 수도 있음
      #1 target안에서 좌우 문맥 단어를 몇번 뽑을 것인가
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)#이미 나온거는 안됨. target과 target은 페어가 되면 안됨.
      targets_to_avoid.append(target)#다음에 뽑은 새로운 target을 다시 뽑으면 안되니까 추가
      batch[i * num_skips + j] = buffer[skip_window]#타켓 단어를 (인덱스는 : 행렬로 생각하면 쉬움)
      labels[i * num_skips + j, 0] = buffer[target]#타켓 옆에 다시 뽑은 단어. j만큼 도니까 numskips 갯수만큼
    buffer.append(data[data_index])#사이즈 안변하니까 append하면 하나씩 앞으로 밀릴것 제일 앞의 것 사라짐.
    data_index = (data_index + 1) % len(data)
  return batch, labels

# Step 4: Skip Gram 모델을 구성하고 학습한다.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2           # How many times to reuse an input to generate a label. (context, target)쌍의 갯수
#skip window를 통해 결정된 좌우 비교할 단어들 중 모두를 다 뽑아서 target과 비교할 수도 있고 일부만 뽑아서 비교할 수도 있기 때문에 skipwindow와 따로있는것
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: 트레이닝을 시작한다.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # 트레이닝을 시작하기 전에 모든 변수들을 초기화한다.
  tf.initialize_all_variables().run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # optimizer op을 평가(evaluating)하면서 한 스텝 업데이트를 진행한다.
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # 평균 손실(average loss)은 지난 2000 배치의 손실(loss)로부터 측정된다.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # nearest neighbors의 개수
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()   #nomalized_embeddings를 rnn의 input으로 넣는다


# Step 6: embeddings을 시각화한다.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")