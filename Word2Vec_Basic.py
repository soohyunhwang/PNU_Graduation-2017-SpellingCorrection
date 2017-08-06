from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
import zipfile
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
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
vocabulary_size = 430000    # 사전 크기는 데이터에 맞춰서 조정해야 함. UNK token의 Count가 너무 크지 않도록 적절하게 맞추는 것이 좋음.

def build_dataset(words):
  count = [['UNK', -1]] #unknown count의 갯수 -1은 초기화때문에
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1)) # words내에서 출현빈도가 높은 순대로 vacabulary_size-1개 추출
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary) #(UNK,0) (word1,1) (word2,2) (word3,3)...
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else: #dictionary에 없으면 출현빈도 낮음
      index = 0  # dictionary['UNK']
      unk_count += 1 #UNK 단어갯수 증가
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) #(0,UNK) (1,word1) (2,word2) 형태로
  return data, count, dictionary, reverse_dictionary
  #data:랜덤으로 뽑은 단어에 대한 index list, count:단어에 대한 빈도수 dictionary:빈도순대로 정렬된 단어들

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# Step 3: Skip Gram Model을 위한 Batch를 생성하기 위한 함수.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0 #num_skips를 여러번 하는게 batch_size
  assert num_skips <= 2 * skip_window #2보다 작으면 좌우로 1개씩 단어들을 모두 다 뽑지는 않는다는 의미.(랜덤으로 일부만비교)
  batch = np.ndarray(shape=(batch_size), dtype=np.int32) #1차원
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32) #2차원
  span = 2 * skip_window + 1  # [ skip_window target skip_window ] -> 입력들어가는 배치의 형태
  buffer = collections.deque(maxlen=span) # 사이즈 안변함 deque이므로 append하면 앞의 원소가 자동으로 삭제된다.

  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data) #다음 단어 인덱스로 이동. 가진 data 인덱스 길이까지만.

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
      batch[i * num_skips + j] = buffer[skip_window]#타켓 단어 (인덱스는 : 행렬로 생각하면 쉬움) num_skips만큼 target단어 반복
      labels[i * num_skips + j, 0] = buffer[target]#타켓 옆에 다시 뽑은 단어. j만큼 도니까 numskips 갯수만큼
    buffer.append(data[data_index])#사이즈 안변하니까 append하면 하나씩 앞으로 밀릴것 제일 앞의 것 사라짐.
    data_index = (data_index + 1) % len(data)
  return batch, labels

# Step 4: Skip Gram 모델을 구성하고 학습한다.

batch_size = 128      #batch 배열의 크기 일반적으로 16 <=batch_size<=512
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right. target좌우로 몇개의 단어와 비교하여 볼 것인가
num_skips = 2         # How many times to reuse an input to generate a label. (context, target)쌍의 갯수
#skip window를 통해 결정된 좌우 비교할 단어들 중 모두를 다 뽑아서 target과 비교할 수도 있고 일부만 뽑아서 비교할 수도 있기 때문에 skipwindow와 따로있는것

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on. 유사성을 평가할 단어 집합 크기
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False) #앞쪽의 100개의 데이터 중 16개를 랜덤선택
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size]) #정수로 선언 target들
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) #정수로 선언 context들
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32) #16개의 단어들로 만들어진 dataset.

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs) #원본 단어들의 vector확인

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))    # (430000,128) 어휘에 대한 가중치
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))   #(430000) 편향

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
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss) #loss가 감소하는 방향으로 움직이기 위해 기울기 구하여 움직임

  #값이 너무 커지게 되면 경사하강법으로 gradientdescent로 접근하였을 때 오차가 커진다 그러므로 정규화를 통해 데이터의 크기를 낮춘다
  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)) #정규분포 표준편차*루트n 값
  normalized_embeddings = embeddings / norm #임베딩 정규화 -> 표준편차로 나눠주면 정규화된 임베딩
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset) #단어에 대한 임베딩
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True) #cosine similarity 구하기위해 matrix 곱셈

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Traning 시작
num_steps = 200001 #마지막 반복을 출력하기 위해서 +1

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for step in range(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval() #16개의 단어와 vacabulary의 모든 단어까지의 거리 계산 : 샘플하나에 430000번계산하는 것  """Evaluate analogy questions and reports accuracy."""
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]] #dictionary에 있는 첫번째 단어부터 넣음
        top_k = 8  # number of nearest neighbors #가까운 이웃 8개 선택할 것
        nearest = (-sim[i, :]).argsort()[1:top_k + 1] #argsort는 배열을 정렬시키는 index 순서 리스트를 반환, -sim은 정렬을 반대로.
        log_str = "Nearest to %s:" % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]] #뽑은 단어에 가까운 8개의 단어를 close_word에 넣음
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

  # 학습 결과를 저장한다.
  saver = tf.train.Saver()
  save_path = saver.save(session, './model/word2vec.ckpf')

  with open('./model/word2vec_dic', 'wb') as f:
    pickle.dump(dictionary, f)

  with open('./model/word2vec_revdic', 'wb') as f:
    pickle.dump(reverse_dictionary, f)

