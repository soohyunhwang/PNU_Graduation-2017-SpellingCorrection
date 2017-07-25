from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import tensorflow as tf

vocabulary_size = 430000
embedding_size = 128  # Dimension of the embedding vector.

window_size = 2

with open('./model/word2vec_dic', 'rb') as f:
  dictionary = pickle.load(f)

with open('./test_data/기본_태깅_오류.txt', 'r') as f:
  input_text_list = f.readlines()

embeddings = tf.Variable(
      tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm

init = tf.global_variables_initializer()

with tf.Session() as session:
  init.run()

  saver = tf.train.Saver()
  saver.restore(session, './model/word2vec.ckpf')
  embedding_vec = normalized_embeddings.eval()

  detect_count = 0
  total_count = 0
  for i in range(0, len(input_text_list)):
    if i < 800:
      tokens = input_text_list[i].split()
      total_target_sim = 0
      total_cand_sim = 0
      for j in range(0, len(tokens)):
        if tokens[j].find('기분') >= 0 and tokens[j] in dictionary:
          for k in range(j-window_size, j+window_size+1):
            if 0 <= k < len(tokens) and k != j:
              cand_word = tokens[j].replace('기분', '기본')
              if cand_word in dictionary:
                target_vec = np.ndarray(shape=(1, embedding_size), dtype=np.float32)
                target_vec[0] = embedding_vec[dictionary[tokens[j]]]
                cand_vec = np.ndarray(shape=(1, embedding_size), dtype=np.float32)
                cand_vec[0] = embedding_vec[dictionary[cand_word]]
                if tokens[k] in dictionary:
                  context_vec = np.ndarray(shape=(1, embedding_size), dtype=np.float32)
                  context_vec[0] = embedding_vec[dictionary[tokens[k]]]
                  target_similarity = tf.matmul(target_vec, context_vec, transpose_b=True)
                  cand_simiarity = tf.matmul(cand_vec, context_vec, transpose_b=True)
                else:
                  context_vec = np.ndarray(shape=(1, embedding_size), dtype=np.float32)
                  context_vec[0] = embedding_vec[0]
                  target_similarity = tf.matmul(target_vec, context_vec, transpose_b=True)
                  cand_simiarity = tf.matmul(cand_vec, context_vec, transpose_b=True)
              total_target_sim += target_similarity.eval()
              total_cand_sim += cand_simiarity.eval()
          if(total_cand_sim > total_target_sim):
            detect_count += 1
          total_count += 1
          break

  print('Detection Rate: ', detect_count / total_count * 100)

