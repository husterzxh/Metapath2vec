#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os

import numpy as np
import tensorflow as tf
from GenerateVocabulary import VocabularyGenerator

def build_model(batch_size,vocab_size,embed_size,num_sampled):
    center_node = tf.placeholder(tf.int32, shape=[batch_size], name='center_node')
    context_node = tf.placeholder(tf.int32, shape=[batch_size, 1], name='context_node')
    negative_samples = (tf.placeholder(tf.int32, shape=[num_sampled], name='negative_samples'),
        tf.placeholder(tf.float32, shape=[batch_size,1], name='true_expected_count'),
        tf.placeholder(tf.float32, shape=[num_sampled], name='sampled_expected_count'))
    embed_matrix = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1.0, 1.0),
                        name='embed_matrix')
    embed = tf.nn.embedding_lookup(embed_matrix, center_node, name='embed')
    nce_weight = tf.Variable(tf.truncated_normal([vocab_size, embed_size],
                                                stddev=1.0 / (embed_size ** 0.5)),
                                                name='nce_weight')
    nce_bias = tf.Variable(tf.zeros([vocab_size]), name='nce_bias')
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                        biases=nce_bias,
                                        labels=context_node,
                                        inputs=embed,
                                        sampled_values = negative_samples,
                                        num_sampled=num_sampled,
                                        num_classes=vocab_size), name='loss')
    loss_summary = tf.summary.scalar("loss_summary", loss)

    return center_node, context_node, negative_samples, loss

def traning_op(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimizer

def train(center_node_placeholder,context_node_placeholder,negative_samples_placeholder,loss,dataset,optimizer,NUM_EPOCHS,batch_size,num_sampled,care_type,LOG_DIRECTORY,LOG_INTERVAL,MAX_KEEP_MODEL):
    care_type = True if care_type==1 else False
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=MAX_KEEP_MODEL)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0
        writer = tf.summary.FileWriter(LOG_DIRECTORY, sess.graph)
        global_iteration = 0
        iteration = 0
        while (dataset.epoch <= NUM_EPOCHS):
            current_epoch=dataset.epoch
            center_node_batch,context_node_batch  = dataset.get_batch(batch_size)
            negative_samples  = dataset.get_negative_samples(pos_index=context_node_batch[0],num_negatives=num_sampled,care_type=care_type)
            context_node_batch = context_node_batch.reshape((-1,1))
            loss_batch, _ ,summary_str = sess.run([loss, optimizer,merged],
                                    feed_dict={
                                    center_node_placeholder:center_node_batch,
                                    context_node_placeholder:context_node_batch,
                                    negative_samples_placeholder: negative_samples
                                    })
            writer.add_summary(summary_str,global_iteration)
            total_loss += loss_batch
            iteration+=1
            global_iteration+=1
            if LOG_INTERVAL > 0:
                if global_iteration % LOG_INTERVAL == 0:
                    total_loss = 0.0

            if dataset.epoch - current_epoch > 0:
                dataset.shffule()
                total_loss = 0.0
                iteration=0

        model_path=os.path.join(LOG_DIRECTORY,"model_final.ckpt")
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
        writer.close()

        print("Save final embeddings as numpy array")
        np_node_embeddings = tf.get_default_graph().get_tensor_by_name("embedding_matrix/embed_matrix:0")
        np_node_embeddings = sess.run(np_node_embeddings)
        np.savetxt('b.txt', np_node_embeddings, fmt='%.6f')

if __name__ == "__main__":
    meta_path_txt = './data/in_dbis/dbis.cac.w1000.l100.newconf.txt'
    window_size = 7
    batch_size = 1
    embedding_dim = 128
    negative_samples = 5
    learning_rate = 0.01
    epochs = 100
    type = 0 # metapath2vec
    log = r'./log'
    dataset=VocabularyGenerator(meta_path_txt=meta_path_txt, window_size=window_size)
    center_node_placeholder,context_node_placeholder,negative_samples_placeholder,loss = build_model(batch_size=batch_size,vocab_size=len(dataset.dict_node_index),embed_size=embedding_dim,num_sampled=negative_samples)
    optimizer = traning_op(loss,learning_rate=learning_rate)
    train(center_node_placeholder,context_node_placeholder,negative_samples_placeholder,loss,dataset,optimizer,NUM_EPOCHS=epochs,batch_size=1,num_sampled=negative_samples,care_type=type,LOG_DIRECTORY=log,LOG_INTERVAL=-1,MAX_KEEP_MODEL=10)


