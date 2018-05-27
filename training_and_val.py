import os
import os.path
import math
#from scipy.misc import imread, imresize 

import numpy as np
import tensorflow as tf

import input_data
import VGG
import tools


IMG_W = 224
IMG_H = 224
N_CLASSES = 5
BATCH_SIZE = 8
EVA_BATCH_SIZE = 1
learning_rate = 0.001
MAX_STEP = 15000   
IS_PRETRAIN = True


#Training
def train():
    pre_trained_weights = 'vgg16_pretrain/vgg16.npy'
    train_data_dir = 'F:\\flowersdata\\tfrecord\\train\\traindata.tfrecords*'
    test_data_dir = 'F:\\flowersdata\\tfrecord\\test\\testdata.tfrecords*'
    train_log_dir = 'logs/train/'
    val_log_dir = 'logs/val/'
    with tf.name_scope('input'):
        #read train
        tra_image_batch, tra_label_batch = input_data.read_TFRecord(data_dir=train_data_dir,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=True,
                                                 in_classes=N_CLASSES)
        #read test
        val_image_batch, val_label_batch = input_data.read_TFRecord(data_dir=test_data_dir,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=False,
                                                 in_classes=N_CLASSES)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES]) 
    
    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    
    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
    train_op = tools.optimize(loss, learning_rate, my_global_step)   
    
    saver = tf.train.Saver(tf.global_variables())
#    summary_op = tf.summary.merge_all()   

       
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # load the parameter file, assign the parameters, skip the specific layers
    tools.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])   


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
#    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
#    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x:tra_images, y_:tra_labels})
            
            if step % 50 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                #summary_str = sess.run(summary_op)
                #tra_summary_writer.add_summary(summary_str, step)
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x:val_images,y_:val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))

                #summary_str = sess.run(summary_op)
                #val_summary_writer.add_summary(summary_str, step)
                    
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    


def evaluate():
    with tf.Graph().as_default():
        
        log_dir = 'logs/train/'
        test_data_dir = 'F:\\flowersdata\\tfrecord\\test\\testdata.tfrecords*'
        n_test = 502
                
#read test
        val_image_batch, val_label_batch = input_data.read_TFRecord(data_dir=test_data_dir,
                                                 batch_size= EVA_BATCH_SIZE,
                                                 shuffle=False,
                                                 in_classes=N_CLASSES)

        logits = VGG.VGG16N(val_image_batch, N_CLASSES, IS_PRETRAIN)
        correct = tools.num_correct_prediction(logits, val_label_batch)
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                print('\nEvaluating......')
                num_step = int(math.floor(n_test / EVA_BATCH_SIZE))
                num_sample = num_step*EVA_BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    step += 1
                print('Total testing samples: %d' %num_sample)
                print('Total correct predictions: %d' %total_correct)
                print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
               


if __name__=="__main__":
    #train()
    evaluate()

