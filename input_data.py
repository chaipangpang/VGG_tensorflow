import tensorflow as tf

# Reading data from TFRecord
def read_TFRecord(data_dir, batch_size, shuffle, in_classes):
    #分类数目
    num_classes = in_classes
    #获取record文件
    data_files = tf.gfile.Glob(data_dir)
    # 读取文件。
    filename_queue = tf.train.string_input_producer(data_files,shuffle=True) 
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    # 解析读取的样例。
    features = tf.parse_single_example(serialized_example,
                                       features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                       'img_width': tf.FixedLenFeature([], tf.int64),
                                       'img_height': tf.FixedLenFeature([], tf.int64),
                                       })  #取出包含image和label的feature对象
#tf.decode_raw可以将字符串解析成图像对应的像素数组
    #解析图片数据 string--unit8
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    height = tf.cast(features['img_height'],tf.int32)
    width = tf.cast(features['img_width'],tf.int32)
    label = tf.cast(features['label'], tf.int32)
    channel = 3
    image = tf.reshape(image, [height,width,channel])
    #reshape   向量---三维矩阵
    image = tf.reshape(image,[height,width,channel])
    #图像的缩放处理
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    #image = tf.image.resize_images(image, [240,240], method=0)
    image = tf.image.per_image_standardization(image)
    #unit8 -- float32
    #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5 
    image = tf.cast(image, tf.float32)
    #组合batch
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
                                             [image, label], 
                                             batch_size=batch_size,
                                             num_threads= 64,
                                             capacity=capacity, 
                                             min_after_dequeue=min_after_dequeue)
    else:
        image_batch, label_batch = tf.train.batch(
                                            [image, label], 
                                            batch_size=batch_size,
                                            num_threads = 64,
                                            capacity=capacity)
    ## ONE-HOT      
    label_batch = tf.reshape(label_batch, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    label_batch = tf.sparse_to_dense(
                  tf.concat(values=[indices, label_batch], axis=1),
                  [batch_size, num_classes], 1.0, 0.0)
    print(image_batch)
    print(label_batch)

    # 添加图片总结 summary
    #tf.summary.image('image_batch', image_batch)
    #return image_batch, label_batch

    #n_classes = 10
    #label_batch = tf.one_hot(label_batch, depth= n_classes)
    #label_batch = tf.cast(label_batch, dtype=tf.int32)
    #label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    return image_batch, label_batch







