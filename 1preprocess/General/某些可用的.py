# 以下为读取ckpt模型 并打印所有tensor名称

import tensorflow as tf
meta_path = '/content/drive/MyDrive/2105Dinghu/saver5/model-49000.meta'
model_path = '/content/drive/MyDrive/2105Dinghu/saver5/model-49000'
saver = tf.train.import_meta_graph(meta_path) # 导入图
with tf.Session() as sess:

  saver.restore(sess, model_path) # 导入变量值

  tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]

  for tensor_name in tensor_name_list:
    if 'discriminator/' in tensor_name: #筛选
      print(tensor_name,'\n')