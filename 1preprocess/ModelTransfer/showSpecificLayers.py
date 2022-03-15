# import tensorflow as tf
import tensorflow.compat.v1 as tf
import imageio
import matplotlib.pyplot as plt
import numpy as np

'''
1.关于模型
'''
# 模型存储路径
model_path = "/毕设/experiment/风格迁移实验20200902/模型训练/vgg19模型训练部分/save/model.ckpt-599.meta"

'''
2.关于影像
'''
# 影像路径
img_path = '/毕设/experiment/风格迁移实验20200902/模型训练/vgg19模型训练部分/水墨风格样本/fg2.4.jpg' # 读取文件的路径
# 读取影像
image = imageio.imread(img_path)
# 将三维扩大至四维
# 上一步读取的影像为(x,y,z),但模型的输入应为(batch_size,x,y,z)，因此应该添加第0维（axis=0）
image = np.expand_dims(image, axis=0)
# 保存文件的路径
save_path = '/毕设/experiment/风格迁移实验20200902/模型训练/vgg19模型训练部分/data/fg1.3_conv41.jpg'

'''

3.关于节点
'''
x_name = 'batch:0' # 输入层
output_layer_name = 'conv4_1/BiasAdd:0' #'out/MatMul:0'# 待查看的节点名 # BiasAdd 可以不改

'''
4.查看分到各类的概率
查看前需把上面的改为 output_layer_name = 'out/MatMul:0'
'''
#tf.compat.v1.disable_eager_execution()
#test = tf.placeholder(tf.float32,shape=[1,2])
#alpha = tf.placeholder(tf.float32,shape=[1,2])
#alpha = tf.nn.softmax(test)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(model_path, clear_devices=True)
    '''
    # 查看各节点名称  
    # 在进行显示和打印前需单独进行这一部分，打印出所有节点名，确定要显示的节点名称（与output_layer_name对应）
    '''
    '''
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)   
    node_list = [n.name for n in graph_def.node]
    for node in node_list:
        print("node_name", node)
    '''
    #加载ckpt
    saver.restore(sess, tf.train.latest_checkpoint('/毕设/experiment/风格迁移实验20200902/模型训练/vgg19模型训练部分/save/'))

    '''
    # 运行需要显示的节点
    '''
    # 输入层
    x = tf.get_default_graph().get_tensor_by_name(x_name)
    # 待查看层（括号内为节点名称）
    test_layer = tf.get_default_graph().get_tensor_by_name(output_layer_name) 
    # 运行该节点
    test_layer_values = sess.run(test_layer, feed_dict={x:image}) 
    
    '''
    # 显示
    # 以第1卷积层的第1特征层输出为例
    '''
    # 最后一位代表通道数/特征层数，如0代表第1特征层，1代表第2特征层
#    plt.imshow(test_layer_values[0,:,:,0], cmap='viridis') # gist_gray 灰度显示
    # 删除绘图的坐标刻度
#    plt.axis('off')

    # 保存结果
#    imageio.imwrite(save_path,test_layer_values[0,:,:,0]) # 保存出来的是灰度

    '''
    # 打印出属于各类的概率
    # 需把前面定义第4部分注释符号删掉
    '''
#    print(sess.run(alpha, feed_dict={test:test_layer_values}))