'''
.pt只保存模型参数，不保存结构
可以将pt转存为onnx/h5，方便可视化feature map

参考 https://blog.csdn.net/weixin_44034578/article/details/120947140?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-1-120947140.pc_agg_new_rank&utm_term=PyTorch%E6%A8%A1%E5%9E%8B%E8%BD%AC%E4%B8%BAH5&spm=1000.2123.3001.4430

input:
    model:需要提前加载或定义
    img: 一张图片
'''

import torch
import onnx
from onnx_tf.backend import prepare
from onnx2keras import onnx_to_keras
import keras
import tensorflow as tf


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def pth_to_onnx(torch_model, img_path, input_path, output_path):
    '''
    1)声明：使用本函数之前，必须保证你手上已经有了.pth模型文件.
    2)功能：本函数功能四将pytorch训练得到的.pth文件转化为onnx文件。
    '''

    torch_model = load_model(torch_model, input_path, True)
    torch_model.eval()

    transform = transforms.ToTensor()
    img = get_picture(img_path, transform)
    img = img.unsqueeze(0)
    x = img

    export_onnx_file = output_path  # 输出.onnx文件的文件路径及文件名
    torch.onnx.export(torch_model,
                      x,
                      export_onnx_file,
                      opset_version=9,  # 操作的版本，稳定操作集为9
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=["input"],  # 输入名
                      output_names=["output"],  # 输出名
                      dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                                    "output": {0: "batch_size"}}
                      )
    # onnx_model = onnx.load('model_all.onnx')    #加载.onnx文件
    # onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))       #打印.onnx文件信息


def onnx_to_pb(output_path):
    '''
    将.onnx模型保存为.pb文件模型
    '''
    # h5保存路径
    pboutput_path = output_path.replace("onnx", "pb")

    model = onnx.load(output_path)  # 加载.onnx模型文件
    tf_rep = prepare(model)
    tf_rep.export_graph(pboutput_path)  # 保存最终的.pb文件


def onnx_to_h5(output_path):
    '''
    将.onnx模型保存为.h5文件模型,并打印出模型的大致结构
    '''
    # h5保存路径
    h5output_path = output_path.replace("onnx", "h5")

    onnx_model = onnx.load(output_path)
    k_model = onnx_to_keras(onnx_model, ['input'])
    keras.models.save_model(k_model, h5output_path, overwrite=True, include_optimizer=True)  # 第二个参数是新的.h5模型的保存地址及文件名
    # 下面内容是加载该模型，然后将该模型的结构打印出来
    model = tf.keras.models.load_model(h5output_path)
    model.summary()
    print(model)


if __name__ == '__main__':
    # pt path
    input_path = "/content/drive/MyDrive/pytorchRelated/ResUnet-li/dataset/s2/checkpoints/GoogleattU_dV2_t3/GoogleattU_dV2_t3_checkpoint_4494.pt"  # 输入需要转换的.pth模型路径及文件名
    # onnx path
    output_path = "/content/drive/MyDrive/pytorchRelated/ResUnet-li/dataset/s2/checkpoints/model_all.onnx"  # 转换为.onnx后文件的保存位置及文件名

    # image
    img_path = '/content/drive/MyDrive/2105Dinghu/data/v2/train/image_crop/W1_1940_1746.tif'

    # model structure
    model = AttU_Net()  # 需要修改！！！！

    pth_to_onnx(model, img_path, input_path, output_path)  # 执行pth转onnx函数，具体转换参数去该函数里面修改
    # onnx_pre(output_path)   #【可选项】若有需要，可以使用onnxruntime进行部署测试，看所转换模型是否可用，其中，output_path指加载进去的onnx格式模型所在路径及文件名
    # onnx_to_pb(output_path)   #将onnx模型转换为pb模型
    onnx_to_h5(output_path)  # 将onnx模型转换为h5模型