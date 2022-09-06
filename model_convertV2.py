
import tensorflow as tf
import os
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.platform import gfile
import numpy as np

"""
将tf2.0 训练的网络 转换为pb文件
"""

# ------------------ 冻结图转换 ------------------ #
# 1.在tf2.x下训练模型，获得权重模型，必须要加载训练时的网络模型
# 2.网络结构里面的所有操作都是通过tf.keras完成的, 不能出现类似tf.nn 的tensorflow自己的操作符
# 3.tf2.0下保存的模型是.h5格式的,并且仅保存了weights, 即通过model.save_weights保存的模型.

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    将模型冻结为计算图
    会创建一个新的计算图，变量节点会被设置为常量值，会对模型进行裁剪，不需要输出的节点，将会被裁剪。
    @param session: The TensorFlow session to be frozen.
    @param keep_var_names: A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names: Names of the relevant graph outputs.
    @param clear_devices: Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def(add_shapes=True)
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph
 
def get_var_val():
    # 获取变量值
    sess = tf.Session()
    with gfile.FastGFile("./1/tf_model.pb", 'rb') as f: #自己保存的pb文件
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      sess.graph.as_default()
      tf.import_graph_def(graph_def, name='') 
      print(sess.run('avg/Mean/reduction_indices:0'))

def print_ops():
    # 打印出所有节点的名字,根据此来 设置输入输出节点的名称，例如 avg/Mean:0
    GRAPH_PB_PATH = './1/tf_model.pb' #path to your .pb file
    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            for i,n in enumerate(graph_def.node):
                print("Name of the node - %s" % n.name)

def predict(pb_path):
    sess = tf.Session()
    with gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图
        # 需要有一个初始化的过程
        sess.run(tf.global_variables_initializer()) 

        # 输入
        input_x = sess.graph.get_tensor_by_name('input:0')
        # input_y = sess.graph.get_tensor_by_name('y:0')

        # 输出     
        ave = sess.graph.get_tensor_by_name('avg/Mean:0')
        q_cls = sess.graph.get_tensor_by_name('out_cls/Mean:0')

        # 创建输入值
        na = np.random.random((1,1,257))
        print("-----------",na.shape)

        # 多输入，单输出
        # res = sess.run(ave, feed_dict={input_x: na, input_y: 5})

        # 多输出，单输入
        res = sess.run([ave,q_cls], feed_dict={input_x: na})
        # feed_dict参数的作用是替换图中的某个tensor的值或设置graph的输入值。
        print(res)


if __name__ == '__main__':
    # ------------------ 参  数  设  置 ---------------------#
    pb_path = "1"
    pb_name = "tf_model.pb" 
    h5_path = "test.h5" # 训练出的h5权重
    import model   # 训练时的模型名称
    
    # ------------------ 冻结模型为计算图 --------------------# 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # 如果模型里有dropout , bn层的话需要加上, 测试过加不加结果都一样
    tf.keras.backend.set_learning_phase(0)
    # 首先是定义你的模型, 这个需要和tf2.0下一毛一样
    MOSNet = model.CNN_BLSTM()
    Model = MOSNet.build() 
    Model.load_weights(h5_path)

    output_names = [out.op.name for out in Model.outputs] #对此列表进行修改，便可以控制裁剪的输出。['avg/Mean', 'frame/Reshape_1']
    print("-----------\n", output_names)
    print("-----------")

    
    # 此处outpus是模型的输出的节点的名称集合
    frozen_graph = freeze_session(tf.keras.backend.get_session(), 
    output_names= output_names) # )
    tf.train.write_graph(frozen_graph, pb_path, pb_name, as_text=False)

    # ------------------ 打印出所有节点 ------------------ #
    pb_path = os.path.join(pb_path,pb_name)
    print_ops()

    # ------------------ 获取节点的变量 ------------------ #
    # get_var_val()

    # ------------------ 预         测 ------------------ #
    predict(pb_path)
  