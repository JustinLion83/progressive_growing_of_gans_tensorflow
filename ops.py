import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np

# the implements of leakyRelu
def lrelu(x , alpha=0.2 , name="LeakyReLU"):
    # 可考慮給lrelu加上界線(6.0)
    with tf.name_scope(name):
        # return tf.minimum(6.0, tf.maximum(x , alpha*x))
        return tf.maximum(x , alpha*x)

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    '''獲取權重的方法, 用來調節各layer的學習速率(參考自He Normalize)'''
    if fan_in is None:
        fan_in = np.prod(shape[:-1])    # Input的Node數量(N*H*W)
    print "current", shape[:-1], fan_in
    std = gain / np.sqrt(fan_in)        # He init

    if use_wscale: # 是否乘以權重縮放係數 std
        wscale = tf.constant(np.float32(std), name='wscale') # 標準差
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
    
############################################################################################################
def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, gain=np.sqrt(2), use_wscale=False, padding='SAME',
           name="conv2d", with_w=False):
    '''
    param: d_h, d_w   kernel在 [h,w]移動的步長, 預設是2, 代表輸出的[h,w]會是input_的一半
    param: use_wscale 是否要權重縮放係數
    param: with_w     一個Flag, 控制是否要回傳 w跟biases
    '''
    with tf.variable_scope(name):
        
        # 獲取一個跟input_同樣DataType的權重, 形狀為[k_h, k_w, input_channel, output_dim]
        w = get_weight([k_h, k_w, input_.shape[-1].value, output_dim], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, input_.dtype)
        
        # 填充的方式...不想搞這個的話就用SAME
        if padding == 'Other':
            padding = 'VALID'
            input_ = tf.pad(input_, [[0,0], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        elif padding == 'VALID':
            padding = 'VALID'
            
        # 開始卷積操作
        conv   = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv   = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        # 是否要一起回傳 權重(w) & 修正(biase)
        if with_w:
            return conv, w, biases

        else:
            return conv

def fully_connect(input_, output_size, gain=np.sqrt(2), use_wscale=False, name=None, with_w=False):
  '''
  參數意思基本跟上面的Conv相同, 差別只在Shape而已
  '''
  shape = input_.get_shape().as_list()      # Ex [64,128,128,3]
  with tf.variable_scope(name or "Linear"):

    w = get_weight([shape[1], output_size], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, input_.dtype)
    bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))

    output = tf.matmul(input_, w) + bias # f(x) = input_ * w + bias

    if with_w:
        return output, with_w, bias

    else:
        return output
############################################################################################################
def conv_cond_concat(x, y):
    '''將 y 拼接到 x 的 feature_map 後面'''
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3 , [x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])])

def batch_normal(input , scope="scope" , reuse=False):
    # updates_collections=None 就不會將Moving_mean等加到Update_OP裡面, 可省去控制依賴那些動作(control_dependencies)
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse= reuse , updates_collections=None)
############################################################################################################
def resize_nearest_neighbor(x, new_size):
    '''上採樣方法 最近鄰插植'''
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)  # 重點在 h , w 其他兩個維度隨意, 因為這裡是要把圖片放大scale倍(預設是2倍)
    return resize_nearest_neighbor(x, (h * scale, w * scale))

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape                    # Ex [64,128,128,-1]

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape] # 回傳Shape, 如果是None的話就是-1
############################################################################################################
def downscale2d(x, k=2):
    '''下採樣用avg_pool'''
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')
############################################################################################################
def Pixl_Norm(x, eps=1e-8):
    '''
    Pixel norm沿着channel維度做歸一化，這樣歸一化的一個好處在於，feature map的每個位置都具有單位長度。
    這個歸一化策略與作者設計的Generator輸出有較大關係
    注意到Generator的輸出層並沒有Tanh或者Sigmoid激活函數
    '''
    # 找出該沿著哪個維度(Axis)做標準化
    if len(x.shape) > 2:
        axis_ = 3
    else:
        axis_ = 1
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis_, keep_dims=True) + eps)
############################################################################################################
def MinibatchstateConcat(input, averaging='all'):
    s = input.shape
    
    # 計算所有樣本的標準差
    adjusted_std = lambda x, **kwargs: tf.sqrt(tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) **2, **kwargs) + 1e-8)
    vals = adjusted_std(input, axis=0, keep_dims=True)
    
    # 所有像素位置標準差求均值
    if averaging == 'all':
        vals = tf.reduce_mean(vals, keep_dims=True)
    else:
        print ("nothing")
        
    # 複製vals構成特徵圖並加入fmaps
    vals = tf.tile(vals, multiples=[s[0], s[1], s[2], 1])
    return tf.concat([input, vals], axis=3)
