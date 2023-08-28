# -*- coding: utf-8 -*-

import tensorflow as tf

def batch_norm(input_, name='BN', bn_train=True):
    return tf.layers.batch_normalization(input_, scale=True, epsilon=1e-8,
                                    training=bn_train, name=name)
    
def conv2d(input, filters, kernel_size, name, strides = (1,1), 
                            paddings = 'same',dilation_rate=(1, 1)):
    return tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=strides,
                            padding=paddings, dilation_rate=(1, 1),
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), 
                            bias_initializer=tf.constant_initializer(0.01),
                            name=name)

def conv2d_transpose(input,filters,kernel_size,name,paddings='same',strides=[2,2]):
    return tf.layers.conv2d_transpose(input,filters,kernel_size, strides=strides, padding=paddings, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            bias_initializer=tf.constant_initializer(0.01),name=name)

def lrelu(inputs):
    return tf.nn.relu(inputs)

def SA(input, name='SA'):

    with tf.variable_scope(name):

        y_mean = tf.reduce_mean(input, axis=3, keepdims=True)  # (B, W, H, 1)
        y_max = tf.reduce_max(input, axis=3, keepdims=True)  # (B, W, H, 1)
        y_var = tf.layers.conv2d(input,1,1,padding='same',name='SA1')
        y_fuse = tf.concat([y_mean, y_max, y_var], axis=-1)     # (B, W, H, 2)
        y_sig = tf.layers.conv2d(y_fuse, 1, 7, padding='same', activation=tf.nn.sigmoid)    # (B, W, H, 1)
        y_sa = tf.multiply(input, y_sig)  # (B, W, H, C)

    return y_sa

def DenseBlockv1Att(input, num=64, name='dense', train_sign=True):
    with tf.variable_scope(name):
        if num != input.get_shape().as_list()[-1]:
            x = conv2d(input, num, [1,1], 'x')
        else:
            x = input
            
        c1 = conv2d(input, num, [3,3], 'c1')	
        b1 = batch_norm(c1, 'b1', bn_train=train_sign)
        r1 = lrelu(b1)
        c2 = conv2d(tf.concat([r1,input],3), num, [3,3], 'c2')
        b2 = batch_norm(c2, 'b2', bn_train=train_sign)
        r2 = lrelu(b2)
        c3 = conv2d(tf.concat([r2, r1, input],3), num, [3,3], 'c3')
        b3 = batch_norm(c3, 'b3', bn_train=train_sign)
        r3 = lrelu(SA(b3) + x)
        return r3

def downOp(inputs):
    return tf.nn.max_pool(inputs, [1,2,2,1],[1,2,2,1],'VALID')

def upOp(inputs, concat, num, name):
    up  = conv2d_transpose(inputs, num, [3,3], name)
    out = tf.concat([up, concat],3)
    return out  

def BaseModelv1SubAtt(inputs, train_sign=True, reuse=None):
    num = 8
    with tf.variable_scope('BaseModelSub', reuse=reuse):
        input_ = conv2d(inputs, num, [3,3], 'input_')
        encod1 = DenseBlockv1Att(input_, num, 'e1', train_sign)
        down1  = downOp(encod1)
        
        encod2 = DenseBlockv1Att(down1, num*2, 'e2', train_sign)
        down2  = downOp(encod2)
        
        encod3 = DenseBlockv1Att(down2, num*4, 'e3', train_sign)
        down3  = downOp(encod3)
        
        encod4 = DenseBlockv1Att(down3, num*8, 'e4', train_sign)
        down4  = downOp(encod4)
        
        encod5 = DenseBlockv1Att(down4, num*16, 'e5', train_sign)
        
        upsam1 = upOp(encod5, encod4, num*8, 'up1')
        decod1 = DenseBlockv1Att(upsam1, num*8, 'd1', train_sign)
        
        upsam2 = upOp(decod1, encod3, num*4, 'up2')
        decod2 = DenseBlockv1Att(upsam2, num*4, 'd2')
        
        upsam3 = upOp(decod2, encod2, num*2, 'up3')
        decod3 = DenseBlockv1Att(upsam3, num*2, 'd3')
        
        upsam4 = upOp(decod3, encod1, num, 'up4')
        decod4 = DenseBlockv1Att(upsam4, num, 'd4')
        
        out    = conv2d(decod4, 1, [3,3], 'out')
        return out, decod4
  
def BaseModelv1MainAtt(inputs, train_sign=True, reuse=None):
    o1, f1 = BaseModelv1SubAtt(inputs, train_sign, reuse)
    
    num = 16
    with tf.variable_scope('BaseModelMain', reuse=reuse):
        input_ = conv2d(inputs, num, [3,3], 'input_')
        encod1 = DenseBlockv1Att(input_, num, 'e1', train_sign)
        down1  = downOp(encod1)
        
        encod2 = DenseBlockv1Att(down1, num*2, 'e2', train_sign)
        down2  = downOp(encod2)
        
        encod3 = DenseBlockv1Att(down2, num*4, 'e3', train_sign)
        down3  = downOp(encod3)
        
        encod4 = DenseBlockv1Att(down3, num*8, 'e4', train_sign)
        down4  = downOp(encod4)
        
        encod5 = DenseBlockv1Att(down4, num*16, 'e5', train_sign)
        
        upsam1 = upOp(encod5, encod4, num*8, 'up1')
        decod1 = DenseBlockv1Att(upsam1, num*8, 'd1', train_sign)
        
        upsam2 = upOp(decod1, encod3, num*4, 'up2')
        decod2 = DenseBlockv1Att(upsam2, num*4, 'd2')
        
        upsam3 = upOp(decod2, encod2, num*2, 'up3')
        decod3 = DenseBlockv1Att(upsam3, num*2, 'd3')
        
        upsam4 = upOp(decod3, encod1, num, 'up4')
        decod4 = DenseBlockv1Att(upsam4, num, 'd4')
        
        out    = conv2d(tf.concat([decod4, f1],3), 1, [3,3], 'out')
        
        return tf.nn.relu(out + inputs), o1    
      
        
        
        
        



