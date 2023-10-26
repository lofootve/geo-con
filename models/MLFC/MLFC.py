import tensorflow as tf


class ConvBlock(tf.keras.layer.Layer):
    def __init__(self, num):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(num, (3,3), padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation("relu")
        self.max = tf.keras.layers.MaxPool2D((2, 2))
    
    def call(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.act(x)
        x = self.max(x)
        return x 
    
    
class BuildConvBlock(tf.keras.layer.Layer):
    def __init__(self, block_list):
        super().__init__()
        self.block = []
        for i in block_list:
             self.temp = ConvBlock(i)
             self.block.append(self.temp)
        self.flatten = tf.keras.layers.Flatten()   
             
    def __call__(self, input):
        x = self.block[0](input)
        x = self.block[1](x)
        x = self.block[2](x)
        x = self.flatten(x)
        
        return x
        
        
class DenseBlock(tf.keras.layer.Layer):
    def __init__(self, num):
        super().__init__()
        self.dense = tf.keras.layers.Dense(num)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation("relu")
        
    def __call__(self, x):
        x = self.dense(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x
    

class BuildDenseBlock(tf.keras.layer.Layer):
    def __init__(self, block_list):
        super().__init__()
        self.block = []
        for i in block_list:
            self.temp = DenseBlock(i)
            self.block.append(self.temp)
            
    def __call__(self, sum):
        x = self.block[0](sum)
        x = self.block[1](x)
        x = self.block[2](x)
        
        return x
        
        
                
class MLFC(tf.keras.Model):
    def __init__(self, row, col, conv_block_list, dense_block_list):
        super(MLFC, self).__init__()
        self.p_pdg_input = tf.keras.layers.Input(shape=(row, col-5, 3))
        self.p_tpt_input = tf.keras.layers.Input(shape=(row, col-1, 3))
        self.t_tpt_input = tf.keras.layers.Input(shape=(row, col-2, 3))
        self.p_mon_ckp_input = tf.keras.layers.Input(shape=(row, col-3, 3))
        self.t_jus_ckp_input = tf.keras.layers.Input(shape=(row, col-4, 3))

        self.p_pdg_convblock = BuildConvBlock(conv_block_list)
        self.p_tpt_convblock = BuildConvBlock(conv_block_list)
        self.t_tpt_convblock = BuildConvBlock(conv_block_list)
        self.p_mon_ckp_convblock = BuildConvBlock(conv_block_list)
        self.t_jus_ckp_convblock = BuildConvBlock(conv_block_list)
        
        self.concatenated = tf.keras.layers.concatenate()
        self.sum = tf.reduce_sum(axis=1)

        self.dense_block = BuildDenseBlock(dense_block_list)
        self.output = tf.keras.layers.Dense(9, activation = "softmax")
        
    def call(self, inputs):
        p_pdg_input = self.p_pdg_input(inputs[0])
        p_tpt_input = self.self.p_tpt_input(inputs[1])
        t_tpt_input = self.self.p_tpt_input(inputs[2])
        p_mon_ckp_input = self.self.p_mon_ckp_input(inputs[3])
        t_jus_ckp_input = self.self.t_jus_ckp_input(inputs[4])
        
        p_pdg = self.p_pdg_convblock(self.p_pdg_input)
        p_tpt = self.p_tpt_convblock(self.p_tpt_input)
        t_tpt = self.t_tpt_convblock(self.p_tpt_input)
        p_mon_ckp = self.p_mon_ckp_convblock(self.p_mon_ckp_input)
        t_jus_ckp = self.t_jus_ckp_convblock(self.t_jus_ckp_input)
        
        contcat = self.concatenated([p_pdg, p_tpt, t_tpt, p_mon_ckp, t_jus_ckp], axis=1)
        sum = self.sum(contcat, axis=1)

        output = self.dense_block(sum)
        output = self.output(output)
        
        return output