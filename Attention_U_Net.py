
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, UpSampling3D, Conv3D, Conv3DTranspose, Reshape, Concatenate, Add, add, Softmax, Lambda, Multiply, GlobalAveragePooling3D, Dense, Permute, BatchNormalization, Dropout, Activation


def conv_block(x, filter_size, size, dropout, batch_norm=False):
    
    conv = Conv3D(size, (filter_size), padding="same")(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=4)(conv)
    conv = Activation("relu")(conv)

    conv = Conv3D(size, (filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=4)(conv)
    conv = Activation("relu")(conv)
    
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    return conv

def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4),
                          arguments={'repnum': rep})(tensor)

#dDefining Gating signal for Attention Block
def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = Conv3D(out_size, (1, 1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

#Defining Attention Block
def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = Conv3D(inter_shape, (2, 2, 2), strides=(2, 2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv3D(inter_shape, (1, 1, 1), padding='same')(gating)
    upsample_g = Conv3DTranspose(inter_shape, (3, 3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3]),
                                 padding='same')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv3D(1, (1, 1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling3D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[4])

    y = layers.multiply([upsample_psi, x])

    result = Conv3D(shape_x[4], (1, 1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

########## ## Progressive Enhancement Module (PEM)

#Dilated Spatial Attention
def dilated_attention_module(x, dilation_rates, filters=64):
    # Apply dilated convolutions to generate q, k, and v
    q = Conv3D(filters, kernel_size=3, dilation_rate=dilation_rates, padding='same', activation='relu')(x)
    k = Conv3D(filters, kernel_size=3, dilation_rate=dilation_rates, padding='same', activation='relu')(x)
    v = Conv3D(filters, kernel_size=3, dilation_rate=dilation_rates, padding='same', activation='relu')(x)

    # Perform matrix multiplication
    #attention_map = tf.matmul(q, k, transpose_b=True)
    attention_map = Multiply()([q, k])
    attention_map = tf.transpose(attention_map, perm=[0, 1, 2, 3, 4])
    
    # Softmax normalization
    attention_map = Softmax(axis=-1)(attention_map)

    # Reshape attention map and multiply with v
    attention_output = Multiply()([attention_map, v])

    return attention_output

## Gated Convolution
def gated_convolution(input_high, input_low, filters=64, kernel_size=3):
    assert input_high.shape[-1] == input_low.shape[-1], "Number of channels must be the same in both inputs"

    # Apply convolution to generate gate maps
    gate_high = Conv3D(filters, kernel_size, padding='same')(input_high)
    gate_low = Conv3D(filters, kernel_size, padding='same')(input_low)

    # Multiply inputs with corresponding embedding matrices
    gated_high = Multiply()([gate_high, input_high])
    gated_low = Multiply()([gate_low, input_low])

    # Sigmoid activation for the gate maps
    gate_high = Activation('sigmoid')(gate_high)
    gate_low = Activation('relu')(gate_low)  # Changed activation to ReLU for gate_low

    # Ensure shapes match before element-wise addition
    reshape_lambda = Lambda(lambda x: tf.expand_dims(x, axis=1))
    gated_high = reshape_lambda(gate_high)
    gated_low = reshape_lambda(gate_low)

    # Final output after addition
    gated_conv_output = Add()([gated_high, gated_low])
    gated_conv_output = Lambda(lambda x: tf.reduce_sum(x, axis=1))(gated_conv_output)

    return gated_conv_output

## PEM implementation
def progressive_enhancement_module(input_tensor):
    # 1x1 Convolution with ReLU activation
    conv_1x1_input = Conv3D(filters=64, kernel_size=(1, 1, 1), activation='relu')(input_tensor)

    # 3x3 Convolution
    conv_3x3 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_1x1_input)

    # DSA with dilated rate of 2
    dsa_dilated_2 = dilated_attention_module(conv_1x1_input, dilation_rates=2)

    # Gated Convolution with DSA output and 3x3 Convolution output
    gc_1 = gated_convolution(conv_3x3, dsa_dilated_2)

    # DSA with dilated rate of 3 using conv_1x1_input
    dsa_dilated_3 = dilated_attention_module(conv_1x1_input, dilation_rates=3)

    # Gated Convolution with DSA output and previous Gated Convolution output
    gc_2 = gated_convolution(gc_1, dsa_dilated_3)

    # Combine 3x3 Convolution output, output of the first Gated Convolution, and final Gated Convolution output
    final_output = Add()([conv_3x3, gc_1, gc_2])

    # 1x1 Convolution with ReLU activation for the final output
    final_output = Conv3D(filters=64, kernel_size=1, activation='relu')(final_output)

    return final_output

########## For Semantic Guide Attention (SGA) Module
#Channel Selection block
def channel_selection(input_tensor):
    # Compute Channel Importance Scores Pc
    pc = GlobalAveragePooling3D()(input_tensor)

    # Dense layer to compute task correlation Aw
    w = Dense(units=1, activation='sigmoid')(pc)

    # Reshape to (1, 1, C) for broadcasting
    w = Reshape((1, 1, -1))(w)

    # Apply channel-wise scaling to the input
    selected_channels = Multiply()([input_tensor, w])

    return selected_channels

#Semantic Guide Attention (SGA)
def sga_module(low_level_input, high_level_input, channels=64):
    # Apply 3x3 convolution to low-level input
    low_level_conv = Conv3D(filters=channels, kernel_size=(3, 3, 3), padding='same', activation='relu')(low_level_input)

    # Apply 3x3 convolution to high-level input
    high_level_conv = Conv3D(filters=channels, kernel_size=(3, 3, 3), padding='same', activation='relu')(high_level_input)

    # Reshape low-level input into K and V
    k = low_level_conv
    v = low_level_conv

    # Apply channel selection (CS) to select important channels of K
    k = channel_selection(k)

    # Reshape high-level input into Q and apply CS to select important channels
    q = high_level_conv
    q = channel_selection(q)
    
    # Permute dimensions for scaling
    q = Permute((2, 1, 3, 4))(q)
    k = Permute((2, 1, 3, 4))(k)
    v = Permute((2, 1, 3, 4))(v)

    # Perform multiplication operation
    #attn_score = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True) / tf.sqrt(tf.cast(tf.shape(x[1])[-1], dtype=tf.float32)))((q, k))
    attn_score = Multiply()([q, k])
    
    #Applying softmax activation
    attn_score = Softmax(axis=-1)(attn_score)

    # Apply attention to values
    #attn_output = Lambda(lambda x: tf.matmul(x[0], x[1]))((attn_score, v))
    contextual_attention = Multiply()([attn_score, v])

    # Reshape and concatenate attention heads
       
    contextual_attention = Reshape((contextual_attention.shape[1], contextual_attention.shape[2], contextual_attention.shape[3], -1))(contextual_attention)

    # Concatenate reshaped low-level and high-level features
    sga_output = Concatenate(axis=-1)([high_level_conv, contextual_attention])

    return sga_output


## Defining Attention U-Net architecture
def Attention_UNet(input_shape, NUM_CLASSES=4, dropout_rate=0.0, batch_norm=True):
    print("Input shape:",input_shape)
    # network structure
    FILTER_NUM = 32 # number of basic filters for the first layer
    FILTER_SIZE = (3,3,3) # size of the convolutional filter
    UP_SAMP_SIZE = (2,2,2) # size of upsampling filters
    
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling3D(pool_size=(2,2,2))(conv_128)
    
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling3D(pool_size=(2,2,2))(conv_64)
   
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling3D(pool_size=(2,2,2))(conv_32)
    
    #adding PEM module at 3rd encoding
    pem_pool_16 = progressive_enhancement_module(pool_16) 
    
    # DownRes 4
    conv_16 = conv_block(pem_pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling3D(pool_size=(2,2,2))(conv_16)
    
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = layers.UpSampling3D(size=(UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=4)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    #adding SGA module at 2nd decoding
    sga_32 = sga_module(att_32, conv_32)
    up_32 = layers.UpSampling3D(size=(UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, sga_32], axis=4)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = layers.UpSampling3D(size=(UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=4)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling3D(size=(UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=4)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    conv_final = layers.Conv3D(NUM_CLASSES, kernel_size=(1,1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=4)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="Attention_UNet")
    return model


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



# input_shape = (128,128,128,3)
# Attention_UNet(input_shape, NUM_CLASSES=4, dropout_rate=0.0, batch_norm=True)
