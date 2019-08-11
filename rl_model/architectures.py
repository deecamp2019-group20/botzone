from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras import backend as KTF


def bottleneck(inp, channels, use_shortcut = False):
    in_shape = KTF.int_shape(inp)
    conv1 = Conv2D(channels, (3,3), activation='relu', padding='same')(inp)
    conv2 = Conv2D(channels, (3,3), activation='relu', padding='same')(conv1)
    if use_shortcut:
        shortcut = inp
        if in_shape[3]!=channels:
            shortcut = Conv2D(channels, (1,1))(inp)
        plus = Add()([conv2, shortcut])
        return Activation('relu')(plus)
    return conv2


def MiniResNet(input_shape, use_HER):
    state_in = Input(shape=input_shape)
    action_in = Input(shape=(15, 4))
    action = Reshape((15, 4, 1))(action_in)
    if use_HER:
        goal_in = Input(shape=(15, 4))
        goal = Reshape((15, 4, 1))(goal_in)
        In = Concatenate()([state_in, action, goal])
        inputs = [state_in, action_in, goal_in]
    else:
        In = Concatenate()([state_in, action])
        inputs = [state_in, action_in]
    block1 = bottleneck(In, 64, True)
    block2 = bottleneck(block1, 128, True)
    block3 = bottleneck(block2, 256, True)
    block4 = bottleneck(block3, 256, True)
    pool = MaxPooling2D((15,4))(block4)
    out = Flatten()(pool)
    out = Dense(1000, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(1)(out)
    m = Model(inputs=inputs, outputs=[out])
    return m


def SimpleConv(input_shape, use_HER):
    state_in = Input(shape=input_shape)
    action_in = Input(shape=(15, 4))
    action = Reshape((15, 4, 1))(action_in)
    if use_HER:
        goal_in = Input(shape=(15, 4))
        goal = Reshape((15, 4, 1))(goal_in)
        In = Concatenate()([state_in, action, goal])
        inputs = [state_in, action_in, goal_in]
    else:
        In = Concatenate()([state_in, action])
        inputs = [state_in, action_in]

    conv0 = Conv2D(512, (1, 4), activation='relu')(In)
    conv1 = Conv2D(64, (1, 1), activation='relu', padding='SAME')(conv0)
    conv2 = Conv2D(64, (3, 1), activation='relu', padding='SAME')(conv0)
    conv5 = Conv2D(128, (5, 1), activation='relu', padding='SAME')(conv0)
    conv9 = Conv2D(128, (9, 1), activation='relu', padding='SAME')(conv0)
    conv13= Conv2D(256, (13,1), activation='relu', padding='SAME')(conv0)
    conv15= Conv2D(256, (15,1), activation='relu', padding='SAME')(conv0)
    conc = Concatenate()([conv1, conv2, conv5, conv9, conv13, conv15])  #None x 15 x 1 x all channel
    conv1x1 = Conv2D(512, (1,1))(conc)
    bnorm = BatchNormalization()(conv1x1)
    acti = Activation('relu')(bnorm)
    out = Flatten()(acti)
    out = Dense(256, activation='relu')(out)
    out = Dropout(0.2)(out)
    out = Dense(1)(out)
    m = Model(inputs=inputs, outputs=[out])
    return m

