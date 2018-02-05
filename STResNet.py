'''
This code implements ST-ResNet: deep spatial-temporal residual network
'''
from __future__ import print_function
from keras.layers import Input
from keras.layers import Activation
from keras.layers import merge
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


# from keras.utils.visualize_util import plot

# Short cut
def _shortcut(input, residual):
    return merge([input, residual], mode='sum')


# Batch normalization residual network
# def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=True):
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             border_mode="same")(activation)

    return f


# Residual net unit
def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        # TODO: test no convolution case
        # residual=_bn_relu_conv(nb_filter, 1, 1)(input)
        # residual=_bn_relu_conv(nb_filter, 1, 1)(residual)
        return _shortcut(input, residual)

    return f


# Residual net unit
def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter, init_subsample=init_subsample)(input)
        return input

    return f


# Spatial-temporal residual network
# TODO: (3, 1, 32, 32) needs to be changed for different cases
def stresnet(c_conf=(3, 1, 32, 32), p_conf=(3, 1, 32, 32), t_conf=(3, 1, 32, 32), external_dim=5,
             nb_residual_unit=8):  # TODO: change nb_residual_unit->8
    '''
    c: temporal closeness; p: period; t: trend
    conf=(len_sequence, nb_flow, map_height, map_width)
    external_dim: dimension of meta-data and other external features
    nb_residual_unit: number of residual layers
    '''
    main_inputs = []  # Input data
    outputs = []  # After conv+res+conv to get output

    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            main_inputs.append(input)

            # Conv1
            conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(
                input)  # TODO: 64 can be increased
            # TODO: test no convolution case
            # conv1=Convolution2D(nb_filter=64, nb_row=1, nb_col=1, border_mode="same")(input)
            # [nb_residual_unit] Residual units
            residual_output = ResUnits(_residual_unit, nb_filter=64, repetations=nb_residual_unit)(
                conv1)  # TODO: 64 can be increased
            # Activation
            activation = Activation('relu')(residual_output)
            # Conv2
            conv2 = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(activation)
            # TODO: test no convolution case
            # conv2=Convolution2D(nb_filter=nb_flow, nb_row=1, nb_col=1, border_mode="same")(activation)
            outputs.append(conv2)

        # Parameter-matrix based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        from iLayer import iLayer
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = merge(new_outputs, mode='sum')

    # Fusing with external components
    if external_dim != None and external_dim > 0:
        # External input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(output_dim=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        main_output = merge([main_output, external_output], mode='sum')
    else:
        print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)
    model = Model(input=main_inputs, output=main_output)
    return model


# The main function
if __name__ == "__main__":
    model = stresnet(external_dim=28, nb_residual_unit=12)
    # plot(model, to_file='ST-ResNet.png', show_shapes=True)
    model.summary()
