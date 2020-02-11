
import cga

softmax_search_space = cga.SoftmaxMLP.SearchSpace(
    hidden_neurons=[()]  # no hidden layers
)

conv_search_space = cga.ConvolutionalLayer.SearchSpace(
    kernel_search_space=cga.Kernel.SearchSpace(
        depths=[64, 128, 256],
        resolutions=[(3, 3)],
        strides=[(1, 1)],
        padding_types=('S',)
    )
)

pool_kernel_search_space = cga.Kernel.SearchSpace(
    depths=[None],
    resolutions=[(2, 2)],
    strides=[(2, 2)],
    padding_types=('V',)
)

pool_search_space = cga.PoolingLayer.SearchSpace(
    pooling_types=['AVG', 'MAX'],
    kernel_search_space=pool_kernel_search_space,
    mutate=cga.StochasticFunction({
        lambda pooling_layer: pooling_layer.swap_pooling_type(pool_search_space): 1,
        lambda pooling_layer: pooling_layer.change_kernel(pool_search_space.kernel_search_space): 0,
    })
)

skip_search_space = cga.SkipLayer.SearchSpace(
    depths=(2,),
    mutate=cga.StochasticFunction({
        lambda skip_layer: skip_layer.add_conv(conv_search_space): 0,
        lambda skip_layer: skip_layer.remove_conv(): 0,
        lambda skip_layer: skip_layer.mutate_random_conv(conv_search_space): 1
    }),
    conv_search_space=conv_search_space
)

search_space = cga.CNN.SearchSpace(
    layers=[cga.PoolingLayer, cga.SkipLayer],
    init_depth_range=((10, 120)),
    conv_search_space=conv_search_space,
    skip_search_space=skip_search_space,
    clfs=[cga.SoftmaxMLP],
    clf_search_space=softmax_search_space,
    mutate=cga.StochasticFunction({
        lambda cnn: cnn.insert_layer(skip_search_space, cga.SkipLayer): .7,
        lambda cnn: cnn.insert_layer(pool_search_space, cga.PoolingLayer): .1,
        lambda cnn: cnn.insert_layer(conv_search_space, cga.ConvolutionalLayer): 0,
        lambda cnn: cnn.remove_layer(search_space):  .1,
        lambda cnn: cnn.change_layer(search_space):  .1,
    }),
    num_epoch_range=(2, 10)
)
