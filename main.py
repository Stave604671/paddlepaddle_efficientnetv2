"""

  EfficientNetV2: Smaller Models and Faster Training.
  https://arxiv.org/abs/2104.00298
"""
import copy
import paddle
import tools.efficientnetv2_config as efficientnetv2_config  # import get_model_config
import tools.hparams as hparams  # import base_config
import tools.utils as tool_utils  # import get_act_fn, cal_padding, round_filters, round_repeats, normalization


class SEBlock(paddle.nn.Layer):
    """
    copy from
    https://github.com/PaddlePaddle/PaddleX/blob/a244292498f0075618d15dd9178f4c52f378c21a/paddlex/ppcls/arch/backbone/model_zoo/efficientnet.py#L120
    """

    def __init__(self,
                 input_channels,
                 num_squeezed_channels,
                 oup,
                 mconfig):
        super(SEBlock, self).__init__()

        self._pool = paddle.nn.AdaptiveAvgPool2D(1)
        self._conv1 = paddle.nn.Conv2D(
            input_channels,
            num_squeezed_channels,
            kernel_size=1,
            weight_attr=paddle.nn.initializer.Normal(),
            bias_attr=True,
            padding="SAME"
        )
        self._act1 = paddle.nn.Swish()
        self._conv2 = paddle.nn.Conv2D(
            num_squeezed_channels,
            oup,
            kernel_size=1,
            weight_attr=paddle.nn.initializer.Normal(),
            bias_attr=True,
            padding="SAME")
        self._act2 = paddle.nn.Sigmoid()

    def forward(self, inputs):
        x = self._pool(inputs)
        x = self._conv1(x)
        x = self._act1(x)
        x = self._conv2(x)
        x = self._act2(x)
        out = paddle.multiply(inputs, x)
        return out


class Stem(paddle.nn.Layer):
    """Stem layer at the begining of the network."""

    def __init__(self, mconfig, input_channels, stem_filters, name=None):
        super().__init__()
        self._conv_stem = paddle.nn.Conv2D(
            in_channels=input_channels,
            out_channels=tool_utils.round_filters(stem_filters, mconfig),
            kernel_size=3,
            stride=2,
            weight_attr=paddle.nn.initializer.Normal(),
            bias_attr=False,
            padding='SAME')
        self._norm = tool_utils.normalization(
            mconfig.bn_type,
            axis=tool_utils.round_filters(stem_filters, mconfig),  # BatchNorm  input channels
            momentum=mconfig.bn_momentum,
            epsilon=mconfig.bn_epsilon,
            groups=mconfig.gn_groups)
        self._act = tool_utils.get_act_fn(mconfig.act_fn)

    def forward(self, inputs):
        x = self._conv_stem(inputs)
        x = self._norm(x)
        x = self._act(x)
        return x


class DepthwiseConvNorm(paddle.nn.Layer):
    def __init__(self,
                 input_channels,
                 block_args,
                 padding_type):
        super(DepthwiseConvNorm, self).__init__()
        self.k = block_args.kernel_size
        self.s = block_args.strides
        if isinstance(self.s, list) or isinstance(self.s, tuple):
            self.s = self.s[0]
        oup = block_args.input_filters * block_args.expand_ratio

        self._conv = paddle.nn.Conv2D(
            in_channels=input_channels,
            out_channels=oup,
            kernel_size=self.k,
            stride=self.s,
            bias_attr=False,
            groups=input_channels,
            padding=padding_type,  # SAME
            weight_attr=paddle.nn.initializer.Normal()
        )
        self._bn = paddle.nn.BatchNorm(
            num_channels=oup,
            act="swish",
            momentum=0.99,
            epsilon=0.001,
            param_attr=paddle.ParamAttr(),
            bias_attr=paddle.ParamAttr())

    def forward(self, inputs):
        return self._bn(self._conv(inputs))


class MbConvBlock(paddle.nn.Layer):
    def __init__(self,
                 input_channels,
                 block_args,
                 mconfig):
        super(MbConvBlock, self).__init__()

        # oup = block_args.input_filters * block_args.expand_ratio
        self._block_args = copy.deepcopy(block_args)
        self._mconfig = copy.deepcopy(mconfig)
        self._local_pooling = mconfig.local_pooling
        self._data_format = mconfig.data_format
        self._channel_axis = 1

        self._act = tool_utils.get_act_fn(mconfig.act_fn)
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)

        self.endpoints = None

        # self.id_skip = block_args.id_skip
        self.expand_ratio = block_args.expand_ratio
        self.drop_connect_rate = 0.2

        filters = self._block_args.input_filters * self._block_args.expand_ratio
        # kernel_size = self._block_args.kernel_size

        if self._block_args.expand_ratio != 1:
            self._expand_conv = paddle.nn.Conv2D(
                input_channels,
                out_channels=filters,
                # block_args,######################################################################################################
                kernel_size=1,
                stride=1,
                weight_attr=paddle.nn.initializer.Normal(),
                bias_attr=False,
                padding="SAME")
            self._norm0 = tool_utils.normalization(mconfig.bn_type, axis=filters,
                                                   momentum=mconfig.bn_momentum, epsilon=mconfig.bn_epsilon,
                                                   groups=mconfig.gn_groups)

        self._dcn = DepthwiseConvNorm(  ##################################################
            input_channels * block_args.expand_ratio,
            block_args,
            padding_type="SAME")
        self._norm1 = tool_utils.normalization(
            mconfig.bn_type,
            axis=block_args.input_filters * block_args.expand_ratio,
            momentum=mconfig.bn_momentum,
            epsilon=mconfig.bn_epsilon,
            groups=mconfig.gn_groups)
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se = SEBlock(
                input_channels=input_channels * block_args.expand_ratio,
                num_squeezed_channels=num_squeezed_channels,
                oup=self._block_args.input_filters * self._block_args.expand_ratio,
                mconfig=self._mconfig)
        else:
            self._se = None

        self._pcn = paddle.nn.Conv2D(
            input_channels * block_args.expand_ratio,
            block_args.output_filters,
            kernel_size=1,
            stride=1,
            padding="SAME",
            weight_attr=paddle.nn.initializer.Normal(),
            bias_attr=False
        )
        self._norm2 = tool_utils.normalization(
            mconfig.bn_type,
            axis=block_args.output_filters,
            momentum=mconfig.bn_momentum,
            epsilon=mconfig.bn_epsilon,
            groups=mconfig.gn_groups)

    @property
    def block_args(self):
        return self._block_args

    def residual(self, inputs, x, training, survival_prob):
        if (self._block_args.strides == 1 and
                self._block_args.input_filters == self._block_args.output_filters):
            # Apply only if skip connection presents.
            if survival_prob:
                x = tool_utils.drop_connect(x, training, survival_prob)
            x = paddle.add(x, inputs)

        return x

    def forward(self, inputs, survival_prob=None):
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._norm0(x)
            x = self._act(x)

        x = self._dcn(x)
        x = self._norm1(x)
        x = self._act(x)

        if self._mconfig.conv_dropout and self._block_args.expand_ratio > 1:
            x = paddle.nn.functional.dropout(x, self._mconfig.conv_dropout)

        if self._se:
            x = self._se(x)

        self.endpoints = {'expansion_output': x}

        x = self._pcn(x)
        x = self._norm2(x)
        x = self.residual(inputs, x, training=True, survival_prob=self.drop_connect_rate)

        return x


class FusedMbConvBlock(paddle.nn.Layer):
    def __init__(self, input_channels, block_args, mconfig):
        # super().__init__(input_channels, block_args, mconfig)
        super().__init__()
        self._mconfig = copy.deepcopy(mconfig)
        self._block_args = copy.deepcopy(block_args)
        filters = block_args.input_filters * block_args.expand_ratio
        kernal_size = block_args.kernel_size
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self._act = tool_utils.get_act_fn(mconfig.act_fn)
        if block_args.expand_ratio != 1:
            self._expand_conv = paddle.nn.Conv2D(in_channels=input_channels,
                                             out_channels=filters,
                                             kernel_size=kernal_size,
                                             stride=block_args.strides,
                                             weight_attr=paddle.nn.initializer.Normal(),
                                             bias_attr=False,
                                             padding="SAME",
                                             )
            self._norm0 = tool_utils.normalization(mconfig.bn_type,
                                                   axis=filters,
                                                   momentum=mconfig.bn_momentum,
                                                   epsilon=mconfig.bn_epsilon,
                                                   groups=mconfig.gn_groups)
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se = SEBlock(
                input_channels=input_channels * block_args.expand_ratio,
                num_squeezed_channels=num_squeezed_channels,
                oup=filters,
                mconfig=self._mconfig)
        else:
            self._se = None

        self._pcn = paddle.nn.Conv2D(
            input_channels * block_args.expand_ratio,
            block_args.output_filters,
            kernel_size=1 if block_args.expand_ratio != 1 else kernal_size,
            stride=1 if block_args.expand_ratio != 1 else block_args.strides,
            padding="SAME",
            weight_attr=paddle.nn.initializer.Normal(),
            bias_attr=False
        )
        self._norm1 = tool_utils.normalization(
            mconfig.bn_type,
            axis=block_args.output_filters,
            momentum=mconfig.bn_momentum,
            epsilon=mconfig.bn_epsilon,
            groups=mconfig.gn_groups)
    def residual(self, inputs, x, training, survival_prob):
        if (self._block_args.strides == 1 and
                self._block_args.input_filters == self._block_args.output_filters):
            # Apply only if skip connection presents.
            if survival_prob:
                x = tool_utils.drop_connect(x, training, survival_prob)
            x = paddle.add(x, inputs)

        return x
    def forward(self, inputs, survival_prob=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._norm0(x)
            x = self._act(x)

        self.endpoints = {'expansion_output': x}
        if self._mconfig.conv_dropout and self._block_args.expand_ratio > 1:
            x = paddle.nn.functional.dropout(x, self._mconfig.conv_dropout)

        if self._se:
            x = self._se(x)

        x = self._pcn(x)
        x = self._norm1(x)

        if self._block_args.expand_ratio == 1:
            x = self._act(x)

        x = self.residual(inputs, x, training=True, survival_prob=survival_prob)
        return x


class Head(paddle.nn.Layer):
    """Head layer for network outputs."""

    #
    def __init__(self, input_channels, mconfig):
        super().__init__()
        self.endpoints = {}
        self._mconfig = mconfig
        self._conv_head = paddle.nn.Conv2D(
            in_channels=input_channels,
            out_channels=tool_utils.round_filters(mconfig.feature_size or 1280, mconfig),
            kernel_size=1,
            stride=1,
            weight_attr=paddle.nn.initializer.Normal(),
            bias_attr=False,
            padding='SAME')
        self._norm = tool_utils.normalization(
            mconfig.bn_type,
            axis=tool_utils.round_filters(mconfig.feature_size or 1280, mconfig),
            # (1 if mconfig.data_format == 'channels_first' else -1),
            momentum=mconfig.bn_momentum,
            epsilon=mconfig.bn_epsilon,
            groups=mconfig.gn_groups)
        self._act = tool_utils.get_act_fn(mconfig.act_fn)

        self._avg_pooling = paddle.nn.AdaptiveAvgPool2D(1)  # AvgPool2D()#tf.keras.layers.GlobalAveragePooling2D(
        # data_format=mconfig.data_format)

        if mconfig.dropout_rate > 0:
            self._dropout = paddle.nn.Dropout(mconfig.dropout_rate)
        else:
            self._dropout = None

        self.h_axis, self.w_axis = ([2, 3] if mconfig.data_format
                                              == 'channels_first' else [1, 2])

    def forward(self, inputs):
        """Call the layer."""
        outputs = self._conv_head(inputs)
        outputs = self._norm(outputs)
        outputs = self._act(outputs)
        self.endpoints['head_1x1'] = outputs
        if self._mconfig.local_pooling:
            shape = outputs.get_shape().as_list()
            kernel_size = [1, shape[self.h_axis], shape[self.w_axis], 1]
            outputs = paddle.nn.functional.avg_pool2d(
                outputs, kernel_size=kernel_size, stride=[1, 1, 1, 1], padding='VALID')
            self.endpoints['pooled_features'] = outputs
            if self._dropout:
                outputs = self._dropout(outputs)
            self.endpoints['global_pool'] = outputs
            if self._fc:
                outputs = paddle.squeeze(outputs, [self.h_axis, self.w_axis])
                outputs = self._fc(outputs)
            self.endpoints['head'] = outputs
        else:
            outputs = self._avg_pooling(outputs)
            self.endpoints['pooled_features'] = outputs
            if self._dropout:
                outputs = self._dropout(outputs)
            self.endpoints['head'] = outputs
        return outputs


class EffNetV2Model(paddle.nn.Layer):
    """
    Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self,
                 model_name='efficientnetv2-s',
                 model_config=None,
                 include_top=True,
                 name=None):
        """Initializes an `Model` instance.
        Args:
        model_name: A string of model name.
        model_config: A dict of model configurations or a string of hparams.
        include_top: If True, include the top layer for classification.
        name: A string of layer name.
        Raises:
        ValueError: when blocks_args is not specified as a list.
        """
        super().__init__(model_name)
        cfg = copy.deepcopy(hparams.base_config)
        if model_name:
            cfg.override(efficientnetv2_config.get_model_config(model_name))
        cfg.model.override(model_config)
        self.cfg = cfg
        self._mconfig = cfg.model
        self.endpoints = None
        self.include_top = include_top
        self.in_channels = 3  # input_data[1]
        self._build()

    def _build(self):
        """Builds a model."""
        # global block_args
        blocks = []
        # Stem part.
        self._stem = Stem(self._mconfig, self.in_channels, self._mconfig.blocks_args[0].input_filters)
        # Builds blocks.
        output_filters = None
        for block_args in self._mconfig.blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            input_filters = tool_utils.round_filters(block_args.input_filters, self._mconfig)
            output_filters = tool_utils.round_filters(block_args.output_filters, self._mconfig)
            repeats = tool_utils.round_repeats(block_args.num_repeat,
                                               self._mconfig.depth_coefficient)
            block_args.update(
                dict(
                    input_filters=input_filters,
                    output_filters=output_filters,
                    num_repeat=repeats)
            )
            conv_block = {0: MbConvBlock, 1: FusedMbConvBlock}[block_args.conv_type]
            blocks.append(
                conv_block(block_args.input_filters, block_args,
                           self._mconfig))  ######################################???????
            if block_args.num_repeat > 1:
                block_args.input_filters = block_args.output_filters
                block_args.stride = 1
            for _ in range(block_args.num_repeat - 1):
                blocks.append(
                    conv_block(block_args.input_filters, block_args, self._mconfig)  # ?????????????????
                )
        self._blocks = paddle.nn.LayerList(blocks)
        print("maybe")
        self._head = Head(output_filters, mconfig=self._mconfig)
        print("")
        if self.include_top and self._mconfig.num_classes:
            self._flatten = paddle.nn.Flatten()
            self._fc = paddle.nn.Linear(in_features=self._mconfig.feature_size,
                                    out_features=self._mconfig.num_classes,
                                    weight_attr=paddle.nn.initializer.Uniform(),
                                    bias_attr=paddle.nn.initializer.Constant(self._mconfig.headbias or 0))
        else:
            self._fc = None

    def forward(self, inputs, with_endpoints=False):
        self.endpoints = {}
        reduction_idx = 0

        outputs = self._stem(inputs)

        self.endpoints['stem'] = outputs

        for idx, block in enumerate(self._blocks):
            is_reduction = False
            if ((idx == len(self._blocks) - 1) or
                    self._blocks[idx + 1]._block_args.strides > 1
            ):
                is_reduction = True
                reduction_idx += 1
            survial_prob = self._mconfig.survival_prob
            if survial_prob:
                drop_rate = 1.0 - survial_prob
                survial_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
                # 'block_%s survival_prob: %s idx survival_prob
            outputs = block(outputs, survival_prob=survial_prob)
            self.endpoints['block_%s' % idx] = outputs
            if is_reduction:
                self.endpoints['reduction_%s' % reduction_idx] = outputs
            if block.endpoints:
                for k, v in block.endpoints.items():
                    self.endpoints['block_%s/%s' % (idx, k)] = v
                    if is_reduction:
                        self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
        self.endpoints['features'] = outputs
        outputs = self._head(outputs)
        self.endpoints.update((self._head.endpoints))

        if self._fc:
            outputs = self._flatten(outputs)
            outputs = self._fc(outputs)

        if with_endpoints:
            return [outputs] + list(
                filter(lambda endpoint: endpoint is not None, [
                    self.endpoints.get('reduction_1'),
                    self.endpoints.get('reduction_2'),
                    self.endpoints.get('reduction_3'),
                    self.endpoints.get('reduction_4'),
                    self.endpoints.get('reduction_5'),
                ]))
        return outputs


if __name__ == '__main__':
    model = EffNetV2Model(model_name='efficientnetv2-s')
    data_inputs = paddle.rand(shape=[1, 3, 224, 224])
    out = model(data_inputs)
    paddle.save(model.state_dict(), "efficientnetv2-s.pdparams")
    paddle.jit.save(model, "efficientnetv2-s", [paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype="float32")])
