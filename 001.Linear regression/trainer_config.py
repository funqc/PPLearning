# -*- coding: utf-8 -*-
from paddle.trainer_config_helpers import *

# 1. 定义数据来源，调用上面的process函数获得观测数据
data_file = 'empty.list'
with open(data_file, 'w') as f: f.writelines(' ')
define_py_data_sources2(train_list=data_file, test_list=None, module='dataprovider', obj='process', args={})

# 2. 学习算法。控制如何改变模型参数 w 和 b
settings(batch_size=12, learning_rate=1e-3, learning_method=MomentumOptimizer())

# 3. 神经网络配置
x = data_layer(name='x', size=1)
y = data_layer(name='y', size=1)
# 线性计算单元：y_predict = wx + b
y_predict = fc_layer(input=x, param_attr=ParamAttr(name='w'), size=1, act=LinearActivation(), bias_attr=ParamAttr(name='b'))
# 损失计算，度量 y_predict 和真实 y 之间的差距
cost = regression_cost(input=y_predict, label=y)
outputs(cost)
