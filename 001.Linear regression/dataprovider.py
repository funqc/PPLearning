# -*- coding:utf-8 -*-
from  paddle.trainer.PyDateProvider2 import *
import random

# 定义输入数据的类型：2个浮点数
@provider(input_type=[dense_vector(1), dense_vector(1)], use_seq=False)
def process(settings, input_file):
    for i in xrange(2000):
        x = random.random()
        yield [x], [2*x+0.3]
