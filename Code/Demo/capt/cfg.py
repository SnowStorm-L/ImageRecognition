"""
train.py 参考文章：http://blog.topspeedsnail.com/archives/10858

capt demo 是来自 https://github.com/zhengwh/captcha-tensorflow
"""

# 验证码中的字符, 就不用汉字了

import os

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

gen_char_set = number + ALPHABET + alphabet  # 用于生成验证码的数据集
# 有先后的顺序的

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4  # 一共是4位
print("验证码文本最长字符数", MAX_CAPTCHA)  # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐

# 文本转向量
# char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
# char_set = number + alphabet
CHAR_SET_LEN = len(gen_char_set)

print('CHAR_SET_LEN:', CHAR_SET_LEN)

# 桌面上创建这几个文件夹
home_root = '/Users/l/desktop'  # 在不同操作系统下面Home目录不一样
workspace = os.path.join(home_root, 'work/crack/my_capt_data')  # 用于工作的训练数据集
model_path = os.path.join(home_root, 'work/crack/model')
model_tag = 'crack_capcha.model'
save_model = os.path.join(model_path, model_tag)

print('workspace:', workspace)
print('model_path:', save_model)

# 输出日志 tensorboard监控的内容
tb_log_path = '/tmp/mnist_logs'
