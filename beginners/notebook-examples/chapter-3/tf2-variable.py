import tensorflow as tf

# 定义变量 W
W = tf.Variable(
    initial_value=tf.random.normal(
        shape=(1, 4), 
        mean=100, 
        stddev=0.35
    ), 
    name="W"
)

# 打印变量 W 的值
print(W.numpy())

# 使用变量 W 进行一些计算
some_input = tf.constant([[1.0, 2.0, 3.0, 4.0]])
output = tf.matmul(some_input, W, transpose_b=True)
print(output.numpy())



import tensorflow as tf

# 创建变量
W = tf.Variable(tf.random.normal(shape=(1, 4), mean=100, stddev=0.35), name="W")
b = tf.Variable(tf.zeros([4]), name="b")

# 打印变量的值
print("W:", W.numpy())
print("b:", b.numpy())

# 更新变量 b 的值
b.assign_add([1, 1, 1, 1])
print("Updated b:", b.numpy())

# 保存模型
checkpoint = tf.train.Checkpoint(W=W, b=b)
checkpoint.save('./summary/test.ckpt')

# 恢复模型
new_checkpoint = tf.train.Checkpoint(W=tf.Variable(tf.zeros_like(W)), b=tf.Variable(tf.zeros_like(b)))
new_checkpoint.restore(tf.train.latest_checkpoint('./summary'))

print("Restored W:", new_checkpoint.W.numpy())
print("Restored b:", new_checkpoint.b.numpy())
