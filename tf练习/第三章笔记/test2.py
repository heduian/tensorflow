#coding:utf-8
# 随机数据矩阵 

import tensorflow as tf

# shape = [2,3]  2行3列
weights = tf.Variable(tf.random_normal([2,3],stddev = 2))
biases = tf.Variable(tf.zeros([3]))
print weights
print biases

# w2 = tf.Variable(weights.initialized_value())
# w3 = tf.Variable(weights.initialized_value()*2)

# sess = tf.Session()

# sess.run(weights)


# 例子

w1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed = 1))

x = tf.constant([[0.7,0.9]])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()

sess.run(w1.initializer)
sess.run(w2.initializer)

#print sess.run(y)
print '---------------------------------------------------------------------------'
print 'w1:',sess.run(w1)
print 'w2:',sess.run(w2)
print 'x:',sess.run(x)
print 'a:',sess.run(a)
print 'y:',sess.run(y)
print '---------------------------------------------------------------------------'
sess.close()