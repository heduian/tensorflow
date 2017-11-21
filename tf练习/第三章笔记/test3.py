#coding:utf-8
# 随机数据矩阵 

import tensorflow as tf


# 反向传播 

w1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed = 1))

x = tf.placeholder(tf.float32,shape = (3,2),name = "input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()

sess.run(w1.initializer)
sess.run(w2.initializer)

#print sess.run(y)
print '---------------------------------------------------------------------------'
print 'w1:',sess.run(w1)
print 'w2:',sess.run(w2)
print '---------------------------------------------------------------------------'
print sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]})

sess.close()