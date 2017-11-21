#coding:utf-8
# 简单的张量加减法运算   第三章学习

import tensorflow as tf

a = tf.constant([1.0,2.0], name = "a")
b = tf.constant([2.0,3.0], name = "b")

result = a + b

print '--------------------------------------------------------------------'
print a
print '--------------------------------------------------------------------'
print b
print '--------------------------------------------------------------------'
print result
print '--------------------------------------------------------------------'

#print tf.Session().run(result)


sess = tf.Session()
sess.run(result)
print sess.run(result)
print '--------------------------------------------------------------------'
with sess.as_default():
	print result.eval()
print '--------------------------------------------------------------------'
print result.eval(session = sess)
sess.close()  # 关闭后 释放  然后无法打印运算结果
