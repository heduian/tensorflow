#coding:utf-8
import tensorflow as tf
from numpy.random import RandomState

# 完整的神经网络

# 定义训练数据batch的大小
batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed = 1))

# n 行 2列   一口气把所有数据喂进去
x = tf.placeholder(tf.float32,shape = (None,2),name = "x-input")
y_ = tf.placeholder(tf.float32,shape = (None,1),name = "y-input")

# 定义前向传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数和反向传播的算法

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1 + x2 < 1)] for (x1 , x2) in X]

# 创建会话来运行tf程序
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()      #tf.initialize_all_variables()   之前的写法已经过时 
	sess.run(init_op)

	print '---------------------------------------------------'
	print sess.run(w1)
	print sess.run(w2)
	print '---------------------------------------------------'

#------------------------------------------------------------------------------------------#
	#设定训练的轮数
	#  需要带with代码块内  因为with结束后 sess会被关掉， 只有在代码块内  下面的才可以运行
	STEPS = 5000

	for i in range(STEPS):
		#每次选取batch_size个样本进行训练
		start = (i * batch_size) % dataset_size
		end = min(start + batch_size , dataset_size)

		# 通过选取的样本训练神经网络并更新参数
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		if i %1000 == 0:
			total_cross_entropy = sess.run(cross_entropy,feed_dict = {x:X,y_:Y})
			print("After %d training step(s),cross entropy on all data is %g" %(i,total_cross_entropy))

	print '---------------------------------------------------'
	print sess.run(w1)
	print sess.run(w2)
	print '---------------------------------------------------'







# sess = tf.Session()

# sess.run(w1.initializer)
# sess.run(w2.initializer)

# #print sess.run(y)
# print '---------------------------------------------------------------------------'
# print 'w1:',sess.run(w1)
# print 'w2:',sess.run(w2)
# print '---------------------------------------------------------------------------'
# print sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]})

# sess.close()