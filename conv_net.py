import math
import math
import random
import tensorflow as tf
sess = tf.InteractiveSession()

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class ConvLayer(object):

	def __init__(self, input_size, filter_size, output_size):
		
		self.input_size = input_size
		self.filter_size = filter_size
		self.output_size = output_size

	def conv_net(self, X, Y, X_test, Y_test):

		with tf.name_scope('place_holder'):
			# Generate placeholders for the images and labels
			images_placeholder = tf.placeholder(tf.float32, shape=[None,32,32,3],\
				name='images')
			labels_placeholder = tf.placeholder(tf.int32, shape=[None],\
				name = 'labels')
		
		with tf.name_scope('first_conv'):
			# define weights and biases
			w_conv1 = tf.Variable(\
				tf.truncated_normal([self.filter_size, self.filter_size, 3, 16],\
				stddev=1.0 / math.sqrt(float(self.input_size))), name='first_weights')
			b_conv1 = tf.Variable(tf.zeros([16]), name='first_biases')
			
			# set image size	
			#x_image = tf.reshape(images_placeholder, [-1, 32, 32, 3])
			
			# compute first convolution layer
			h_conv1 = tf.nn.relu(conv2d(images_placeholder, w_conv1) + b_conv1)	
			h_pool1 = max_pool_2x2(h_conv1)

		with tf.name_scope('second_conv'):
			# define weights and biases		
			w_conv2 = tf.Variable(\
				tf.truncated_normal([self.filter_size, self.filter_size, 16, 64],\
				stddev=1.0 / math.sqrt(float(self.input_size))), name='second_weights')
			b_conv2 = tf.Variable(tf.zeros([64]), name='second_biases')
	
			# compute second convolution layer
			h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
			h_pool2 = max_pool_2x2(h_conv2)

		with tf.name_scope('fc'):
			#define weights and biases
			w_fc1 = tf.Variable(\
				tf.truncated_normal([8*8*64, 1024],\
				stddev=1.0 / math.sqrt(float(self.input_size))), name='fc1_weights')
			#w_fc1 = weight_variable([8*8*16, 64])
			b_fc1 = tf.Variable(tf.zeros([1024]), name='fc1_biases')
			#b_fc1 = bias_variable([1024])
			
			# compute fc layer
			h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
			h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

		with tf.name_scope('dropout'):
			# compute dropout result
			keep_prob = tf.placeholder(tf.float32)
			h_fcl_drop = tf.nn.dropout(h_fc1, keep_prob)	 	
		
		with tf.name_scope('readout'):
			#w_fc2 = weight_variable([1024, 10])
			w_fc2 = tf.Variable(\
				tf.truncated_normal([1024, 20],\
				stddev=1.0 / math.sqrt(float(self.input_size))), name='fc1_weights')
			b_fc2 = tf.Variable(tf.zeros([20]), name='second_biases')
			#b_fc2 = bias_variable([10])
			
			# compute score
			#y_conv = tf.nn.softmax(tf.matmul(h_fcl_drop, w_fc2) + b_fc2)
			logits = tf.matmul(h_fcl_drop, w_fc2) + b_fc2

		with tf.name_scope('final_step'):
			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_placeholder)
			#cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(y_conv[range(400),labels_placeholder])))
			loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
			regularizer = tf.nn.l2_loss(w_conv1) + tf.nn.l2_loss(b_conv1)\
				+ tf.nn.l2_loss(w_conv2) + tf.nn.l2_loss(b_conv2)\
				+ tf.nn.l2_loss(w_fc1) + tf.nn.l2_loss(b_fc1)\
				+ tf.nn.l2_loss(w_fc2) + tf.nn.l2_loss(b_fc2)
			loss += 2e-2 * regularizer
			train_step = tf.train.GradientDescentOptimizer(2e-2).minimize(loss)
			#correct_prediction = tf.equal(tf.argmax(y_conv,0), tf.argmax(labels_placeholder,0))
			correct_prediction = tf.nn.in_top_k(logits, labels_placeholder, 1)
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			sess.run(tf.initialize_all_variables())
			train_his = []
			validation_his = []
			for i in range(5000):
				offset = random.randint(0,49599)
				batch_x = X[offset:(offset+350),:]
				batch_y = Y[offset:(offset+350)]
				batch_xte = X[(offset+350):(offset+400),:]
				batch_yte = Y[(offset+350):(offset+400)]
				if i%100 == 0:
					train_accuracy = accuracy.eval(feed_dict={\
						images_placeholder: batch_x, labels_placeholder: batch_y, keep_prob:1})
					print "step %d, training accuracy %g" %(i, train_accuracy)
					train_his.append(train_accuracy)

					validation_accuracy = accuracy.eval(feed_dict={\
						images_placeholder: batch_xte, labels_placeholder: batch_yte, keep_prob:1})
					print "step %d, validation accuracy %g" %(i, validation_accuracy)
					validation_his.append(validation_accuracy)
				if i%500 == 0:
					accuracy_part = 0
					for j in range(5):
						batch_xt = X_test[j*2000:(j+1)*2000,:]
						batch_yt = Y_test[j*2000:(j+1)*2000]
						accuracy_part += accuracy.eval(feed_dict={images_placeholder: batch_xt, labels_placeholder: batch_yt, keep_prob:1})
						print accuracy_part
				train_step.run(feed_dict={images_placeholder: batch_x, labels_placeholder: batch_y, keep_prob:1})
				#print "iteration number: %d"%(i)
			
			#print X_test.shape
			#print Y_test.shape
		#with tf.name_scope('prediction'):
			#feed_dict={images_placeholder: X_test, labels_placeholder: Y_test, keep_prob:1.0}
			accuracy_part = 0
			for j in range(5):
				batch_xt = X_test[j*2000:(j+1)*2000,:]
				batch_yt = Y_test[j*2000:(j+1)*2000]
				accuracy_part += accuracy.eval(feed_dict={images_placeholder: batch_xt, labels_placeholder: batch_yt, keep_prob:1})
				print accuracy_part
			#precision = sess.run(accuracy,feed_dict=feed_dict)

			print "train accuracy %g" %(train_his[-1])
			print "validation accuracy %g" %(validation_his[-1])
			print "test accuracy %g" %(accuracy_part/5.0)
