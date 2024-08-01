import tensorflow as tf
import numpy as np

import utils.tf_utils as tf_utils
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cv2 as cv
import h5py
'''
打印图片
index = 3
plt.imshow(train_set_x_orig[index])
print(train_set_y_orig[0][index])
plt.show()
'''
# 加载数据
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = tf_utils.load_dataset()

# 重塑数据
train_set_X = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
train_X = train_set_X / 255.0  # 归一化
train_Y = train_set_y_orig

test_set_X = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
test_X = test_set_X / 255.0  # 归一化
test_Y = test_set_y_orig

train_Y = tf_utils.convert_to_one_hot(train_Y, 6)
test_Y = tf_utils.convert_to_one_hot(test_Y, 6)

print("训练集的数据维度：", train_X.shape)
print("训练集的标签维度：", train_Y.shape)

iterations = 5500
learning_rate = 0.00001
layer_dims = [train_X.shape[0], 20, 5, 6]  # 最后一层输出6个类别


def init_parameters(layer_dims):
    L = len(layer_dims)
    parameters = {}
    for l in range(1, L):
        parameters["W" + str(l)] = tf.Variable(
            tf.keras.initializers.GlorotNormal(seed=1)(shape=(layer_dims[l], layer_dims[l - 1])), name="W" + str(l))
        parameters["b" + str(l)] = tf.Variable(tf.zeros((layer_dims[l], 1)), name="b" + str(l))
    return parameters


def forward_propagation(parameters, X, layer_dims):
    L = len(layer_dims)
    A = X
    for j in range(1, L - 1):
        Z = tf.add(tf.matmul(parameters["W" + str(j)], A), parameters["b" + str(j)])
        A = tf.nn.relu(Z)
    ZL = tf.add(tf.matmul(parameters["W" + str(L - 1)], A), parameters["b" + str(L - 1)])
    return ZL


def compute_cost(Z, Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cost = tf.reduce_mean(loss)
    return cost


def model():
    parameters = init_parameters(layer_dims)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]

    seed = 0
    m = train_X.shape[1]
    costs = []
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    for i in range(iterations):
        seed += 1
        cost = 0
        mini_batches_size = 64
        mini_batches = tf_utils.random_mini_batches(train_X, train_Y, mini_batches_size, seed)
        for miniBatches in mini_batches:
            (mini_batches_X, mini_batches_Y) = miniBatches
            with tf.GradientTape() as tape:
                Z = forward_propagation(parameters, mini_batches_X, layer_dims)
                minibatch_cost = compute_cost(Z, mini_batches_Y)
            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_cost, trainable_variables)

            # grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            cost += minibatch_cost

        cost /= len(mini_batches)
        if i % 50 == 0:
            print(f"第{i}次迭代，损失值：{cost}")
            costs.append(cost.numpy())

    # 绘制损失值变化曲线
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per 50)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


# 定义计算准确率的函数
def compute_accuracy(Z3, Y):
    correct_prediction = tf.equal(tf.argmax(Z3, axis=0), tf.argmax(Y, axis=0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


# parameters = {}
parameters = model()


def test_accuracy():
    '''
        测试训练集和测试集准确率
    '''
    # 计算前向传播结果
    Z_train = forward_propagation(parameters, train_X, layer_dims)
    Z_test = forward_propagation(parameters, test_X, layer_dims)
    # # 计算准确率
    train_accuracy = compute_accuracy(Z_train, train_Y)
    test_accuracy = compute_accuracy(Z_test, test_Y)

    print("Train Accuracy:", train_accuracy.numpy())
    print("Test Accuracy:", test_accuracy.numpy())


test_accuracy()


def load_and_process_image(image_path, label):
    my_label_y = label
    image = imageio.imread(image_path)
    image_arr = np.array(image)
    my_image_x = cv.resize(image_arr, (64, 64)).reshape((64 * 64 * 3, 1))
    my_image_z = forward_propagation(parameters, my_image_x, layer_dims)
    accuracy = compute_accuracy(my_image_z, my_label_y)
    return accuracy


def test_my_image():
    path = "images/"
    image_0 = "0.JPG"
    image_1 = "1.JPG"
    image_2 = "2.JPG"
    image_3 = "3.JPG"
    image_4 = "4.JPG"
    image_5 = "5.JPG"

    label_0 = tf.constant([1, 0, 0, 0, 0, 0])
    label_1 = tf.constant([0, 1, 0, 0, 0, 0])
    label_2 = tf.constant([0, 0, 1, 0, 0, 0])
    label_3 = tf.constant([0, 0, 0, 1, 0, 0])
    label_4 = tf.constant([0, 0, 0, 0, 1, 0])
    label_5 = tf.constant([0, 0, 0, 0, 0, 1])

    flag0 = load_and_process_image(path + image_0, label_0)
    flag1 = load_and_process_image(path + image_1, label_1)
    flag2 = load_and_process_image(path + image_2, label_2)
    flag3 = load_and_process_image(path + image_3, label_3)
    flag4 = load_and_process_image(path + image_4, label_4)
    flag5 = load_and_process_image(path + image_5, label_5)
    if flag0:
        print("0.JPG 识别正确")
    else:
        print("0.JPG 识别错误")

    if flag1:
        print("1.JPG 识别正确")
    else:
        print("1.JPG 识别错误")

    if flag2:
        print("2.JPG 识别正确")
    else:
        print("2.JPG 识别错误")

    if flag3:
        print("3.JPG 识别正确")
    else:
        print("3.JPG 识别错误")

    if flag4:
        print("4.JPG 识别正确")
    else:
        print("4.JPG 识别错误")
    if flag5:
        print("5.JPG 识别正确")
    else:
        print("5.JPG 识别错误")


test_my_image()
