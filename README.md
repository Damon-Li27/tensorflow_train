在TensorFlow中，如果直接使用tf.Variable来初始化模型参数，并将这些变量传递给优化器进行训练，那么这些变量会在训练过程中被自动更新。因此，在这种情况下，不需要手动将更新后的变量回写到parameters字典中。

详细解释

	1.	参数初始化和存储：
	•	将模型参数（如W1, b1, 等）存储在parameters字典中，并使用tf.Variable来初始化这些参数。
	2.	训练变量列表：
	•	将parameters字典中的变量提取出来，放入trainable_variables列表中。
	3.	使用优化器进行训练：
	•	使用优化器（如Adam）和apply_gradients方法来更新这些变量。
在这个过程中，parameters 字典中的变量会在每次迭代中被自动更新。因此，不需要手动将更新后的变量回写到 parameters 中。只要你在初始化时使用 tf.Variable 并在训练时使用 TensorFlow 的优化器，变量的更新就会自动进行。

确保所有参数都是 tf.Variable 类型，并且在更新时传递的是这些变量的引用。如果你在初始化参数时使用了其他类型（如 tf.Tensor），那么它们将不会被自动更新。
