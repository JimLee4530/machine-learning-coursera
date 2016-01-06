# machine-learning-coursera
machine-learning-homework-coursera

===============ex1=================

在course上学习机器学习，第一次实验时关于线性回归，较为基础。做题时曾卡在Gradient descent for multiple variables，因为是多特征量
且特征量的差值很大，我们对特征值进行了均值归一化，使其特征值均在正负3之间。但在最后预测时，没有考虑对预测的输入数据也做均值归一
化处理，使得预测的数据一直存在误差。最后将其改正，系统审核通过。

===============ex2=================

cost function and gradient

作业要求我们算出J和梯度Gradient,我一直以为是算迭代后的最终J和theta，对其进行自行迭代。最后发现J是NaN。看了pdf的指导，发现后面要求我们使用fminunc函数，上课时讲到的新方法来算J和theta，所以对原来的代码进行修改，仅根据公式算出梯度和J的算法。
1，发现自己的英文阅读能力有待提高。
2，需要在看下关于fminunc那集
