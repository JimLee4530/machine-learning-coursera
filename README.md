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

plotDescisionBoundary

% Only need 2 points to define a line, so choose two endpoints

plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

% Calculate the decision boundary line

plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

以上两行代码的目的是画出决策边界，plot_x取的是x2中最大和最小的值加2.而plot_y画的其实不是y，而是x2，y是用来标记(x1,x2)点的类型的。
我们可以从 【θ1+ θ2×x1 + θ3×x2 = 0】得出来。那么问题来了，为什么会从这个式子中得出x2的值呢，为什么要等于0呢？我的理解是，在逻辑回归时，我们h小于0.5的判定为0，大于0.5的判定为1，h=-1/(1+exp(-θ×x)），当θ×x等于0时，h正好为0.5，那么正好是y分类的决策边界。

costFunctionReg

在对X进行正则化的时候，需要知道θ0对应的x1的值都为1，不需要进行正则化，所以在算正则化的时候不需要加入θ0.


================ex3===================

ex3_Part1:loading and visualizing data


%Randomly select 100 data points to display

rand_indices = randperm(m);% 得到一个1到m随机排列的1×m的行列式

sel = X(rand_indices(1:100),:);% 因为rand_indices(1:100)是1:m随机的100个数，所以sel从X中随机取出100个训练样本。

displayData(sel);%调用displayData()函数，画出样本.







