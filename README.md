# machine-learning-coursera
machine-learning-homework-coursera

P.S.：在submit的时候，可能会遇到以下反馈，提示你提交失败。

!! Submission failed: unexpected error: urlread: HTTP response code said error

!! Please try again later.

以下是解决方法：

It seems that the conversion from ASCII to the hexadecimal escape the jsonlib uses is not working properly anymore in Octave 4.0. You can get it fixed by replacing

str=[str str0(pos0(i)+1:pos(i)-1) sprintf('_0x%X_',str0(pos(i)))];
by

str=[str str0(pos0(i)+1:pos(i)-1) sprintf('_0x%X_',toascii(str0(pos(i))))];
and

str=sprintf('x0x%X_%s',char(str(1)),str(2:end));
by

str=sprintf('x0x%X_%s',toascii(str(1)),str(2:end));
in loadjson.m and makeValidFieldName.m

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


IrCostFunction

写出代码并不难，难在你要清楚为什么这样写代码，ex3的pdf中Vectorizing Logistic Regression 那小节的矩阵公式的推导是核心，要理解了才能写出了不需要循环的代码，提高运行效率。

predictOneVsAll

One-Vs-All中sigmoid函数的含义是用数学语言表达为P(y=i|X;θ).

=================ex4===================

nnCostFunction

在写J的时候卡在了，关于y的赋值的地方，我们用的是逻辑回归的CostFunction，那么y的值只可能是0或1，而从ex4data1.mat中我们得到的y是5000个10～1中的随机数，并不是0或1，所以需要我们将每个样本的y转化为0或1，再进行CostFunction的计算。具体看以下循环代码段：

for k = 1:num_labels

	y_k = (y == k);%将所有y是k的样本，转化为1，其他为0.

	a3_k = a3(:,k);%获得所有关于k的hθ的值

	J_K =1/m * sum(-y_k .* log(a3(:,k)) - (1 - y_k) .* log(1 - a3(:,k)));%利用y(k)和hθ(k)算出K值的CostFunction。

	J = J + J_K;%将所有的k的CostFunction加到一起

end

	其实在ex3中，就用到了类似y == k，将y样本转化为1或0的方式。

for i = 1:num_labels;

	initial_theta = zeros(n+1,1);

	options = optimset('GradObj','on','MaxIter',400);

	[theta] = fmincg(@(t)(lrCostFunction(t,X,(y == i),lambda)),initial_theta,options);
	
	all_theta(i,:)=theta';
end

以上代码中，

[theta] = fmincg(@(t)(lrCostFunction(t,X,(y == i),lambda)),initial_theta,options);

我们用 y == i，将y的样本转化为1或0.


