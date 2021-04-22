# Multi-behavior Recommendation with Graph Convolutional Networks

**要解决什么问题**

对于电商数据来说，协同过滤对没有过购买行为的用户难以进行较好的推荐，但是可以用其他收藏加购信息作为附加信息对用户进行推荐。这样的问题叫做多行为推荐。

**现在的方法有什么问题**

主要分为两种解决方案：

- 使用多行为数据到采样过程，并使用多类型的采样对加强模型的学习过程。
  - MCBRP
- 第二种方案主要是设计一种模型去抓住多行为信息。
  - 矩阵分解

**这些方法有哪些问题**

- 没有充分利用不同种行为的强度，并且不一定购买这种行为的强度比分享强。所以多反馈推荐模型应当更合理利用多行为的信息
- 多种行为的语义信息没有被考虑在内：用户交互的物品总是有一个原因，在同一段时间，这些被操作的物品可能是互补或者是相近的。当我们没有项目的品牌信息的时候，项目之间的相关关系可以被当作语义信息被训练出来，从而提高效果

**我们的方法怎么解决现有方法的问题**

- 对各种操作行为没有设立任何强度的假设
- 为了捕获不同行为的各种强度，
  - 我们设计了item到user的传播层，不同行为用不同的网络传播。目的是分别抓取不同行为的强度信息。
  - 我们设计了item到item的传播层，目的在于抓取相关的或互补的物品的信息。
