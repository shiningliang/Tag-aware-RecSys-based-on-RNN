# Tag-aware-RecSys-based-on-RNN
# 基于深度学习的标签系统推荐算法
本算法为基于深度学习的推荐算法的一次尝试，在标签系统中传统算法基于主题的模型，如CTR通过 文档-主题和 主题-词对文本进行建模，忽略了用户使用标签的变化，而基于的时间模型不能很好地处理标签语义信息。针对以上两个问题，本算法尝试通过使用RNN一方面通过词序发现用户的兴趣变化趋势，另一方面通过基于word2vec训练的词向量，挖掘词义信息。
## 当前实现的内容
- **数据预处理**：筛选使用标签次数不少于5次的user和item，然后保留评分次数不少于5的用户。
- **标签预训练**：训练wiki2vec模型中有如下代码
``` python
model = Word2Vec(LineSentence(inp), size=100, window=5, min_count=2, workers=multiprocessing.cpu_count())
```
其中min_count是词在语料库中最少出现的次数，window为上下文滑动窗口范围。实验时令min_count=2，则两个数据集中出现少于2次的词为4804/8925和15510/34433
- **标签压缩**：MAX_SEQ_LEN=100，对最近MAX_SEQ_LEN之前的标签向量求算术平均，得到compressed tag
- **RNN**：user和item分别使用单层GRU处理标签，提取特征向量，后接MLP进行特征非线性组合
- **NMF**：与GMF相同，直接BP调节embedding layer，得到latent vector，线性特征
## 当前存在的问题
- [ ] words embedding生成方式
- [ ] 算术平均压缩标签向量丢失大量信息
- [ ] 模型表达能力不强
## 改进计划与实验反馈
1. ~~若一个标签由m个词组成，计算这m个词之间的cosine相似度，则对于每个词有m-1维的相似度向量，对这个m*(m-1)的相似度矩阵做maxpooling，产生的m维向量即为这m个词的words embedding，且与原标签中的词处于不同的语义空间。~~  
该方法最后生成m维向量，与原始的embedidng dim不同
2. 给标签分配age，按时间衰减代替算术平均
3. 用CNN处理标签向量，生成feature map输入GRU
4. 双层GRU，加入L2，改用mAP和NDCG，80-10-10数据分配
5. 额外加入用户评分时序信息
