from text2vec import SentenceModel


from YChain.KnowledgeBase.text import *
from YChain.YChain import *
from YChain.LLM import GLM


name = 'chatglm'
model_name_or_path = '/workspace/models/chatglm3/'
text2vec_model_path = '/workspace/models/text2vec'

max_token = 2048
history_len = 1024
defult_temperature = 0.95
top_p = 0.9



embedding_vectors = []


# 初始化模型
llm = GLM(name,max_token,history_len,defult_temperature,top_p)
llm.load_model(model_name_or_path=model_name_or_path,device="cuda",quant='half')
print('==LLM加载成功')
text2vec_model = SentenceModel(text2vec_model_path)
print('==T2V模型加载成功')

# 

prompt_template = """【任务说明】:你现在的任务是根据相关知识正确回答问题，加油！
    
    这是可能相关的知识：
    {text}

    这是我的问题：
    {prompt}

    请根据相关知识进行回答 ->
    """
# get_prompt()

print('==知识库加载成功')
data_texts = """
为了更好地控制预测输出三元组的元素顺序，我们引入了一种基于元素顺序的提示学习方法。具体而言，我们在生成模型输入之前，通过设计好的目标三元组元素顺序，采用该顺序设置不同的提示模板进行输入。这种方法能够帮助我们更精准地预测和控制生成模型的输出结果。
设置目标元素顺序
为了更好地引导不同元素的生成，我们对元素标记进行了结构化处理。具体来说，我们将属性词、属性观点和属性情感极性分别标记为[A]、[O]、[S]。这三种不同的标记可以进行排列组合，从而产生6种不同的目标元素生成序列。例如，“[A] 属性词[O] 属性观点[S] 属性情感极性”、“[O] 属性观点[S] 属性情感极性[A] 属性词”等等。如果一个输入句子包含多个情感元组信息，我们会用[SEEP]进行连接，将不同的元组连接起来，形成最终的目标序列。这种结构化处理方式有助于我们更准确地引导生成模型输出所需的元素顺序。
设置目标元素顺序
根据设置好的目标元素顺序序列，我们在输入句子的前端添加了提示语句“这是一段有关数据集评价的句子。”。在句子的尾部，根据目标元素的顺序，我们通过文字提示的形式，构建了相应的语句，并按目标元素顺序进行组合。最终，将前端提示、原始输入句子和尾部提示三个部分连接起来，形成新的输入句子。
原输入句子：数据集的时效性很好。
处理后的句子：这是一段有关数据集评价的句子。数据集的时效性很好。这是一个属性词[A]，这是一个情感词[O]，这是一个情感倾向词[S]。
输出：[A]时效性[O]很好[S]积极的
通过构建不同元素顺序提示的模板，能够有效地引导模型生成元素内容，尤其在样本数量不充足的情况下，能够显著提升模型对三元组信息抽取的效果。
 生成模型
在研究中，我们着重设计了不同的输入模板，以适应目标元素的特定顺序，并通过这些模板重新组合每个句子。这种有序的输入被用于微调生成模型，以获得更符合预期的输出。相较于传统的基于规则或监督学习的方法，生成模型在处理复杂语境和多义性方面通常表现更为出色。在我们的研究中，我们选择了T5（Text-To-Text Transfer Transformer）作为预训练语言模型，用于生成文本。T5的生成能力可以通过公式1表示：
                              P(y|x)=∏_(t=1)^T▒P(y_t |y_1,…,y_(t-1),x)                                      (1)
其中，y表示生成的文本序列，x表示输入序列，T表示生成序列的长度。
T5以其强大的生成能力在属性级情感分析任务中表现卓越。其有效确保生成结果的准确性和流畅性，并能够更好地引导模型生成符合预期目标元素顺序的文本。通过利用生成式模型产生多样化输出的优势，我们在微调后的生成模型上采用随机采样和束搜索技术，生成多个备选三元组序列。这为下一阶段的筛选提供了多样性和灵活性，使得我们能够更精准地选择最合理的三元组。这一方法不仅提高了生成模型的性能，而且为任务的后续处理阶段奠定了坚实的基础。
 投票筛选
对多个备选三元组序列进行处理，从三元组序列中将预测的元素信息，通过正则表达式提取出来，并重新组合成新的句子。如图1所示，将三元组序列中的属性词、属性观点和属性情感极性提取出来，填充到设置好的模板“它是[S],因为它的[A][O]”当中，从而组合成新的句子。将原始输入句子与新组合的句子输入到 Sentence-Bert模型[27]中，对每个句子通过BERT[28]编码转化成语义向量，并通过平均池化的方式获取句子的语义表示。然后，将两个句子语义表示进行相似度比较，相似度计算公式如2所示，并把相似度计算结果进行保存。对所有生成的三元组序列都进行上述操作，最终选择相似度结果最高的三元组序列作为最终的输出结果。
Sim_score= Cosine_sim(D,T)                                        (2)
其中，D表示原始输入句子语义信息，T表示新组合的句子语义信息。
"""
chunk_size = 200  # 切片大小
window_size = 400 # 窗口大小
step_size = 200   # 步幅大小
# 两种切片方式
data_list = text_splitter(data_texts,chunk_size=chunk_size)
data_sliding_list = text_splitter_sliding_window(data_texts, window_size=window_size, step_size=step_size)
for i in data_sliding_list:
    print(len(i))
embedding_vectors = text_2_vec(text2vec_model,data_sliding_list,batch_size=1)
print('==知识库向量化成功')

# prompt = '这是我们的问题 * 1'
history = []
show_input = False
while 1:
    prompt = input('请输入：')
    index,response,history = chat_with_data(prompt,embedding_vectors,history,prompt_template,data_sliding_list,text2vec_model,llm,show_input)
    print('数据来源',index,'/',len(data_sliding_list))
    print('回答',response)
    