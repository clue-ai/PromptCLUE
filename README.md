<p align="center">
<br>
<br>
<br>
<a href="https://clueai.cn">
   <img src="docs/imgs/logo.png"  alt="CLUEAI logo" width="72%" height="72%" />   

<br>
<br>
<br>
<b>整合全球中文信息，通过人工智能服务， 使人人皆可访问并从中受益</b>
</p>

 <img src="docs/imgs/promptclue1080_1.gif"  alt="CLUEAI logo"  />  
 <img src="docs/imgs/promptclue1080_2.gif"  alt="CLUEAI logo" />  

# PromptCLUE
PromptCLUE：大规模多任务Prompt预训练中文开源模型。

中文上的三大统一：统一模型框架，统一任务形式，统一应用方式。

支持几十个不同类型的任务，具有较好的零样本学习能力和少样本学习能力。针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对生成任务，可以进行采样自由生成。

千亿中文token上大规模预训练，亿级中文任务数据上完成训练，训练任务超过150+。比base版平均任务提升7个点+；具有更好的理解、生成和抽取能力，并且支持文本改写、纠错、知识图谱问答。

统一模型框架：采用Text-to-Text的生成式预训练模型进行统一建模。

统一任务形式：Prompt统一不同的NLP任务间的差异，转化为统一的text-to-text数据形式。

统一应用方式：对目标任务形成拿来即用的模型，下游应用时都可转化为统一的prompt自适应方式，进行zero-shot/few-shot测试。


### 效果对比--16类中文任务

|  任务类型  | PromptCLUE-base  | PromptCLUE-large    | 
| :----:| :----: | :----: | 
|  **分数** Score  | 63.47  | 70.55(+7.08)   | 
|   参数 Parameters  | 220M |  770M   |  
| **理解任务**（acc，10类） |  | | 
| 分类 classify | 89.56 | 92.89| 
| 情感分析 emotion_analysis | 80.55 | 85.64 | 
| 相似度计算 similar | 70.94 | 78.47 | 
| 自然语言推理 nli | 78.00 | 86.67 | 
| 指代消解 anaphora_resolution | 30.00 | 64.00| 
| 阅读理解 reading_comprehension | 71.69 | 84.78 | 
| 关键词提取 keywords_extraction | 41.44 | 47.78 | 
| 信息抽取 ner | 63.02 | 70.09 | 
| 知识图谱问答 knowledge_graph  | - | 53.11 |
| 中心词提取 Keyword_extraction | 66.50 |71.50 |  
| **生成任务**（rouge，6类） |  |   | 
| 翻译（英中、中英） nmt | 55.92 | 59.67 | 
| 摘要 summary | 31.71 | 34.48| 
| 问答 qa | 21.18 | 27.05 | 
| 生成（文章、问题生成） | 35.86 | 39.87 | 
| 改写 paraphrase | - | 57.68  | 
| 纠错 correct | - | 93.35  | 


### 技术与训练过程
 1. 三大统一：统一模型框架(text-to-text)，统一任务形式(prompt)，统一应用方式(zero-shot/few-shot)。 (<a href='https://arxiv.org/abs/2110.08207'>T0</a>）

 2. 大规模预训练：在t5-large版基础上，使用数百G中文语料，训练了100万步，累积训练了1.5万亿个中文字词级别token
 3. 大规模任务数据：使用了16种任务类型，数百种任务，累积亿级别任务数据。
 4. 混合预训练：一方面将下游任务作为预训练语料，另一方面将下游任务和预训练语料一起训练，减少任务灾难遗忘以及缩短预训练和下游任务的距离，更好的适应下游任务（<a href='https://arxiv.org/abs/2111.10952'>ExT5</a>）
 5. 混合采样：针对众多数据量差异极大的任务，采用在每个训练batch内对所有的任务进行按照比例采样，根据任务的数据量进行平滑采样，并且同时限制任务数据量采样池的上限。
            平滑采样可以减少任务训练有偏危害，在每一batch内训练可以减少异质任务之间训练负迁移的情况(T5)
 6. 分阶段训练：一方面指在预训练分阶段，涉及训练序列长度的分阶段（128和512），加快预训练速度(Bert)；另一方面，在下游训练分阶段，
     涉及学习率和序列长度的变化以及递减式对下游任务的数据量限制，更好的适应下游的不同任务
 7. 增加语言模型的训练：参考t5.1.1, 除了使用Span Corrpution构建的方式进行无监督训练，同时在使用prefix LM的方式训练，增强生成任务的能力(<a href='https://arxiv.org/abs/1910.10683'>LM adapted</a>) 
 8. 增加对模型的encoder以及decoder的训练：根据下游任务数据分别构建Data_text,Data_target预训练数据语料，是加入到预训练中，分别增强模型的encoder理解能力和
    decoder的生成能力（见<a href='https://arxiv.org/abs/2203.12277'>UIE</a>）
 9. 重新构建模型中文字典：使用sentencepiece上在千亿token上学习并构建模型字典，更加符合中文语言习惯
    
    
### 在线使用
<a href='https://www.cluebenchmarks.com/clueai.html' targe='_blank'>在线demo</a> | <a href='https://huggingface.co/ClueAI/PromptCLUE' targe='_blank'>huggingface下载地址</a> |   <a href='https://colab.research.google.com/drive/1noyBA_JrYO6Lk6cwxsNZ_jdJ-Jtaf82G?usp=sharing#scrollTo=Nk2tSi3vnSN0' targe='_blank'>colab使用示例</a> |  <a href='https://colab.research.google.com/drive/1QIQDWAACkV7-iRrkrk18XrRjEekMhOtv?usp=sharing' targe='_blank'>自定义数据集进行训练</a> |  <a href='https://github.com/CLUEbenchmark/pCLUE' targe='_blank'>prompt中文数据集</a>

### Large版在线申请
<a href='https://docs.qq.com/form/page/DRVFUb1dIZExjcGxM'>在线申请</a>

### License（许可证）
1）PromptCLUE-base可直接下载和使用；

2）PromptCLUE-large版的<a href='https://github.com/clue-ai/PromptCLUE/blob/main/LICENCE'>非商用License</a>

### 使用方法
##### 安装需要的项目和包
    git clone https://github.com/huggingface/transformers.git
    pip install ./transformers
    pip install sentencepiece
##### 加载模型
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("ClueAI/PromptCLUE-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("ClueAI/PromptCLUE-base") 
##### 使用模型进行预测
    import torch
    from transformers import AutoTokenizer
    # 修改colab笔记本设置为gpu，推理更快
    device = torch.device('cuda')
    model.to(device)
    def preprocess(text):
      return text.replace("\n", "_")
    def postprocess(text):
      return text.replace("_", "\n")
    def answer(text, sample=False, top_p=0.6):
      '''sample：是否抽样。生成任务，可以设置为True;
         top_p：0-1之间，生成的内容越多样、
      '''
      text = preprocess(text)
      encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device) 
      if not sample: # 不进行采样
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4, length_penalty=0.6)
      else: # 采样（生成）
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128, do_sample=True, top_p=top_p)
      out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
      return postprocess(out_text[0])  
 
### 支持的任务（部分）
    意图分类 
    新闻分类
    情感分析
    自然语言推理
    阅读理解
    阅读理解-自由式
    摘要
    翻译-中英
    翻译-英中
    通用信息抽取
    简历信息抽取
    医疗信息抽取
    电商客户需求分析
    医疗语义相似度
    问题生成
    指代消解
    关键词抽取
    情感倾向
    根据标题文章生成
    .....

### 使用自定义数据集进行训练-PyTorch实现

* 使用pCLUE数据集进行训练、预测和效果验证, pytorch实现--在线colab
  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QIQDWAACkV7-iRrkrk18XrRjEekMhOtv?usp=sharing)
  
### <a href='https://cluebenchmarks.com/pclue.html'>pCLUE基准上的效果</a>
 
  <img src="https://github.com/CLUEbenchmark/pCLUE/blob/main/resources/imgs/baseline_result.png"  width="90%" height="90%" />   


### 示例输入

    新闻分类(classify)
    Input:
    分类任务：
    折价率过低遭抛售基金泰和跌7.15%，证券时报记者 朱景锋本报讯 由于折价率在大盘封基中处于最低水平，基金泰和昨日遭到投资者大举抛售，跌幅达到7.15%，远超大盘。盘面显示，基金泰和随大盘高开，之后开始震荡走低，午后开始加速下行，几乎没有像样反弹。截至收盘时，在沪深300指数仅下跌2.56%的情况下，基金泰和收盘跌幅高达7.15%，在所有封基中跌幅最大，而昨日多数封基跌幅在2%左右。
    选项：财经，娱乐，时政，股票
    答案：
    
    Model output:
    财经
    -----------------
    意图分类(classify)
    Input:
    意图分类：
    帮我定一个周日上海浦东的房间
    选项：闹钟，文学，酒店，艺术，体育，健康，天气，其他
    答案：
    
    Model output:
    酒店
    -----------------
    情感分析(classify)
    Input:
    情感分析：
    这个看上去还可以，但其实我不喜欢
    选项：积极，消极
    答案：
    
    Model output:
    消极
    -----------------
    推理(generate)
    Input:
    请推理出上下文的关系：
    前提：对不起事情就是这样。
    假设：事情就是这样，不需要道歉。
    选项：中立，蕴涵，矛盾
    答案：
    
    Model output:
    矛盾
    -----------------
    阅读理解(generate)
    Input:
    阅读文章，给出答案：
    段落：
    港汇指数，全称港元实际汇兑指数（Effective Exchange Rate Index for the Hong Kong Dollar）是由香港政府统计处编制的一项指数，以反映港元与香港主要贸易伙伴之货币的名义有效汇率加权平均数的变动情况。加权比重是按1999年至2000年平均贸易模式所制定，但政府并未有公布详细的计算公式。旧港汇指数基准日为2000年1月1日，基数为100点。由2012年1月3日起，新系列港汇指数 (包括15种货币及以2010年1月 = 100) 已取代旧港汇指数系列。港汇指数的作用，主要是用于反映香港的货品及服务的价格相对于其主要贸易伙伴的变动，并通常被视作反映香港价格竞争力的指标。
    问题：港汇指数的加权比重如何制定？
    答案：
    
    Model output:
    按1999年至2000年平均贸易模式所制定
    -----------------
    阅读理解-自由式(generate)
    Input:
    阅读以下对话并回答问题。
    男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。
    问题：女的怎么样？
    选项：正在工作，感冒了，在打电话，要出差。
    答案：
    
    Model output:
    感冒了
    -----------------
    摘要(generate)
    Input:
    为下面的文章生成摘要：
    北京时间9月5日12时52分，四川甘孜藏族自治州泸定县发生6.8级地震。地震发生后，领导高度重视并作出重要指示，要求把抢救生命作为首要任务，全力救援受灾群众，最大限度减少人员伤亡
    答案：
    
    Model output:
    四川甘孜发生6.8级地震
    -----------------
    翻译-中英(generate)
    Input:
    翻译成英文：
    议长去了台湾，中国人民很愤怒。
    答案：
    -----------------
    Model output:
    The secretary went to Taiwan and the Chinese people were angry.
    
    翻译-英中(generate)
    Input:
    翻译成中文：
    This is a dialogue robot that can talk to people.
    答案：
    -----------------
    Model output:
    这是一台可以与人交谈的对话机器人。
    
    通用信息抽取(generate)
    Input:
    信息抽取：
    据新华社电广东省清远市清城区政府昨日对外发布信息称,日前被实名举报涉嫌勒索企业、说“分分钟可以搞垮一间厂”的清城区环保局局长陈柏,已被免去清城区区委委员
    问题：机构名，人名，职位
    答案：
    -----------------
    Model output:
    机构名：新华社，清城区政府，清城区环保局，清城区区委
    人名：陈柏
    职位：局长，区委委员
    
    简历信息抽取(generate)
    Input:
    阅读文本抽取关键信息：
    张玄武2000年出生中国国籍无境外居留权博士学历现任杭州线锁科技技术总监。
    问题：机构，人名，职位，籍贯，专业，国籍，学历，种族
    答案：
    -----------------
    Model output:
    人名：张玄武
    职位：杭州线锁科技技术总监
    国籍：中国国籍
    学历：博士学历
    -----------------
    医疗信息抽取(generate)
    Input:
    从文本中抽取信息：
    患者精神可，饮食可，睡眠可，二便正常。患者通过综合治疗头晕症状较前减轻，患者继续口服改善脑血管及调整血压变化药物。
    问题：症状，治疗，检查，身体部位，疾病
    答案：
    
    Model output:
    症状：头晕
    治疗：改善脑血管及调整血压变化药物
    身体部位：二便
    -----------------
    电商客户需求分析(classify)
    Input:
    电商客户诉求分类：
    收到但不太合身，可以退换吗
    选项：买家咨询商品是否支持花呗付款，买家表示收藏关注店铺，买家咨询退换货规则，买家需要商品推荐
    答案：
    
    Model output:
    买家咨询退换货规则
    -----------------
    医疗语义相似度(classify)
    Input:
    下面句子是否表示了相同的语义：
    文本1：糖尿病腿麻木怎么办？
    文本2：糖尿病怎样控制生活方式
    选项：相似，不相似
    答案：
    
    Model output:
    不相似
    -----------------
    问题生成(generate)
    Input:
    问题生成：
    中新网2022年9月22日电 22日，商务部召开例行新闻发布会，商务部新闻发言人束珏婷表示，今年1-8月，中国实际使用外资1384亿美元，增长20.2%；其中，欧盟对华投资增长123.7%(含通过自由港投资数据)。这充分表明，包括欧盟在内的外国投资者持续看好中国市场，希望继续深化对华投资合作。
    答案：
    
    Model output:
    1、商务部发言人束珏婷表示，中国去年1-8月实际使用外资1384亿美元,增长20.2%，其中欧盟对华投资增长多少？
    2、欧盟对华投资的增长是否达到预期？
    3、欧盟在对华投资合作方面取得什么样的成果？
    -----------------
    指代消解(generate)
    Input:
    指代消解：
    段落：
    少平跟润叶进了她二爸家的院子，润生走过来对他（代词）说：“我到宿舍找了你两回，你到哪里去了？”
    问题：代词“他”指代的是？
    答案：
    
    Model output:
    少平
    -----------------
    关键词抽取(generate)
    Input:
    抽取关键词：
    当地时间21日，美国联邦储备委员会宣布加息75个基点，将联邦基金利率目标区间上调到3.00%至3.25%之间，符合市场预期。这是美联储今年以来第五次加息，也是连续第三次加息，创自1981年以来的最大密集加息幅度。
    关键词：
    
    Model output:
    美联储，利率目标区间，加息，基点
    -----------------
    情感倾向(classify)
    文字中包含了怎样的情感：
    超可爱的帅哥，爱了。。。
    选项：厌恶，喜欢，开心，悲伤，惊讶，生气，害怕
    答案：
    
    Model output:
    喜欢
    -----------------
  
    
### 技术交流和问题反馈
<p float="left">
   <img src="https://github.com/clue-ai/PromptCLUE/blob/main/docs/imgs/promptclue_group.jpeg"  width="29%" height="29%" />   
   <img src="https://github.com/clue-ai/clueai-python/blob/main/docs/imgs/brightmart.jpeg"  width="29%" height="29%" /> 
</p> 


### 相关资料

1. <a href='https://arxiv.org/abs/1910.10683'>t5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</a>
2. <a href='https://github.com/google-research/text-to-text-transfer-transformer'>t5 repo</a>
3. <a href='https://arxiv.org/abs/2110.08207'>T0: Multitask Prompted Training Enables Zero-Shot Task Generalization</a>
4. <a href='https://arxiv.org/abs/2202.01279'>PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts</a>
5. <a href='https://www.163.com/dy/article/H2IN7ES10511CQLG.html'>后Prompt时代 | NLP统一范式：预训练+大规模多任务学习</a>
6. <a href='https://arxiv.org/abs/2201.06910'>ZeroPrompt: Scaling Prompt-Based Pretraining to 1,000 Tasks Improves Zero-Shot Generalization</a>
7. <a href='https://arxiv.org/abs/2111.10952'>ExT5: Towards Extreme Multi-Task Scaling for Transfer Learning
8. <a href='https://arxiv.org/abs/2204.07705'>Natural-Instructions v2: Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks</a>
9. <a href='https://arxiv.org/abs/1910.10683'>LM adapted: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</a>
10. <a href='https://arxiv.org/abs/2203.12277'>UIE: Unified Structure Generation for Universal Information Extraction</a>
11. <a href='https://arxiv.org/abs/2111.10952'>ExT5: Towards Extreme Multi-Task Scaling for Transfer Learning</a>
</a>