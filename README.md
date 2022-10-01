<p align="center">
<br>
<br>
<br>
<a href="https://clueai.cn"><img src="docs/imgs/logo.png" alt="CLUEAI logo: The data structure for unstructured data" width="200px"></>
<br>
<br>
<br>
<b>整合全球中文信息，通过人工智能服务， 使人人皆可访问并从中受益</b>
</p>


# PromptCLUE
PromptCLUE：多任务中文预训练模型

### 简介
PromptCLUE：是一个多任务中文模型，支持众多中文任务，并具有零样本学习能力。
针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对生成任务，可以进行采样自由生成。
基于t5模型，使用1000亿中文token（字词级别）进行大规模预训练，并且在100+任务上进行多任务学习获得。

### 在线使用
<a href='https://www.cluebenchmarks.com/clueai.html' targe='_blank'>在线demo</a> | <a href='https://huggingface.co/ClueAI/PromptCLUE' targe='_blank'>huggingface下载地址</a> |   <a href='https://colab.research.google.com/drive/1noyBA_JrYO6Lk6cwxsNZ_jdJ-Jtaf82G?usp=sharing#scrollTo=Nk2tSi3vnSN0' targe='_blank'>colab使用示例</a> |  <a href='https://github.com/CLUEbenchmark/pCLUE' targe='_blank'>prompt中文数据集</a>

### 使用方法
##### 安装需要的项目和包
    git clone https://github.com/huggingface/transformers.git
    pip install ./transformers
    pip install -U nlp
    pip install datasets
    pip install sentencepiece
    pip install gcsfs
##### 加载模型
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("ClueAI/PromptCLUE")
    model = AutoModelForSeq2SeqLM.from_pretrained("ClueAI/PromptCLUE") 
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
   <img src="https://github.com/clue-ai/clueai-python/blob/main/docs/imgs/clueai1.jpeg"  width="29%" height="29%" />   
   <img src="https://github.com/clue-ai/clueai-python/blob/main/docs/imgs/brightmart.jpeg"  width="29%" height="29%" /> 
</p> 



