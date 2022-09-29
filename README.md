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
PromptCLUE：是一个多任务模型，支持100+中文任务，并具有零样本学习能力。
针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对生成任务，可以进行采样自由生成。
基于t5模型，使用1000亿中文token（字词级别）进行大规模预训练，并且在众多下游任务上进行多任务学习获得。

### 在线使用
<a href='https://www.cluebenchmarks.com/clueai.html'>在线demo</a> | <a href='https://huggingface.co/ClueAI/PromptCLUE'>huggingface下载地址</a> |   <a href='https://colab.research.google.com/drive/1noyBA_JrYO6Lk6cwxsNZ_jdJ-Jtaf82G?usp=sharing#scrollTo=Nk2tSi3vnSN0'>colab使用示例</a>

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

### 技术交流和问题反馈
<p float="left">
   <img src="https://github.com/clue-ai/clueai-python/blob/main/docs/imgs/clueai.jpeg"  width="25%" height="25%" />   
   <img src="https://github.com/clue-ai/clueai-python/blob/main/docs/imgs/brightmart.jpeg"  width="25%" height="25%" /> 
</p> 



