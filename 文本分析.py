docs = [
    "财务大数据平台实现了企业凭证、报表与预算的自动采集与整合，显著提高了数据处理效率。",
    "区块链电子发票系统提升了发票流转透明度，降低了税务风险与重复报销问题。",
    "智能审计利用自然语言处理和异常检测算法，实现了凭证级别的智能抽查与风险识别。",
    "ESG报告披露平台通过大数据技术整合环境、社会与治理信息，提升了企业可持续发展水平。",
    "财务共享中心依托云计算与人工智能技术，构建了集中化、标准化的会计核算体系。",
    "数字化财务转型要求财务人员具备数据分析与业务洞察能力，实现财务管理由核算向决策支持转变。"
]
# -*- coding: utf-8 -*-
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 1. 数据准备与分词
docs = [
    "财务大数据平台实现了企业凭证、报表与预算的自动采集与整合，显著提高了数据处理效率。",
    "区块链电子发票系统提升了发票流转透明度，降低了税务风险与重复报销问题。",
    "智能审计利用自然语言处理和异常检测算法，实现了凭证级别的智能抽查与风险识别。",
    "ESG报告披露平台通过大数据技术整合环境、社会与治理信息，提升了企业可持续发展水平。",
    "财务共享中心依托云计算与人工智能技术，构建了集中化、标准化的会计核算体系。",
    "数字化财务转型要求财务人员具备数据分析与业务洞察能力，实现财务管理由核算向决策支持转变。"
]

# 中文分词
seg_docs = [" ".join(list(jieba.cut(doc))) for doc in docs]

# 2. 构建 TF-IDF 矩阵
vectorizer = TfidfVectorizer(max_features=30)
tfidf_matrix = vectorizer.fit_transform(seg_docs)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

print("【TF-IDF 关键词权重表】")
print(tfidf_df.head())

# 3. LDA 主题模型
lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
lda_model.fit(tfidf_matrix)

# 输出每个主题的关键词
def display_topics(model, feature_names, n_top_words=5):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\n主题 {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

print("\n【LDA主题关键词】")
display_topics(lda_model, vectorizer.get_feature_names_out(), 6)
