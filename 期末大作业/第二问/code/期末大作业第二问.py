# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('Agg')  # 强制使用'Agg'后端

import os
import sys
import time
import json
import csv
import random
import requests
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# --- 新增：获取脚本所在目录，确保文件保存在正确位置 ---
# __file__ 是一个特殊变量，代表当前脚本的文件名
# os.path.dirname() 获取该文件所在的目录
# os.path.abspath() 将其转换为绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"脚本正在运行，所有生成的文件将被保存在这个目录下:\n{SCRIPT_DIR}\n")

# 设置中文字体支持
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']

# 设置NLTK数据下载路径
nltk_data_dir = r"D:\Setup\NLTKData"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# 下载NLTK数据
try:
    # ... (NLTK下载代码保持不变) ...
    print("正在检查并下载NLTK数据包，请稍候...")
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
    print("NLTK数据包准备就绪。")
except Exception as e:
    print(f"NLTK数据下载失败: {e}")


class NewsTopicAnalyzer:
    # ... (__init__, load_data, remove_urls, preprocess_text, preprocess_corpus, build_model, display_topics 等函数保持不变) ...
    def __init__(self):
        self.news_titles = []
        self.processed_texts = []
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['rt', 'http', 'https', 'www', 'com', 'amp', 'co', 'news', 'report'])

    def load_data(self, train_data_path):
        print("正在从训练集加载新闻标题数据...")
        try:
            df = pd.read_csv(train_data_path)
            self.news_titles = df['post_text'].head(10).tolist()
            print(f"成功加载 {len(self.news_titles)} 条新闻标题")
            print("\n新闻标题列表:")
            for i, title in enumerate(self.news_titles, 1):
                print(f"{i}. {title}")
            return True
        except Exception as e:
            print(f"加载训练集数据失败: {e}")
            return False

    def remove_urls(self, text):
        url_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'\S+\.(?:com|org|net|edu|gov|mil|int|co|io|me|ly|it|de|fr|uk|cn|jp|kr|au|ca)\S*',
            r'[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/\S*)?',
            r't\.co/\S+', r'bit\.ly/\S+', r'tinyurl\.com/\S+', r'goo\.gl/\S+',
        ]
        for pattern in url_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text

    def preprocess_text(self, text):
        if pd.isna(text) or not text: return []
        text = str(text).lower()
        text = self.remove_urls(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def preprocess_corpus(self):
        print("\n正在进行文本预处理...")
        self.processed_texts = [self.preprocess_text(title) for title in self.news_titles if
                                self.preprocess_text(title)]
        print(f"预处理完成，保留 {len(self.processed_texts)} 条有效文本")
        print("\n预处理示例:")
        for i, (original, processed) in enumerate(zip(self.news_titles[:3], self.processed_texts[:3])):
            print(f"原文 {i + 1}: {original}\n处理后: {processed}\n" + "-" * 60)

    def build_model(self, num_topics=3, passes=20, alpha='auto', eta='auto'):
        print("\n正在构建模型...")
        print("构建词典...")
        self.dictionary = corpora.Dictionary(self.processed_texts)
        self.dictionary.filter_extremes(no_below=1, no_above=0.8)
        print("生成语料库...")
        self.corpus = [self.dictionary.doc2bow(text) for text in self.processed_texts]
        print(f"词典大小: {len(self.dictionary)}, 语料库大小: {len(self.corpus)}")
        print(f"正在训练LDA模型（主题数: {num_topics}）...")
        self.lda_model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics,
                                  random_state=42, passes=passes, alpha=alpha, eta=eta, per_word_topics=True)
        print("LDA模型训练完成!")
        self.display_topics()

    def display_topics(self):
        print("\n发现的主题:")
        for idx, topic in self.lda_model.print_topics(-1):
            print(f"主题 {idx}: {topic}\n" + "-" * 50)

    def visualize_with_pyldavis(self):
        """
        4. 可视化分析：pyLDAvis交互图 (已修改)
        """
        print("\n正在生成pyLDAvis交互图...")
        try:
            vis_data = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)
            # --- 修改：使用os.path.join确保保存到脚本目录 ---
            save_path = os.path.join(SCRIPT_DIR, 'news_lda_visualization.html')
            pyLDAvis.save_html(vis_data, save_path)
            print(f"✓ pyLDAvis交互图已保存到:\n  {save_path}")
            return vis_data
        except Exception as e:
            print(f"pyLDAvis可视化生成失败: {e}")
            return None

    def generate_wordclouds(self):
        """
        生成词云图 (已修改)
        """
        print("\n正在生成词云图...")
        try:
            num_topics = self.lda_model.num_topics
            cols = 3
            rows = (num_topics + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
            axes = axes.flatten()

            for topic_id in range(num_topics):
                topic_words = dict(self.lda_model.show_topic(topic_id, topn=15))
                wordcloud = WordCloud(width=400, height=300, background_color='white',
                                      colormap='viridis', max_words=15).generate_from_frequencies(topic_words)
                axes[topic_id].imshow(wordcloud, interpolation='bilinear')
                axes[topic_id].set_title(f'主题 {topic_id}', fontsize=14, fontweight='bold')
                axes[topic_id].axis('off')

            for i in range(num_topics, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            # --- 修改：使用os.path.join确保保存到脚本目录 ---
            save_path = os.path.join(SCRIPT_DIR, 'news_topic_wordclouds.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # plt.show() # 在Agg后端下，此行无效且会引发警告，已注释
            print(f"✓ 词云图已保存到:\n  {save_path}")
        except Exception as e:
            print(f"词云图生成失败: {e}")

    def generate_heatmap(self):
        """
        生成热力图 (已修改)
        """
        print("\n正在生成热力图...")
        try:
            doc_topic_matrix = np.array([
                [prob for _, prob in self.lda_model.get_document_topics(doc, minimum_probability=0)]
                for doc in self.corpus
            ])
            plt.figure(figsize=(10, 8))
            sns.heatmap(doc_topic_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                        xticklabels=[f'主题{i}' for i in range(self.lda_model.num_topics)],
                        yticklabels=[f'新闻{i + 1}' for i in range(len(doc_topic_matrix))],
                        cbar_kws={'label': '概率'})
            plt.title('文档-主题概率分布热力图', fontsize=16, fontweight='bold')
            plt.xlabel('主题', fontsize=12)
            plt.ylabel('新闻文档', fontsize=12)
            plt.tight_layout()
            # --- 修改：使用os.path.join确保保存到脚本目录 ---
            save_path = os.path.join(SCRIPT_DIR, 'news_document_topic_heatmap.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # plt.show() # 在Agg后端下，此行无效且会引发警告，已注释
            print(f"✓ 热力图已保存到:\n  {save_path}")
        except Exception as e:
            print(f"热力图生成失败: {e}")

    # ... (analyze_with_llm, call_llm_model, basic_topic_analysis, get_topic_distribution_for_documents, run_complete_analysis 等函数保持不变) ...
    def analyze_with_llm(self):
        print("\n正在使用大模型分析主题内容...")
        topics_analysis = []
        for topic_id in range(self.lda_model.num_topics):
            topic_words = [word for word, _ in self.lda_model.show_topic(topic_id, topn=10)]
            topic_words_str = ", ".join(topic_words)
            prompt = f"""
            请分析以下新闻主题的关键词，并给出深入的主题分析：
            主题关键词: {topic_words_str}
            请从以下几个方面进行分析：
            1. 这个主题的核心内容是什么？ 2. 这个主题涉及哪些具体领域或事件类型？
            3. 主题的情感倾向（积极、消极、中性）和重要性 4. 给这个主题起一个简洁明了的名称
            5. 这类新闻对社会的潜在影响
            请用中文回答，分析要深入且简洁明了。
            """
            try:
                analysis = self.call_llm_model(prompt)
                topics_analysis.append({'topic_id': topic_id, 'keywords': topic_words_str, 'analysis': analysis})
                print(
                    f"\n=== 主题 {topic_id} 深度分析 ===\n关键词: {topic_words_str}\n大模型分析:\n{analysis}\n" + "=" * 80)
            except Exception as e:
                print(f"主题 {topic_id} 分析失败: {e}")
                basic_analysis = self.basic_topic_analysis(topic_words)
                topics_analysis.append({'topic_id': topic_id, 'keywords': topic_words_str, 'analysis': basic_analysis})
        return topics_analysis

    def call_llm_model(self, prompt):
        url = "http://localhost:11434/api/generate"
        data = {"model": "gemma3", "prompt": prompt, "stream": False}
        try:
            response = requests.post(url, json=data, timeout=120)
            response.raise_for_status()
            return response.json().get('response', '模型响应为空')
        except requests.exceptions.RequestException as e:
            print(f"大模型API调用失败: {e}")
            raise e

    def basic_topic_analysis(self, topic_words):
        keywords_str = ", ".join(topic_words)
        analysis = f"基于关键词 '{keywords_str}' 的基础分析：\n"
        if any(word in topic_words for word in ['government', 'policy', 'announce']):
            analysis += "1. 核心内容：政府政策相关主题\n"
        elif any(word in topic_words for word in ['technology', 'ai', 'cyber']):
            analysis += "1. 核心内容：科技创新相关主题\n"
        elif any(word in topic_words for word in ['market', 'economic', 'stock']):
            analysis += "1. 核心内容：经济金融相关主题\n"
        elif any(word in topic_words for word in ['health', 'worker', 'education']):
            analysis += "1. 核心内容：社会民生相关主题\n"
        else:
            analysis += "1. 核心内容：综合性新闻主题\n"
        analysis += "2. 涉及领域：新闻报道的重要社会议题\n"
        analysis += "3. 情感倾向：中性，具有新闻价值\n"
        analysis += f"4. 主题名称：{topic_words[0].title()}相关新闻\n"
        analysis += "5. 社会影响：对公众具有信息价值的重要事件"
        return analysis

    def get_topic_distribution_for_documents(self):
        print("\n=== 新闻文档主题分布统计 ===")
        topic_counts = {i: 0 for i in range(self.lda_model.num_topics)}
        for i, doc in enumerate(self.corpus):
            topic_dist = self.lda_model.get_document_topics(doc)
            if topic_dist:
                dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
                topic_counts[dominant_topic] += 1
                print(f"新闻 {i + 1}: 主导主题 {dominant_topic}")
        print("\n各主题的新闻数量分布:")
        for topic_id, count in topic_counts.items():
            percentage = (count / len(self.corpus)) * 100
            print(f"主题 {topic_id}: {count} 篇新闻 ({percentage:.1f}%)")

    def run_complete_analysis(self, num_topics=3, train_data_path=None):
        print("=" * 80 + "\n开始基于大模型的新闻主题分析...\n" + "=" * 80)
        if not (train_data_path and self.load_data(train_data_path)):
            print("未提供或加载训练集失败，分析终止");
            return
        self.preprocess_corpus()
        if not self.processed_texts:
            print("没有有效的文本数据，分析终止");
            return
        self.build_model(num_topics=num_topics)
        print("\n" + "=" * 50 + "\n开始可视化分析...\n" + "=" * 50)
        self.visualize_with_pyldavis()
        self.generate_wordclouds()
        self.generate_heatmap()
        self.get_topic_distribution_for_documents()
        print("\n" + "=" * 50 + "\n开始大模型深度分析...\n" + "=" * 50)
        self.analyze_with_llm()
        print("\n" + "=" * 80 + "\n=== 分析完成 ===\n" + "=" * 80)
        print(
            "生成的文件:\n✓ news_lda_visualization.html\n✓ news_topic_wordclouds.png\n✓ news_document_topic_heatmap.png")
        print("\n所有功能已成功实现！")


if __name__ == "__main__":
    analyzer = NewsTopicAnalyzer()
    train_data_path = "D:\\大三下\\云计算与大数据分析\\大作业\\期末大作业\\第二问\\data\\训练集.csv"
    analyzer.run_complete_analysis(num_topics=3, train_data_path=train_data_path)
