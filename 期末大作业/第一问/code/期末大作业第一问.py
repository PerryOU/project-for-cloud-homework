# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import csv
import random
import requests
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# 步骤 0: 辅助与配置函数
# ==============================================================================

def configure_matplotlib_for_chinese():
    """配置Matplotlib以支持中文显示"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("Matplotlib中文环境配置成功。")
    except Exception:
        print("Matplotlib中文环境配置失败，图表中的中文可能无法正常显示。请确保已安装'SimHei'字体。")


def get_true_labels(file_path, sample_ratio=1.0):
    """读取验证集的真实标签，支持数据采样"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            all_data = [{'content': row[1], 'label': 1 if row[6].lower() == 'real' else 0}
                        for row in reader if len(row) >= 7]

            sample_size = int(len(all_data) * sample_ratio)
            if sample_size > len(all_data):
                sample_size = len(all_data)

            sampled_data = random.sample(all_data, sample_size)

            print(f"从 {file_path} 加载数据成功。")
            print(f"总数据量：{len(all_data)}条，采样数据量：{sample_size}条 ({sample_ratio * 100:.1f}%)")
            return sampled_data
    except Exception as e:
        print(f"读取真实标签时出错：{e}")
        return []


def call_ollama_model(prompt, model="gemma3"):
    """通用Ollama API调用函数，包含重试逻辑"""
    for attempt in range(3):
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get('response', '')
            time.sleep(1)
        except Exception as e:
            print(f"调用API出错 (尝试 {attempt + 1}/3): {e}")
            time.sleep(1)
    return ""


# ==============================================================================
# 步骤 1, 2: 模型调用函数
# ==============================================================================

def get_basic_prediction(content):
    """(1) 使用prompt语句，调用大模型判别每条新闻是真新闻还是假新闻"""
    print("  - [步骤1] 正在进行基础真伪判别...")
    prompt = f"判断这条新闻是真(1)还是假(0)，只回答数字1或0：{content[:300]}"
    response_text = call_ollama_model(prompt)
    for char in response_text:
        if char in ['0', '1']:
            print(f"    模型预测: {char}")
            return int(char)
    print("    模型未返回有效结果，默认预测为 1 (真)")
    return 1


def get_sentiment_analysis(content):
    """(2) 使用prompt语句，调用大模型分析文档的语义情感"""
    print("  - [步骤2] 正在进行情感语义分析...")
    prompt = f"分析以下文本的情感倾向。只回答：积极、消极或中性。\n文本内容：{content[:300]}"
    response_text = call_ollama_model(prompt).lower().strip()
    print(f"    模型响应: {response_text[:50]}...")
    if '积极' in response_text: return '积极'
    if '消极' in response_text: return '消极'
    return "中性"


# ==============================================================================
# 步骤 3: 结合情感进行判别 (已更新为方案一Prompt)
# ==============================================================================

def get_prediction_with_sentiment(content, sentiment):
    """(3) 加上情感分析，设计prompt语句，判别每条新闻是真新闻还是假新闻"""
    print("  - [步骤3] 正在结合情感进行真伪判别...")

    # --- 已更新为方案一: 角色扮演与提示引导 ---
    prompt = f"""请扮演一个专业的新闻事实核查员。
你的任务是基于以下信息，判断新闻的真伪。请特别注意，煽动性强或极端情绪化的新闻，有更高的可能性是虚假信息。

- 新闻文本："{content[:300]}..."
- 已分析出的情感倾向：{sentiment}

请综合判断，这条新闻是真是假？你的回答只能是一个数字：1 (代表真) 或 0 (代表假)。"""

    response_text = call_ollama_model(prompt)
    for char in response_text:
        if char in ['0', '1']:
            print(f"    模型预测: {char}")
            return int(char)
    print("    模型未返回有效结果，默认预测为 1 (真)")
    return 1


# ==============================================================================
# 步骤 4: 结果分析与可视化
# ==============================================================================

def analyze_and_report_results(results):
    """(4) 分析准确率是否有提升，并生成与各步骤对应的可视化图表"""
    print("\n==================================================")
    print("步骤 4: 分析与可视化")
    print("==================================================")

    if not results:
        print("没有可供分析的结果。")
        return

    # --- 数据准备 ---
    true_labels = [{'label': r['true_label']} for r in results]
    basic_predictions = [{'predicted': r['basic_prediction']} for r in results]
    sentiment_predictions = [{'predicted': r['sentiment_prediction']} for r in results]

    # --- 准确率计算 ---
    basic_metrics = calculate_accuracy_metrics(basic_predictions, true_labels)
    sentiment_metrics = calculate_accuracy_metrics(sentiment_predictions, true_labels)
    accuracy_improvement = sentiment_metrics['overall_accuracy'] - basic_metrics['overall_accuracy']

    # --- 文本报告输出 ---
    print("\n--- 最终分析报告 ---\n")
    print("【要求(1) & (3) & (4): 准确率统计与提升分析】\n")
    print(f"  - [步骤1] 基础方法准确率: {basic_metrics['overall_accuracy']:.2f}%")
    print(f"  - [步骤3] 结合情感分析准确率: {sentiment_metrics['overall_accuracy']:.2f}%")
    if accuracy_improvement > 0:
        print(f"  - [步骤4] 结论: 准确率提升了 {accuracy_improvement:.2f}%\n")
    else:
        print(f"  - [步骤4] 结论: 准确率未提升 (变化: {accuracy_improvement:.2f}%)\n")

    # --- 可视化图表生成 ---
    print("--- 生成对应的可视化图表 ---\n")

    # (对应要求1)
    print("  -> 正在生成【要求(1)】的可视化图表: 单独显示基础准确率...")
    plot_single_accuracy(basic_metrics,
                         title='基础方法准确率 (对应要求1)',
                         filename='(图表1)基础方法准确率图.png',
                         color='#42A5F5')

    # (对应要求2)
    print("  -> 正在生成【要求(2)】的可视化图表: 情感分析结果...")
    sentiment_stats = {
        '积极': sum(1 for r in results if r['sentiment'] == '积极'),
        '消极': sum(1 for r in results if r['sentiment'] == '消极'),
        '中性': sum(1 for r in results if r['sentiment'] == '中性')
    }
    plot_sentiment_distribution(sentiment_stats, filename='(图表2)情感分布图.png')

    # (对应要求3)
    print("  -> 正在生成【要求(3)】的可视化图表: 单独显示结合情感分析后的准确率...")
    plot_single_accuracy(sentiment_metrics,
                         title='结合情感分析准确率 (对应要求3)',
                         filename='(图表3)结合情感分析准确率图.png',
                         color='#FFA726')

    # (对应要求4)
    print("  -> 正在生成【要求(4)】的可视化图表: 准确率提升对比分析...")
    plot_accuracy_comparison(basic_metrics, sentiment_metrics, filename='(图表4)准确率对比图.png')

    # (额外补充分析图)
    print("  -> 正在生成【补充分析】的可视化图表: 情感与新闻真伪的关联...")
    plot_sentiment_vs_truth(results, filename='(补充图)情感与真伪关联图.png')

    print("\n分析流程结束。所有图表已保存至脚本所在目录。")


def calculate_accuracy_metrics(predictions, true_labels):
    """计算各项准确率指标"""
    total, total_correct, true_correct, fake_correct, total_true, total_fake = 0, 0, 0, 0, 0, 0
    for pred, true in zip(predictions, true_labels):
        predicted_label, true_label = pred['predicted'], true['label']
        total += 1
        if true_label == 1:
            total_true += 1
        else:
            total_fake += 1
        if predicted_label == true_label:
            total_correct += 1
            if true_label == 1:
                true_correct += 1
            else:
                fake_correct += 1
    return {
        'overall_accuracy': (total_correct / total * 100) if total > 0 else 0,
        'true_accuracy': (true_correct / total_true * 100) if total_true > 0 else 0,
        'fake_accuracy': (fake_correct / total_fake * 100) if total_fake > 0 else 0
    }


# --- 可视化绘图函数 ---

def plot_single_accuracy(metrics, title, filename, color):
    """绘制单种方法的准确率条形图"""
    labels = ['整体准确率', '真新闻准确率', '假新闻准确率']
    accuracies = [
        metrics['overall_accuracy'],
        metrics['true_accuracy'],
        metrics['fake_accuracy']
    ]
    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels, accuracies, width=0.5, color=color)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.title(title, fontsize=16)
    plt.ylim(0, 110)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"     '{filename}' 已保存。")


def plot_sentiment_distribution(sentiment_stats, filename):
    labels, counts = list(sentiment_stats.keys()), list(sentiment_stats.values())
    plt.figure(figsize=(8, 6));
    bars = plt.bar(labels, counts, color=['#4CAF50', '#F44336', '#FFC107'])
    plt.title('新闻情感倾向分布 (对应要求2)', fontsize=16);
    plt.xlabel('情感倾向');
    plt.ylabel('新闻数量')
    for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval),
                                                       va='bottom', ha='center')
    plt.savefig(filename);
    plt.close();
    print(f"     '{filename}' 已保存。")


def plot_accuracy_comparison(basic_metrics, sentiment_metrics, filename):
    labels = ['整体准确率', '真新闻准确率', '假新闻准确率'];
    width = 0.35
    basic_acc = [basic_metrics['overall_accuracy'], basic_metrics['true_accuracy'], basic_metrics['fake_accuracy']]
    sentiment_acc = [sentiment_metrics['overall_accuracy'], sentiment_metrics['true_accuracy'],
                     sentiment_metrics['fake_accuracy']]
    x = np.arange(len(labels));
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, basic_acc, width, label='基础方法', color='#42A5F5')
    rects2 = ax.bar(x + width / 2, sentiment_acc, width, label='结合情感分析', color='#FFA726')
    ax.set_ylabel('准确率 (%)');
    ax.set_title('准确率提升对比分析 (对应要求4)', fontsize=16)
    ax.set_xticks(x);
    ax.set_xticklabels(labels);
    ax.legend();
    ax.set_ylim(0, 110)

    def autolabel(rects):
        for rect in rects: height = rect.get_height(); ax.annotate(f'{height:.2f}%',
                                                                   xy=(rect.get_x() + rect.get_width() / 2, height),
                                                                   xytext=(0, 3), textcoords="offset points",
                                                                   ha='center', va='bottom')

    autolabel(rects1);
    autolabel(rects2);
    fig.tight_layout();
    plt.savefig(filename);
    plt.close();
    print(f"     '{filename}' 已保存。")


def plot_sentiment_vs_truth(results, filename):
    counts = {'积极': {'real': 0, 'fake': 0}, '消极': {'real': 0, 'fake': 0}, '中性': {'real': 0, 'fake': 0}}
    for r in results:
        if r['sentiment'] in counts: counts[r['sentiment']]['real' if r['true_label'] == 1 else 'fake'] += 1
    labels = list(counts.keys());
    real_counts = [counts[s]['real'] for s in labels];
    fake_counts = [counts[s]['fake'] for s in labels]
    x = np.arange(len(labels));
    width = 0.35;
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, real_counts, width, label='真新闻', color='#66BB6A')
    rects2 = ax.bar(x + width / 2, fake_counts, width, label='假新闻', color='#EF5350')
    ax.set_ylabel('新闻数量');
    ax.set_title('补充分析：不同情感下的新闻真伪分布', fontsize=16)
    ax.set_xticks(x);
    ax.set_xticklabels(labels);
    ax.legend()

    def autolabel(rects):
        for rect in rects: height = rect.get_height(); ax.annotate(f'{height}',
                                                                   xy=(rect.get_x() + rect.get_width() / 2, height),
                                                                   xytext=(0, 3), textcoords="offset points",
                                                                   ha='center', va='bottom')

    autolabel(rects1);
    autolabel(rects2);
    fig.tight_layout();
    plt.savefig(filename);
    plt.close();
    print(f"     '{filename}' 已保存。")


# ==============================================================================
# 主执行函数 (Main)
# ==============================================================================

def main():
    configure_matplotlib_for_chinese()

    validation_path = r"D:\大三下\云计算与大数据分析\大作业\期末大作业\data\测试集.csv"
    file_path = sys.argv[1] if len(sys.argv) > 1 else validation_path
    sample_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

    try:
        true_labels = get_true_labels(file_path, sample_ratio)
        if not true_labels:
            print("未能加载数据，程序终止。")
            return

        results = []

        print(f"\n开始处理 {len(true_labels)} 条新闻...")

        for i, item in enumerate(true_labels, 1):
            content, true_label = item['content'], item['label']
            print(f"\n--- 正在处理第 {i}/{len(true_labels)} 条新闻 ---")
            print(f"真实标签: {'真新闻' if true_label == 1 else '假新闻'}")

            basic_prediction = get_basic_prediction(content)
            sentiment = get_sentiment_analysis(content)
            sentiment_prediction = get_prediction_with_sentiment(content, sentiment)

            results.append({
                'content': content, 'true_label': true_label,
                'basic_prediction': basic_prediction, 'sentiment': sentiment,
                'sentiment_prediction': sentiment_prediction
            })

        analyze_and_report_results(results)

    except Exception as e:
        print(f"\n程序执行过程中发生严重错误：{e}")
        return


if __name__ == "__main__":
    main()
