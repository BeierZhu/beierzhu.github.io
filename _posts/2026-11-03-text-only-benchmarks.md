---
layout: post
title: "大模型测评基准梳理I：纯文本基准"
date: 2026-11-03
permalink: /blog/2026/text-only-benchmarks/
description: "按任务定位、数据规模与评测指标梳理纯文本基准，本次先以 Pile 为例。"
tags: [大模型, 测评基准]
categories: [blog]
related_posts: false
_styles: |
  .benchmark-preview-table table { min-width: 680px; margin-bottom: 0; }
  .benchmark-preview-table th { white-space: nowrap; }
  .benchmark-preview-table td { vertical-align: top; }
  .benchmark-preview-table td:nth-child(2) { min-width: 20rem; }
  .benchmark-preview-table td:not(:nth-child(2)) { white-space: nowrap; }
---

纯文本基准覆盖的能力并不完全相同：有些考察知识与推理，有些则关注模型对文本的预测能力。整理时，除了数据集名称，还需要一起看它的**任务定位、数据规模和评测指标**。

## Table 1：通用推理

<div class="table-responsive benchmark-preview-table" markdown="1">

| 名称 | 简介 | 规模 | 评测指标 | SoTA |
| :--- | :--- | :--- | :--- | :--- |
| **Pile** [[1]](#ref-pile) | 由 **22 个多领域子数据集**组成的大规模英文文本语料库，涵盖学术论文、书籍、代码、网页、法律文档及在线对话等，主要用于评估语言模型的**跨领域语言建模能力**。（2020 年发布） | 825 GB | PPL / BPB | — |

</div>

### 如何理解这两个指标？

- **PPL（Perplexity，困惑度）**：衡量模型预测文本的能力。在相同的分词方式和评测设置下，数值越低，通常意味着模型对文本的预测越好。
- **BPB（Bits Per Byte，每字节比特数）**：按字节归一化的平均编码长度，数值越低越好。相比按 token 计算的指标，它减少了不同分词方式对归一化单位的影响。

> **阅读提示：** Pile 主要考察跨领域语言建模能力，并不是直接用答题正确率衡量的推理题集。表中的“—”表示原笔记未提供对应的 SoTA 数值。

### 参考文献

<p id="ref-pile">[1] Gao et al. <a href="https://arxiv.org/abs/2101.00027">The Pile: An 800GB Dataset of Diverse Text for Language Modeling</a>. arXiv:2101.00027.</p>
