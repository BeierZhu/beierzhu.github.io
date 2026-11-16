---
layout: post
title: "大模型测评III：音频与全模态基准"
date: 2026-11-16
permalink: /blog/2026/audio-omni-benchmarks/
description: "梳理主流音频与全模态大模型测评基准，便于查阅参考。"
tags: [大模型, 测评基准]
categories: [blog]
related_posts: false
toc:
  beginning: true
_styles: |
  .benchmark-preview-table table { width: 100%; min-width: 800px; table-layout: fixed; margin-bottom: 0; }
  .benchmark-preview-table thead { background-image: linear-gradient(var(--global-divider-color), var(--global-divider-color)), linear-gradient(var(--global-divider-color), var(--global-divider-color)); background-size: 100% 1px; background-position: top, bottom; background-repeat: no-repeat; }
  .benchmark-preview-table th { white-space: nowrap; text-align: left; vertical-align: middle; padding: .5rem 0; border: 0; }
  .benchmark-preview-table td { vertical-align: top; }
  .benchmark-preview-table tbody tr { background-image: linear-gradient(var(--global-divider-color), var(--global-divider-color)); background-size: 100% 1px; background-position: bottom; background-repeat: no-repeat; }
  .benchmark-data-row td { padding: .4rem 0; vertical-align: top; }
  .benchmark-citation-toggle { display: inline; padding: 0; border: 0; background: none; color: var(--global-theme-color); font: inherit; text-align: left; cursor: pointer; }
  .benchmark-citation-row[hidden] { display: none; }
  .benchmark-citation-row td { padding: .75rem; white-space: normal; overflow-wrap: anywhere; font-size: .85rem; line-height: 1.6; }
---

<br>
## 1. 音频基准

<br>
## 2. 全模态基准

### 2.1 情感理解

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="全模态情感理解基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 51%">
      <col style="width: 15%">
      <col style="width: 12%">
    </colgroup>
    <thead>
      <tr>
        <th scope="col">名称</th>
        <th scope="col">发布时间</th>
        <th scope="col">简介</th>
        <th scope="col">规模</th>
        <th scope="col">评测指标</th>
      </tr>
    </thead>
    <tbody>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mer2023-citation" title="展开或收起引用">MER 2023</button></td>
        <td>2023 年</td>
        <td>多模态视频情绪识别基准，包含三个评测赛道：MER-MULTI 联合预测离散情绪类别与连续情绪效价（Valence）；MER-NOISE 评估噪声干扰下的模态鲁棒性；MER-SEMI 利用大规模未标注视频进行半监督情绪识别。</td>
        <td>视频片段：78.2K<br>平均时长：3.3 秒</td>
        <td>Weighted F1 / Combined Score</td>
      </tr>
      <tr id="mer2023-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Lian et al. <em>MER 2023: Multi-label Learning, Modality Robustness, and Semi-Supervised Learning.</em> ACM MM, 2023.
          <a href="https://arxiv.org/abs/2304.08981">论文链接</a>。
          <p><b>数据集构成：</b>基于 CHEAVD 扩展构建，包含 Train&Val（3,373）、MER-MULTI（411）、MER-NOISE（412）和 MER-SEMI（834 个标注样本及 73,148 个未标注样本），共计 78,178 个视频片段。平均时长约 3.3 秒，视频主要来源于中文电影和电视剧中的人物情绪表达片段。</p>
          <p><b>标注形式：</b>提供离散与维度情绪标注。离散情绪包括 Neutral、Anger、Happiness、Sadness、Worry 和 Surprise 六类；维度情绪标注为 Valence 连续分数，表示情绪从消极到积极的程度，并非完整的 Valence-Arousal-Dominance（VAD）三维标注。</p>
          <ul>
            <li><b>MER-MULTI：</b>联合预测离散情绪类别与连续 Valence，考察多标签学习及两类情绪标注之间的关联。</li>
            <li><b>MER-NOISE：</b>在视频模糊、背景噪声等干扰下评估模型的模态鲁棒性。</li>
            <li><b>MER-SEMI：</b>利用 73,148 个未标注视频样本，通过半监督学习提升情绪识别性能。</li>
          </ul>
          <p><b>评测指标：</b>MER-MULTI 和 MER-NOISE 使用 Combined Score：F1<sub>weighted</sub> − 0.25 × MSE<sub>valence</sub>，综合衡量离散情绪分类与连续效价预测；MER-SEMI 仅评估离散情绪识别，采用 Weighted Average F-score。数据规模和视频时长依据论文 Table 1，情绪类别与标注形式依据论文第 2 页。</p>
        </td>
      </tr>
    </tbody>
  </table>
</div>
