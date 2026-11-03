---
layout: post
title: "大模型测评基准I：纯文本基准"
date: 2026-11-03
permalink: /blog/2026/text-only-benchmarks/
description: "本文旨在系统梳理主流基准，便于个人查阅参考。"
tags: [大模型, 测评基准]
categories: [blog]
related_posts: false
toc:
  beginning: true
_styles: |
  .benchmark-preview-table table { min-width: 800px; margin-bottom: 0; }
  .benchmark-preview-table th { white-space: nowrap; }
  .benchmark-preview-table td { vertical-align: top; }
  .benchmark-preview-table td:nth-child(3) { min-width: 20rem; }
  .benchmark-preview-table td:not(:nth-child(3)) { white-space: nowrap; }
  .benchmark-reference summary { cursor: pointer; color: var(--global-theme-color); }
  .benchmark-citation { width: 16rem; max-width: 70vw; margin-top: .5rem; white-space: normal; font-size: .85rem; line-height: 1.6; }
---


## 1. 通用推理

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table>
    <thead>
      <tr>
        <th scope="col">名称</th>
        <th scope="col">发布时间</th>
        <th scope="col">简介</th>
        <th scope="col">规模</th>
        <th scope="col">评测指标</th>
        <th scope="col">SoTA</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>
          <details class="benchmark-reference">
            <summary title="点击展开或收起引用"><strong>Pile</strong></summary>
            <div class="benchmark-citation">[1] Gao et al. <em>The Pile: An 800GB Dataset of Diverse Text for Language Modeling.</em> arXiv:2101.00027. <a href="https://arxiv.org/abs/2101.00027">论文链接</a>。</div>
          </details>
        </td>
        <td>2020 年</td>
        <td>由 <strong>22 个多领域子数据集</strong>组成的大规模英文文本语料库，涵盖学术论文、书籍、代码、网页、法律文档及在线对话等，主要用于评估语言模型的<strong>跨领域语言建模能力</strong>。</td>
        <td>825 GB</td>
        <td>PPL / BPB</td>
        <td>—</td>
      </tr>
    </tbody>
  </table>
</div>


### 参考文献

<p id="ref-pile">[1] Gao et al. <a href="https://arxiv.org/abs/2101.00027">The Pile: An 800GB Dataset of Diverse Text for Language Modeling</a>. arXiv:2101.00027.</p>
