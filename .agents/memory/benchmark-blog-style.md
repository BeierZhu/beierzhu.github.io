---
name: Benchmark blog style
description: User-approved editorial presentation for the Chinese text-only benchmark blog.
---

该测评基准博客的后续条目沿用简洁对比表格的呈现方式，引用等补充信息按需展开，不另设独立参考文献章节。样例通过简介末尾的“[样例]”展开，使用文字，不直接嵌入截图。

**Why:** 用户在迭代样稿后明确表示“可以 不错”，并要求“按照类似的风格做 Natural Questions”；认可 Natural Questions 的可展开文字样例后表示“好”，要求继续制作 TriviaQA。

**How to apply:** 补充该博客的其他基准时保持已确认的呈现方式，不因新增条目重新设计整篇文章。

“规模”栏默认指测试集规模，不逐项重复标注“测试集”，也不要混入训练集或整体语料规模。

**Why:** 用户要求“规模默认都是测试集规模”，并要求将 Pile 改为 Pile-test、更新其规模，去掉 Natural Questions 的“测试集”标注。

**How to apply:** 新增或修订条目时先核实所用测试划分，再填写对应规模。对于没有统一测试规模的多任务合集，标明“依任务而定”，在简介中另列任务数，不用任务数量代替测试样本数量。
