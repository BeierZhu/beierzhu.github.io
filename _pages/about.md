---
layout: about
title: about
titledisplay: about
permalink: /
subtitle: 

profile:
  align: left
  image: profile.jpg
  image_circular: true # crops the image to make it circular
  more_info: 

news: true # includes a list of news items
selected_papers: true # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page
---
<!-- ## Beier Zhu (朱贝尔)
       
<br/><br/> -->
**Experience**: I’m Beier Zhu (朱贝尔), currently a Research Fellow in the [MReaL Lab](https://mreallab.github.io/) at [Nanyang Technological University](https://www.ntu.edu.sg/), working with [Prof. Hanwang Zhang](https://personal.ntu.edu.sg/hanwangzhang/). I obtained my PhD degree from [Nanyang Technological University](https://www.ntu.edu.sg/), supported by the prestigious [AISG PhD](https://aisingapore.org/research/phd-fellowship-programme/) programme. Prior to that, I received my B.E. and M.E. degrees from [Tsinghua University](https://www.tsinghua.edu.cn/en/) in 2016 and 2019, respectively.

**Research**: My research focuses on developing robust machine learning algorithms with strong theoretical foundations, with particular interests in imbalanced learning, group robustness, out-of-distribution (OOD) generalization, and diffusion solvers. On the application side, I am also interested in multimodal foundation models (VLMs, MLLMs, and diffusion models), with an
emphasis on robust adaptation, faithful reasoning, and controllable generation.

<br/>

### News
<div class="news-scroll-box">
<table class="table table-sm table-borderless">
{% assign news = site.news | reverse %}
{% for item in news limit: site.news_limit %}
  <tr>
    <th scope="row" style="width: 20%">{{ item.date | date: "%b, %Y" }}</th>
    <td>
      {% if item.inline %}
        {{ item.content | remove: '<p>' | remove: '</p>' | emojify }}
      {% else %}
        <a class="news-title" href="{{ item.url | relative_url }}">{{ item.title }}</a>
      {% endif %}
    </td>
  </tr>
{% endfor %}
</table>
</div>

<br/> 
### Selected Publications 
(__First, second, and corresponding author papers__; * and ^ denote equal contribution and corresponding authorship.)

{% include bib_search.liquid %}

<div class="publications">

{% bibliography --file selected_papers %}

</div>
