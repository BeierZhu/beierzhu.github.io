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

news: false # includes a list of news items
selected_papers: true # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page
---
<!-- ## Beier Zhu (朱贝尔)
       
<br/><br/> -->
<div class="language-content language-block" data-language-content="en">
<p><b>Experience</b>: I’m Beier Zhu (朱贝尔), currently a Professor at <a href="https://en.ustc.edu.cn/">University of Science and Technology of China</a> (USTC). Prior to joining USTC, I was a Research Fellow in the <a href="https://mreallab.github.io/">MReaL Lab</a> at <a href="https://www.ntu.edu.sg/">Nanyang Technological University</a> (NTU), working with <a href="https://personal.ntu.edu.sg/hanwangzhang/">Prof. Hanwang Zhang</a>. I obtained my PhD degree from <a href="https://www.ntu.edu.sg/">NTU</a>, supported by the prestigious <a href="https://aisingapore.org/research/phd-fellowship-programme/">AISG PhD</a> programme. Prior to that, I received my B.E. and M.E. degrees from <a href="https://www.tsinghua.edu.cn/en/">Tsinghua University</a> in 2016 and 2019, respectively.</p>

<p><b>Research</b>: My research focuses on <b>foundation models</b>, with particular interests in multimodal reasoning, agentic intelligence, affective intelligence, controllable and efficient diffusion generation, and reliable model adaptation. I am also interested in <b>robust learning and principled optimization</b>, which provide theoretical foundations for improving foundation models.</p>
</div>

<div class="language-content language-block" data-language-content="zh">
<p><b>经历</b>：我目前是<a href="https://en.ustc.edu.cn/">中国科学技术大学</a>教授。我曾是<a href="https://mreallab.github.io/">MReaL Lab</a>研究员，就职于<a href="https://www.ntu.edu.sg/">南洋理工大学</a>，并与<a href="https://personal.ntu.edu.sg/hanwangzhang/">张含望教授</a>合作。我在<a href="https://www.ntu.edu.sg/">南洋理工大学</a>获得博士学位，获<a href="https://aisingapore.org/research/phd-fellowship-programme/">AISG PhD</a>项目资助。此前，我于 2016 年和 2019 年分别获得<a href="https://www.tsinghua.edu.cn/en/">清华大学</a>学士和硕士学位。</p>

<p><b>研究方向</b>：我的研究主要集中在<b>大模型</b>，尤其关注多模态推理、智能体智能、情感智能、可控且高效的扩散生成，以及可靠的模型适配。同时，我也研究<b>鲁棒机器学习和优化方法</b>，这些方向为改进基础模型提供理论基础。</p>
</div>

 
<!-- <div style="height: 1.5em;"></div> -->
<!-- <div class="hiring-banner"> -->
  <!-- <b> 📢 We're hiring!</b> Positions for PhD and master students are available! Students who are interested are welcome to email beier.zhu@ustc.edu.cn -->
   <!-- for inquiries, and please attach your CV.  -->
  <!-- <a href="/招生信息/">See 招生信息 →</a> -->
<!-- </div> -->

<div style="height: 1.5em;"></div>

<h3><span data-i18n="home.news">News</span></h3>
<hr style="margin-top: 0.3em; margin-bottom: 1em;">
<div class="news-scroll-box" style="font-weight: 300;">
<table class="table table-sm table-borderless" style="font-weight: 300;">
{% assign news = site.news | reverse %}
{% for item in news limit: site.news_limit %}
  <tr>
    <td style="width: 20%">
      <span class="language-content language-inline" data-language-content="en">{{ item.date | date: "%b, %Y" }}</span>
      <span class="language-content language-inline" data-language-content="zh">{{ item.date | date: "%Y年%-m月" }}</span>
    </td>
    <td>
      {% if item.inline %}
        {% assign news_key = item.path | remove: '_news/' | remove: '.md' %}
        <span class="language-content language-inline" data-language-content="en">{{ item.content | remove: '<p>' | remove: '</p>' | emojify }}</span>
        <span class="language-content language-inline" data-language-content="zh">{{ site.data.news_zh[news_key] | emojify }}</span>
      {% else %}
        <a class="news-title" href="{{ item.url | relative_url }}">{{ item.title }}</a>
      {% endif %}
    </td>
  </tr>
{% endfor %}
</table>
</div>

<div style="height: 3em;"></div>

<h3>
  <span data-i18n="home.selectedPublications">Selected Publications</span>
  <a href="/publications/" class="view-full-list-btn-inline"><span data-i18n="home.viewFullList">View Full List</span></a>
</h3>
<hr style="margin-top: 0.3em; margin-bottom: 1em;">
 <i class="fa-solid fa-handshake" style="font-size: 0.7em; vertical-align: super; margin-left: 1px;"></i>
 <span class="language-content language-inline" data-language-content="en">and <i class="fa-solid fa-envelope"></i> denote equal contribution and corresponding authorship.</span>
 <span class="language-content language-inline" data-language-content="zh">和 <i class="fa-solid fa-envelope"></i> 表示共同贡献和通讯作者。</span>

{% include bib_search.liquid %}

<div class="publications selected-publications">

{% bibliography --file selected_papers --group_by none %}

</div>

<div class="view-more-container">
  <button class="view-more-btn" id="toggle-publications" data-i18n="home.viewMore">View More Publications</button>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const toggleBtn = document.getElementById('toggle-publications');
  const pubSection = document.querySelector('.selected-publications');
  let expanded = false;

  function updatePublicationToggleLabel() {
    toggleBtn.dataset.i18n = expanded ? 'home.showLess' : 'home.viewMore';
    if (window.applySiteLanguage) {
      window.applySiteLanguage();
    }
  }
  
  toggleBtn.addEventListener('click', function() {
    expanded = !expanded;
    if (expanded) {
      pubSection.classList.add('expanded');
    } else {
      pubSection.classList.remove('expanded');
    }
    updatePublicationToggleLabel();
  });
});
</script>
