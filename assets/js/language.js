(function () {
  'use strict';

  const STORAGE_KEY = 'site-language';
  const translations = {
    en: {
      'nav.about': 'about',
      'nav.publications': 'publications',
      'nav.notes': 'notes',
      'nav.blog': 'blog',
      'home.professor': 'Professor',
      'home.curriculumVitae': 'Curriculum Vitae',
      'home.news': 'News',
      'home.selectedPublications': 'Selected Publications',
      'home.viewFullList': 'View Full List',
      'home.viewMore': 'View More Publications',
      'home.showLess': 'Show Less',
      'page.publications': 'Publications',
      'page.references': 'References',
      'publications.byVenue': 'By Venue and Authorship',
      'publications.byTopic': 'By Research Topic',
      'publications.venue': 'Venue',
      'publications.papers': 'Papers',
      'publications.firstCorresponding': '1st and',
      'publications.total': 'Total',
      'publications.category': 'Category',
      'publications.topic': 'Topic',
      'publications.searchPlaceholder': 'Type to filter',
      'bib.abstract': 'Abs',
      'bib.arxiv': 'arXiv',
      'bib.bib': 'Bib',
      'bib.html': 'HTML',
      'bib.pdf': 'PDF',
      'bib.supp': 'Supp',
      'bib.video': 'Video',
      'bib.blog': 'Blog',
      'bib.code': 'Code',
      'bib.project': 'PROJECT',
      'bib.poster': 'Poster',
      'bib.slides': 'Slides',
      'bib.website': 'Website',
    },
    zh: {
      'nav.about': '关于',
      'nav.publications': '发表论文',
      'nav.notes': '笔记',
      'nav.blog': '博客',
      'home.professor': '教授',
      'home.curriculumVitae': '个人简历',
      'home.news': '新闻',
      'home.selectedPublications': '精选论文',
      'home.viewFullList': '查看完整列表',
      'home.viewMore': '查看更多论文',
      'home.showLess': '收起',
      'page.publications': '发表论文',
      'page.references': '参考文献',
      'publications.byVenue': '按会议和作者身份',
      'publications.byTopic': '按研究方向',
      'publications.venue': '会议',
      'publications.papers': '论文数',
      'publications.firstCorresponding': '第一作者及',
      'publications.total': '总计',
      'publications.category': '类别',
      'publications.topic': '方向',
      'publications.searchPlaceholder': '输入关键词筛选',
      'bib.abstract': '摘要',
      'bib.arxiv': 'arXiv',
      'bib.bib': 'Bib',
      'bib.html': '网页',
      'bib.pdf': 'PDF',
      'bib.supp': '补充材料',
      'bib.video': '视频',
      'bib.blog': '博客',
      'bib.code': '代码',
      'bib.project': '项目主页',
      'bib.poster': '海报',
      'bib.slides': '幻灯片',
      'bib.website': '网站',
    },
  };

  function readLanguage() {
    try {
      return localStorage.getItem(STORAGE_KEY) === 'zh' ? 'zh' : 'en';
    } catch (error) {
      return 'en';
    }
  }

  function saveLanguage(language) {
    try {
      localStorage.setItem(STORAGE_KEY, language);
    } catch (error) {
      // Ignore storage restrictions; the current page can still switch.
    }
  }

  function applySiteLanguage(language, persist) {
    const nextLanguage = language === 'zh' ? 'zh' : 'en';
    const dictionary = translations[nextLanguage];
    document.documentElement.dataset.language = nextLanguage;
    document.documentElement.lang = nextLanguage === 'zh' ? 'zh-CN' : 'en';

    document.querySelectorAll('[data-i18n]').forEach(function (element) {
      const value = dictionary[element.dataset.i18n];
      if (value !== undefined) {
        element.textContent = value;
      }
    });

    document.querySelectorAll('[data-i18n-placeholder]').forEach(function (element) {
      const value = dictionary[element.dataset.i18nPlaceholder];
      if (value !== undefined) {
        element.setAttribute('placeholder', value);
      }
    });

    const toggle = document.getElementById('language-toggle');
    if (toggle) {
      const label = toggle.querySelector('[data-language-label]');
      if (label) {
        label.textContent = nextLanguage === 'zh' ? 'EN' : '中';
      }
      const nextLabel = nextLanguage === 'zh' ? 'Switch to English' : 'Switch to Chinese';
      toggle.setAttribute('aria-label', nextLabel);
      toggle.setAttribute('title', nextLabel);
    }

    if (persist !== false) {
      saveLanguage(nextLanguage);
    }
  }

  window.applySiteLanguage = applySiteLanguage;

  document.documentElement.dataset.language = readLanguage();

  document.addEventListener('DOMContentLoaded', function () {
    applySiteLanguage(readLanguage(), false);

    const toggle = document.getElementById('language-toggle');
    if (toggle) {
      toggle.addEventListener('click', function () {
        const currentLanguage = document.documentElement.dataset.language || 'en';
        applySiteLanguage(currentLanguage === 'zh' ? 'en' : 'zh');
      });
    }
  });
})();