# Beier Zhu's Academic Website

## Overview

This is a personal academic website for Beier Zhu, built using Jekyll with the al-folio theme. The site showcases research publications, notes, and news updates. It is designed to be hosted on GitHub Pages and features a clean, professional design suitable for academic researchers.

The website includes:
- About page with profile and selected publications
- Publications page with searchable bibliography
- Notes page linking to research PDFs
- News/announcements section
- Dark/light theme support

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Static Site Generator
- **Jekyll**: Ruby-based static site generator that transforms Markdown and Liquid templates into HTML
- The site uses the al-folio theme, a popular Jekyll theme designed for academics
- Configuration is managed through `_config.yml` (not shown but implied by Jekyll structure)

### Content Organization
- `_pages/`: Main site pages (about, publications, notes, 404)
- `_news/`: News/announcement items as individual Markdown files
- `_layouts/`: HTML templates for page rendering
- `_includes/`: Reusable Liquid template components
- `_site/`: Generated static HTML output (should not be edited directly)
- `assets/`: Static assets including CSS, JavaScript, images, and PDFs

### Frontend Stack
- **Bootstrap 4.6.2**: CSS framework for responsive layout
- **MDB (Material Design for Bootstrap)**: Additional Material Design styling
- **jQuery**: JavaScript library for DOM manipulation
- **Custom JavaScript modules**: Theme switching, bibliography search, code copying, zoom functionality

### Build System
- Jekyll builds the site from source files
- PurgeCSS configured via `purgecss.config.js` to remove unused CSS
- Prettier with Shopify Liquid plugin for code formatting
- Dev container configuration for consistent development environment

### Bibliography/Publications
- Uses Jekyll-Scholar or similar plugin for BibTeX bibliography management
- Custom JavaScript (`bibsearch.js`) provides client-side publication search with highlighting
- Publications can be marked as "selected" for homepage display

### Theme System
- Supports light, dark, and system-preference themes
- Theme state persisted in localStorage
- Dynamic theme switching affects syntax highlighting, Giscus comments, and embedded content

## External Dependencies

### Editing team members

The Teams page is populated from `_data/teams/*.json`, one file per student.
Duplicate an existing file to add a student; files are displayed in filename order within each degree group.
Use `"degree": "phd"` or `"degree": "master"`.
`name.en` and `name.zh` provide the English and Chinese names. Display only the active language's name on one line, without a hyperlink.
`research.en` and `research.zh` provide the research direction displayed at the bottom of the card in each language.
Separate research topics with English or Chinese commas (`,` / `，`). Each nonempty trimmed topic appears on its own centered line without the comma; the topic block is vertically centered within the shared research area for that row.
`graduated` is a JSON boolean: `false` (or omitted for older records) keeps the student in PhD/Master; `true` moves them exclusively to the collapsible Alumni section at the end, which is closed initially.
Graduated students must supply `graduation_year` (a four-digit integer) and `graduation_destination.en` / `.zh` (bilingual text). These replace the research direction at the bottom of the alumni card; research is not required for alumni.
PhD Student 1 is marked graduated with a 2026 graduation year and an explicitly fictional destination as an example.
`enrollment` is a `"YYYY-MM"` string; display it as `Intake Sep. 2026` in English (abbreviated month with a period) and `入学时间 2026年9月` in Chinese.
`email` supplies the mail link, and `photo` is a site-local path, for example `/assets/img/teams/student.jpg`.
Email is displayed as an envelope icon, not as visible address text.
`homepage` is optional: omit it or set it to `""` to hide the homepage icon. Names are never linked; the homepage is accessed only through its icon.
`google_scholar` is an optional full Google Scholar profile URL: omit it or set it to `""` to hide its icon.
Cards are centered vertically with circular portraits and four columns on desktop, reducing columns on smaller screens.
The initial seven students, emails, photos, example.com homepage links, and generic Google Scholar links are placeholders, not real members.
Updating these files and rebuilding automatically updates the layout.

### CDN Resources
- **MDBootstrap**: CSS framework loaded from jsdelivr CDN
- **Google Fonts**: Roboto and Roboto Slab font families
- **Academicons**: Academic icon font for social links
- **Font Awesome**: Icon library (implied by code references)

### Development Tools
- **nbconvert**: Jupyter notebook conversion (in requirements.txt)
- **Prettier**: Code formatting
- **Node.js/npm**: JavaScript package management

### Hosting
- Designed for GitHub Pages deployment
- Uses Git for version control with pushes to `beierzhu.github.io`

### Optional Integrations
- **Giscus**: GitHub-based commenting system (referenced in theme.js)
- **Medium Zoom**: Image zoom functionality
- **Mermaid**: Diagram rendering (conditionally loaded)
- **Chart.js/ECharts/Vega-Lite**: Data visualization libraries (conditionally loaded)