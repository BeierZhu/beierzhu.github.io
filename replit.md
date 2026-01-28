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