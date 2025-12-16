# FrustraMPNN GitHub Pages Site

This directory contains the static website for the FrustraMPNN paper landing page.

## Structure

```
site/
├── index.html          # Main paper landing page
├── docs.html           # Documentation page
├── 404.html            # 404 error page
├── _config.yml         # Jekyll configuration
├── .nojekyll           # Bypass Jekyll processing
├── assets/
│   └── css/
│       └── style.css   # Main stylesheet
└── README.md           # This file
```

## Deployment

### GitHub Pages

1. Go to repository Settings → Pages
2. Set Source to "Deploy from a branch"
3. Select branch: `main` (or your default branch)
4. Select folder: `/docs/site`
5. Save

The site will be available at: `https://schoederlab.github.io/frustraMPNN/`

### Manual Deployment

To test locally:

```bash
cd docs/site
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

## Design Principles

The site follows a minimalist, sober scientific design:

- **Typography**: Source Sans 3 (sans-serif), Source Serif 4 (serif), JetBrains Mono (code)
- **Colors**: Muted professional palette with blue accent
- **Layout**: Single-column, max-width 900px for readability
- **Frustration colors**: Red (highly), Gray (neutral), Green (minimally)

## Pages

### index.html (Paper Landing Page)

- Hero section with title, authors, affiliations
- Key findings (speedup, accuracy metrics)
- Abstract
- What is frustration explanation
- Results summary with tables
- Methods overview
- Resources (installation, quick start, links)
- Citation with BibTeX
- Contact information

### docs.html (Documentation)

- Installation instructions
- Quick start guide
- Python API reference
- CLI reference
- Visualization guide
- Validation guide
- Docker/Singularity guide

## Customization

### Colors

Edit CSS variables in `assets/css/style.css`:

```css
:root {
    --color-accent: #2563eb;        /* Primary accent */
    --color-highly: #c62828;        /* Highly frustrated */
    --color-neutral: #757575;       /* Neutral */
    --color-minimal: #2e7d32;       /* Minimally frustrated */
}
```

### Content

Edit the HTML files directly. Key sections are marked with comments.

## License

MIT License - see repository root for details.
