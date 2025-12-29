Hello,

This is my Machine Learning blog in Vietnamese:

[https://machinelearningcoban.com/](http://machinelearningcoban.com/)

## Running tiepvupsu.github.io Locally on macOS (Jekyll + GitHub Pages)

Install Ruby + Bundler
```bash
brew install ruby
gem install bundler
```

Clone the repository

```bash
git clone https://github.com/tiepvupsu/tiepvupsu.github.io.git
cd tiepvupsu.github.io
```

Install Jekyll / GitHub Pages dependencies
```bash
bundle install
```

Serve the site locally (Production mode)
```bash
JEKYLL_ENV=production bundle exec jekyll serve --baseurl=""
```

Then open http://localhost:4000