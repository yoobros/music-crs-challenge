"""`python -m mymodule.feature.crawl` entry point — delegates to `cli.main`."""

from mymodule.feature.crawl.cli import main

raise SystemExit(main())
