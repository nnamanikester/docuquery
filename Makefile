publish-test:
	uv sync
	uv run pytest
	rm -rf dist/
	uv build
	uvx twine check --strict dist/*
	uvx twine upload --repository testpypi dist/*
