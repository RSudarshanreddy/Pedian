# Explicit Dockerfile instead of Cloud Run's buildpack auto-detection --
# buildpacks picked a minimal base image missing libsqlite3, which yfinance's
# cookie/cache layer requires (see load_yfinance() in src/swings.py). Standard
# Debian-based python:3.13-slim always ships with a working sqlite3 module,
# sidestepping that whole class of problem.

FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/swings.py .

ENV PORT=8080
CMD exec functions-framework --target=main --source=swings.py --port=${PORT}
