FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Установка зависимостей (не нужен requirements.txt)
RUN pip install --upgrade pip && \
    pip install explainerdashboard scikit-learn pandas numpy catboost waitress

# Скрипты находятся в dashboard-service/
COPY generate_dashboard.py run_dashboard.py startup.sh ./

RUN mkdir -p models data .cache && chmod +x startup.sh

EXPOSE 9050

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9050/ || exit 1

CMD ["bash", "startup.sh"]
