# run_dashboard.py - Запускает дашборд из конфигурации

from explainerdashboard import ExplainerDashboard
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_dashboard():
    try:
        dashboard = ExplainerDashboard.from_config(
            'dashboard.yaml',
            explainerfile='explainer.joblib'
        )
        
        logger.info("Конфигурация загружена")
        logger.info("Дашборд доступен по адресу: http://0.0.0.0:9050")
        
        dashboard.run(
            host='0.0.0.0',
            port=9050,
            use_waitress=True,
            debug=False
        )
        
    except Exception as e:
        logger.error(f"Ошибка при запуске дашборда: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    run_dashboard()
