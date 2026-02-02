from celery import Celery
from paths import REDIS_URL

celery_app = Celery(
    "iquana_service_instance_discovery", # Must match the name in your other services
    broker=f"{REDIS_URL}/0",
    backend=f"{REDIS_URL}/1"
)