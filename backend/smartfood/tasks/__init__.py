"""
Tasks - Celery tasks for SmartFood
"""

from smartfood.tasks.training_task import train_model, cleanup_failed_jobs

__all__ = ['train_model', 'cleanup_failed_jobs']
