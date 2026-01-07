"""
Models - Database models for SmartFood
"""

from smartfood.models.training_job import TrainingJob, JobStatus, db

__all__ = ['TrainingJob', 'JobStatus', 'db']
