# Databricks notebook source
import sys
sys.path.insert(0,dbutils.widgets.get('code_path'))
from src.steering_diagnostic import report
result=report(dbutils.widgets.get('admission'))
dbutils.notebook.exit(__import__('json').dumps({'status':result['status'],'training_cases':result['training_cases'],'new_model_forwards':0}))
