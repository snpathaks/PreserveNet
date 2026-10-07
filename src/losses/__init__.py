"""
src/losses/__init__.py
"""
from src.losses.budget    import L1BudgetLoss, L2BudgetLoss
from src.losses.agreement import AgreementLoss, TaskLoss

__all__ = ["L1BudgetLoss", "L2BudgetLoss", "AgreementLoss", "TaskLoss"]
