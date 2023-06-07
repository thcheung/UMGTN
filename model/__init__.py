from .MultiModalGraphModel import MultiModalGraphModel
from .MultiModalGraphModel2 import MultiModalGraphModel2

def get_model(model_name, hidden_dim, classes, dropout, language):
    if model_name == 'default':
        return MultiModalGraphModel(hidden_dim, classes,
                            dropout, language)
    if model_name == 'ablation':
        return MultiModalGraphModel2(hidden_dim, classes,
                            dropout, language)