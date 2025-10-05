import yaml
def get_config(which):
    with open('../config.yaml','r',encoding='utf-8') as file:
        config = yaml.safe_load(file)[which]
    return config