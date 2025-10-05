from models.my_model.AAA_HWD_U2Net_V1 import AAA_HWD_U2Net_V1
from models.my_model.AAA_HWD_U2Net_V2 import AAA_HWD_U2Net_V2
from utils.get_config import get_config
class model_creater:
    def __init__(self, train):
        self.model_list=get_config('model_list')
        self.model = str(train['model'])
        if self.model in self.model_list:
            self.net=eval(self.model)()
        else:
            self.net=None
    def get_net(self):
        return self.net

    def get_model_name(self):
        return self.model

