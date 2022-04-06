import torch

class EfficientSolver:
    def __init__(self,model_path,dict_path):
        self.model = torch.load(model_path)
    
    # 暴力查询
    def query_brute_force(self, query_traj, k=10):
        '''
        mapper取出全表轨迹的id和特征向量
        用for循环进行简单的暴力比较,获取前k相似轨迹的id号
        再根据id用mapper取出前k相似轨迹的信息形成result返回
        '''
        pass