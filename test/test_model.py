import torch
from models.my_model.AAA_HWD_U2Net_V2 import AAA_HWD_U2Net_V2
def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = AAA_HWD_U2Net_V2()
    model.load_state_dict(torch.load(model_path, map_location=device)['net'])
    model.to(device)
    model.eval()  # 设置为评估模式
    return model
if __name__ == '__main__':
    path1="/home/huang/work/InfraredSmallTargets/Code/AAA_HWD_U2Net/"
    path2="train/AAA_HWD_U2Net_V2_At_IRSTD-1k/2025_10_05_18_14_07_bs4_lr0.001/checkpoint/Epoch=  1_IoU=0.0009_nIoU=0.0034_Fa=0.31331727_Pd=0.07790463.pth"
    model = load_model(path1+path2)