'''
    按着路径，导入单张图片做预测
'''
from torchvision.models import resnet18
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import time

'''
    加载图片与格式转化
'''

# 图片标准化
transform_BZ= transforms.Normalize(
    mean=[0.5, 0.5, 0.5],# 取决于数据集
    std=[0.5, 0.5, 0.5]
)

val_tf = transforms.Compose([##简单把图片压缩了变成Tensor模式
                transforms.Resize(224),
                transforms.ToTensor(),
                transform_BZ#标准化操作
            ])


def padding_black(img):  # 如果尺寸太小可以扩充
    w, h = img.size
    scale = 224. / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
    size_fg = img_fg.size
    size_bg = 224
    img_bg = Image.new("RGB", (size_bg, size_bg))
    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))
    img = img_bg
    return img


def predict(img_path):
    # time.sleep(5)
    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")

    '''
        加载模型与参数
    '''

    # 加载模型
    model = resnet18(pretrained=False, num_classes=5)#.to(device)  # 43.6%

    if device=="cpu":
        # 加载模型参数
        model.load_state_dict(torch.load("model_resnet18_100.pth",map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load("model_resnet18_100.pth"))

    model = model.to(device)



    img = Image.open(img_path)  # 打开图片
    img = img.convert('RGB')  # 转换为RGB 格式
    img = padding_black(img)
    # print(type(img))

    img_tensor = val_tf(img)
    # print(type(img_tensor))

    # 增加batch_size维度
    img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False).to(device)

    '''
        数据输入与模型输出转换
    '''
    model.eval()
    with torch.no_grad():
        output_tensor = model(img_tensor)
        # print(output_tensor)

        # 将输出通过softmax变为概率值
        output = torch.softmax(output_tensor, dim=1)
        # print(output)

        # 输出可能性最大的那位
        pred_value, pred_index = torch.max(output, 1)
        # print(pred_value)
        # print(pred_index)

        # 将数据从cuda转回cpu
        if torch.cuda.is_available() == False:
            pred_value = pred_value.detach().cpu().numpy()
            pred_index = pred_index.detach().cpu().numpy()

        # print(pred_value)
        # print(pred_index)

        # 增加类别标签
        classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

        # print()
        result = "预测类别为： "+str(classes[pred_index[0]])+ " 可能性为: "+str(pred_value[0] * 100)+ "%"
        return result

if __name__ == "__main__":
    img_path = '../pic/sunflower3.jpg'
    result = predict(img_path)
    print(result)