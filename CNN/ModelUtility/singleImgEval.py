import torch
import torchvision
import torchvision.transforms as transforms
from CNN.CNNModel.model.wide_res_net import WideResNet
from PIL import Image, ImageOps
import torch.nn.functional as F
import os

def predict(IMAGE_ROOT):
    # ------------- Select the device -------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # ------ Load the MODEL for evaluation ------
    path = './wide_resnet_state_dic.pt'
    PATH = os.path.abspath(path)
    print("PATH", end=' ')
    print(PATH)
    MODEL = WideResNet(depth=16, width_factor=8, dropout=.0, in_channels=3, labels=10).to(device)
    CHECK_POINT = torch.load(f=PATH, map_location=device)
    MODEL.load_state_dict(state_dict=CHECK_POINT, strict=True)
    MODEL.to(device)
    # ------- Enable Evaluation Mode for safe -------
    MODEL.eval()

    # ------ Set the evaluating properties ------
    DATA_ROOT = r'../../Dataset'
    fashion_mnist = torchvision.datasets.FashionMNIST(download=True, train=True, root=DATA_ROOT).train_data.float()
    TRANSFORM = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.Pad(padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            (fashion_mnist.mean() / 255,),
            (fashion_mnist.std() / 255,)
        )
    ])

    # ------- [Component] convert the argmax output to the label -------
    def output_label(label):
        output_mapping = {
            0: "T-shirt/Top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot"
        }
        num_index = (label.item() if type(label) == torch.Tensor else label)
        return output_mapping[num_index]

    # ------ [Component] image preprocessing ------
    def padding_black(img):
        """
            Add black padding to the image if necessary.

        Input:
            img (PIL Image)

        Return:
            img (PIL Image)
        """
        w, h = img.size
        scale = 32. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 32
        img_bg = Image.new("L", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    # ------ [Component] Change Fashion MNIST to match wide resnet input format (3, 40, 40) in this setting ------
    def permute_add_channel(inputs: torch.Tensor):
        inputs = inputs.permute(1, 0, 2, 3)
        inputs = torch.cat((inputs, inputs, inputs), dim=0)
        inputs = inputs.permute(1, 0, 2, 3)
        return inputs


    # 图片变成灰白，然后白底图片变成黑底(可用的) -> 直接 Resize -> ToTensor() 就可以了
    img = Image.open(IMAGE_ROOT)  # 打开图片
    img = ImageOps.invert(img)
    img = img.convert('L')

    img = TRANSFORM(img)
    img = torch.unsqueeze(input=img, dim=0)  # img with size [1, 1, 40, 40]
    x = permute_add_channel(img)
    x = torch.squeeze(input=x, dim=0)
    x = x.permute(1, 2, 0)

    img = permute_add_channel(inputs=img)  # img with size [1, 3, 40, 40]
    output=""
    MODEL.eval()
    with torch.no_grad():
        temp_pred_output = MODEL(img.to(device))
        eval_num_label = torch.argmax(temp_pred_output, dim=1)  # 通过x找到最大数对应位
        eval_confidence = torch.max(input=(F.softmax(input=temp_pred_output, dim=1) * 100), dim=1)[0].item()  # 通过x求出置信度
        print(
            "\t\t",
            "({})".format((output_label(eval_num_label))),
            "\t  With conficence: {0:.4f}%".format(eval_confidence)
        )
        x = F.softmax(input=temp_pred_output, dim=1) * 100
        x = torch.squeeze(x, dim=0)
        print("----------")
        for i in range(10):
            output+="\n"
            output+="({})".format(output_label(i))
            output+="{0:.4f}%".format(x[i].item())
            print(
                "({})".format(output_label(i)), "{0:.4f}%".format(x[i].item())
            )

    # result = "\t\t" + "({})".format((output_label(eval_num_label))) + "\t  With conficence: {0:.4f}%".format(
    #     eval_confidence)
    clothes = "({})".format((output_label(eval_num_label)))
    matrix=[0,1]
    # matrix[0]=clothes + "\nWith conficence: {0:.4f}%\n".format(
    #         eval_confidence)
    # matrix[1]=clothes = "({})".format((output_label(eval_num_label)))
    if clothes == '(Bag)':
        result = 'Wrong image entered!'
    else:
        result = clothes + "\nWith conficence: {0:.4f}%\n".format(
            eval_confidence)
    matrix[0]=result
    matrix[1]=output
    return matrix
