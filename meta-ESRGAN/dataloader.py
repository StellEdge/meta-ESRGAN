from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from work_header import *

transform=transforms.Compose([
    transforms.ToTensor()
])

r_transform=transforms.Compose([
    transforms.ToPILImage()
])

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def get_image_tensor(data_origin,transform):
    imgs=os.listdir(data_origin)
    origin_img_paths=[os.path.join(data_origin,k) for k in imgs]
    all_arr=[]
    for path in origin_img_paths:
        if path.lower().endswith(IMG_EXTENSIONS):
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                imgt=transform(img)
                #img_arr=np.asarray(img)   
                #may translate into 3 channel triumph
                all_arr.append(imgt.view([1,3,256,256]))
    origin_face_tensor=torch.cat(all_arr)
    return origin_face_tensor

def get_pic_dataset(folder_name):
    return ImageFolder(work_folder+folder_name,transform)
