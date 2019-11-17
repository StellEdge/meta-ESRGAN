from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from work_header import *
from PIL import Image

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
    return LRHR_Dataset(work_folder+folder_name+'/LR',work_folder+folder_name+'/HR')

def get_pic_dataloader(folder_name,batch_size):
    return DataLoader(get_pic_dataset(folder_name), batch_size, shuffle=True, num_workers=0, drop_last=False)

class LRHR_Dataset(Dataset):
    def __init__(self,data_path_LR,data_path_HR):
        super(LRHR_Dataset,self).__init__()
        LR_imgs = os.listdir(data_path_LR)
        self.LR_imgs = [os.path.join(data_path_LR, img) for img in LR_imgs]

        HR_imgs = os.listdir(data_path_HR)
        self.HR_imgs = [os.path.join(data_path_HR, img) for img in HR_imgs]

    def __getitem__(self,index):
        img_LR=transform(Image.open(self.LR_imgs[index]).convert('RGB'))
        img_HR=transform(Image.open(self.HR_imgs[index]).convert('RGB'))
        return {
            'LR': img_LR,
            'HR': img_HR,
            'LR_path': self.LR_imgs[index],
            'HR_path': self.HR_imgs[index]
        }
    def __len__(self):
        return min(len(self.HR_imgs),len(self.LR_imgs))

