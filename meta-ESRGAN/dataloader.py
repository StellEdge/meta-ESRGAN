from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from work_header import *
from PIL import Image
import random
import numpy as np

def addGaussian(img):
    if random.uniform(0,1)<0.1:
        sigma=0.02*255
        clean_image=np.array(img).copy()       
        noise_image=clean_image+(np.random.standard_normal(clean_image.shape)*sigma)
        noise_image=np.clip(noise_image, 0, 255)
        return Image.fromarray(noise_image.astype('uint8')).convert('RGB')
    else:
        return img

transform_n=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],std = [ 0.5, 0.5, 0.5 ])
])

transform=transforms.Compose([
    transforms.Lambda(addGaussian),
    transforms.ToTensor(),
])

r_transform=transforms.Compose([
    transforms.ToPILImage()
])

r_transform_n=transforms.Compose([
    transforms.Normalize(mean = [ -1, -1, -1 ],std = [ 2, 2, 2 ]),
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

def get_pic_dataset(folder_name,use_cut=False):
    return LRHR_Dataset(work_folder+folder_name+'/LR',work_folder+folder_name+'/HR',use_cut)

def get_pic_dataloader(folder_name,batch_size,n_cpu):
    return DataLoader(get_pic_dataset(folder_name), batch_size, shuffle=True, num_workers=n_cpu,pin_memory=True, drop_last=False)

class LRHR_Dataset(Dataset):
    def __init__(self,data_path_LR,data_path_HR,use_cut=False):
        super(LRHR_Dataset,self).__init__()
        LR_imgs = os.listdir(data_path_LR)
        self.LR_imgs = [os.path.join(data_path_LR, img) for img in LR_imgs]

        HR_imgs = os.listdir(data_path_HR)
        self.HR_imgs = [os.path.join(data_path_HR, img) for img in HR_imgs]

        self.use_cut=use_cut
    def cut(self,index):
        img_LR = Image.open(self.LR_imgs[index])
        img_HR = Image.open(self.HR_imgs[index])

        img_LR_width = img_LR.width - 128
        img_LR_height = img_LR.height - 128
        img_HR_width = img_HR.width - 512
        img_HR_height = img_HR.height - 512

        cut_LR_h = min(random.randint(0, img_LR_height), random.randint(0, img_HR_height) / 4)
        cut_LR_w = min(random.randint(0, img_LR_width), random.randint(0, img_HR_width) / 4)
        fixed_img_LR = img_LR.crop((cut_LR_w, cut_LR_h, cut_LR_w + 128, cut_LR_h + 128))
        fixed_img_HR = img_HR.crop((cut_LR_w * 4, cut_LR_h * 4, cut_LR_w * 4 + 512, cut_LR_h * 4 + 512))
        #return fixed_img_LR.convert('RGB'), fixed_img_HR.convert('RGB')
        return fixed_img_LR, fixed_img_HR
    def __getitem__(self,index):
        #img_LR=transform(Image.open(self.LR_imgs[index]).convert('RGB'))
        #img_HR=transform(Image.open(self.HR_imgs[index]).convert('RGB'))
        if self.use_cut:
            fixed_img_LR,fixed_img_HR=self.cut(index)
            img_LR=transform(fixed_img_LR)
            img_HR=transform(fixed_img_HR)
        else:
            img_LR =transform( Image.open(self.LR_imgs[index]))
            img_HR =transform( Image.open(self.HR_imgs[index]))
        return {
            'LR': img_LR,
            'HR': img_HR,
            'LR_path': self.LR_imgs[index],
            'HR_path': self.HR_imgs[index]
        }
    def __len__(self):
        return min(len(self.HR_imgs),len(self.LR_imgs))
'''
class LRHR_Dataset_fast(Dataset):
    #requires super large memory
    def __init__(self,data_path_LR,data_path_HR):
        super(LRHR_Dataset_fast,self).__init__()
        LR_imgs = os.listdir(data_path_LR)
        self.LR_imgs = [Image.open(os.path.join(data_path_LR, img)) for img in LR_imgs]

        HR_imgs = os.listdir(data_path_HR)
        self.HR_imgs = [Image.open(os.path.join(data_path_HR, img)) for img in HR_imgs]
        
        self.LR_imgs_cut=[]
        self.HR_imgs_cut=[]
        for i in range(len(self.LR_imgs)):
            print("Converting %d" % (i))
            l,h=self.cut(i)
            self.LR_imgs_cut.append(l)
            self.HR_imgs_cut.append(h)
        print("complete")
    def cut(self,index):
        img_LR = self.LR_imgs[index]
        img_HR = self.HR_imgs[index]

        img_LR_width = img_LR.width - 128
        img_LR_height = img_LR.height - 128
        img_HR_width = img_HR.width - 512
        img_HR_height = img_HR.height - 512

        cut_LR_h = min(random.randint(0, img_LR_height), random.randint(0, img_HR_height) / 4)
        cut_LR_w = min(random.randint(0, img_LR_width), random.randint(0, img_HR_width) / 4)
        fixed_img_LR = img_LR.crop((cut_LR_w, cut_LR_h, cut_LR_w + 128, cut_LR_h + 128))
        fixed_img_HR = img_HR.crop((cut_LR_w * 4, cut_LR_h * 4, cut_LR_w * 4 + 512, cut_LR_h * 4 + 512))
        return fixed_img_LR.convert('RGB'), fixed_img_HR.convert('RGB')
    def __getitem__(self,index):
        img_LR=transform(self.LR_imgs_cut[index])
        img_HR=transform(self.HR_imgs_cut[index])
        #fixed_img_LR,fixed_img_HR=self.cut(index)
        #img_LR=transform_n(fixed_img_LR)
        #img_HR=transform_n(fixed_img_HR)
        return {
            'LR': img_LR,
            'HR': img_HR,
            #'LR_path': self.LR_imgs_cut[index],
            #'HR_path': self.HR_imgs_[index]
        }
    def __len__(self):
        return min(len(self.HR_imgs),len(self.LR_imgs))
'''
class RandomShotDataset(Dataset):
    #Customized random shot dataset
    def __init__(self,data_path_LR,data_path_HR,shot_size):
        super(RandomShotDataset,self).__init__()
        self.HR_imgs=[]
        self.LR_imgs=[]
        lr_suffix='x4'
        HR_imgs = np.random.choice(os.listdir(data_path_HR), size=shot_size, replace=False, p=None)
        for img in HR_imgs:
            img_name=img.split('.')[0]
            self.HR_imgs.append(os.path.join(data_path_HR, img))
            self.LR_imgs.append(os.path.join(data_path_LR, img_name+lr_suffix+'.png'))

    def cut(self,index):
        img_LR = Image.open(self.LR_imgs[index])
        img_HR = Image.open(self.HR_imgs[index])

        img_LR_width = img_LR.width - 128
        img_LR_height = img_LR.height - 128
        img_HR_width = img_HR.width - 512
        img_HR_height = img_HR.height - 512

        cut_LR_h = min(random.randint(0, img_LR_height), random.randint(0, img_HR_height) / 4)
        cut_LR_w = min(random.randint(0, img_LR_width), random.randint(0, img_HR_width) / 4)
        fixed_img_LR = img_LR.crop((cut_LR_w, cut_LR_h, cut_LR_w + 128, cut_LR_h + 128))
        fixed_img_HR = img_HR.crop((cut_LR_w * 4, cut_LR_h * 4, cut_LR_w * 4 + 512, cut_LR_h * 4 + 512))
        return fixed_img_LR.convert('RGB'), fixed_img_HR.convert('RGB')
    def __getitem__(self,index):
        fixed_img_LR,fixed_img_HR=self.cut(index)
        img_LR=transform_n(fixed_img_LR)
        img_HR=transform_n(fixed_img_HR)
        return {
            'LR': img_LR,
            'HR': img_HR,
            'LR_path': self.LR_imgs[index],
            'HR_path': self.HR_imgs[index]
        }
    def __len__(self):
        return min(len(self.HR_imgs),len(self.LR_imgs))



def fetch_dataloaders(types,n_cpu,folder_name):
    """
    Fetches the DataLoader object for each type in types from task.
    In super resolution tasks there is no such thing like task.
    So it is replaced with totally Random datasets.
    TODO for MAML

    Args:
        types: (list) has one or more of 'train', 'val', 'test' 
               depending on which data is required
    Returns:
        dataloaders: (dict) contains the DataLoader object for each type in types
    """
    shot_size=2
    #how many images in a shot

    test_suffix='_test'
    val_suffix='_val'

    dataloaders = {}
    for split in ['train', 'test', 'val']:
        if split in types:
            # use the train_transformer if training data,
            # else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(
                    RandomShotDataset(work_folder+'/'+folder_name+'/LR',
                                      work_folder+'/'+folder_name+'/HR',shot_size),
                    batch_size=shot_size,  # full-batch in episode
                    shuffle=True,num_workers=n_cpu,pin_memory=True)
            elif split == 'test':
                dl = DataLoader(
                    RandomShotDataset(work_folder+'/'+folder_name+test_suffix+'/LR',
                                      work_folder+'/'+folder_name+test_suffix+'/HR',shot_size),
                    batch_size=shot_size,  # full-batch in episode
                    shuffle=False,num_workers=n_cpu,pin_memory=True)
            elif split == 'val':
                dl = DataLoader(
                    RandomShotDataset(work_folder+'/'+folder_name+val_suffix+'/LR',
                                      work_folder+'/'+folder_name+val_suffix+'/HR',shot_size),
                    batch_size=shot_size,  # full-batch in episode
                    shuffle=False,num_workers=n_cpu,pin_memory=True)
            else:
                raise NotImplementedError()
            dataloaders[split] = dl

    return dataloaders    