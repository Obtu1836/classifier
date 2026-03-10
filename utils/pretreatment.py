import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class LetterResize:
    def __init__(self, newshape, limitbig=False):

        self.newshape = newshape
        self.limitbig = limitbig

    def _letter(self, img):

        img_array = np.asarray(img)[:, :, ::-1]
        orih, oriw = img_array.shape[:2]
        newh, neww = self.newshape

        r = min(newh/orih, neww/oriw)

        if  self.limitbig:
            r = min(r, 1)

        unpadh, unpadw = int(r*orih), int(r*oriw)

        if orih != unpadh and oriw != unpadw:
            img_array = cv2.resize(
                img_array, (unpadw, unpadh), interpolation=cv2.INTER_LINEAR)

        dh, dw = newh-unpadh, neww-unpadw

        top = dh//2
        down = dh-top

        left = dw//2
        right = dw-left

        # res = cv2.copyMakeBorder(img_array, top, down,
        #                          left, right, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        res = cv2.copyMakeBorder(img_array, top, down,
                                 left, right, borderType=cv2.BORDER_REFLECT)

        return res[:, :, ::-1]

    def __call__(self, img):

        array = self._letter(img)
        return Image.fromarray(array)
    
class Resize:
    def __init__(self,use_letter:bool,newshape=(224,224), limitbig=False):

        self.use_letter=use_letter
        self.newshape=newshape
        self.limitbig=limitbig
    
    def __call__(self,img):

        if self.use_letter:
            fun=LetterResize(self.newshape,self.limitbig)
            return fun(img)
        else:
            fun=transforms.Compose([
                transforms.Resize(self.newshape[0]+32),
                transforms.CenterCrop(self.newshape)]
            )
            return fun(img)
        


def main():

    path = r'/Users/mac/Datas/animals/raw-img/butterfly/OIP--uJxQZUw1ibjIJDEuEXzpAHaEo.jpeg'

    obj=Image.open(path)

    letter=LetterResize((600,480),True)
    res=letter(obj)

    res.show()


if __name__ == '__main__':
    main()
